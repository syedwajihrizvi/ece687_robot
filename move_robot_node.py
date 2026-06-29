#!/usr/bin/env python3
"""
move_robot_node.py  -- COMPLIANT REWRITE of the teammate's chassis controller.

Project requirement (ECE 486/687, Sec. 1.1):
  The robot is modeled as a UNICYCLE:
        x_dot = v cos(theta)
        y_dot = v sin(theta)
        theta_dot = omega
  i.e. the ONLY inputs are v (Twist.linear.x) and omega (Twist.angular.z).
  Twist.linear.y MUST stay 0, even though the EP mecanum base could strafe.

Control method (required): APPROXIMATE LINEARIZATION (Sec. 1.1, eqs. 2-3).
  Define a point p a distance l in front of the robot -- chosen to coincide
  with the tip of the stick:
        p = [x + l cos(theta);  y + l sin(theta)]
  Its motion is
        p_dot = R(theta) L(l) [v; omega],   L(l)=diag(1, l)            (eq. 2)
  so, once we design a controller for p_dot, we recover the unicycle inputs by
        [v; omega] = L^-1(l) R^T(theta) p_dot                          (eq. 3)
  We use the simple proportional point controller  p_dot = Kp (p_des - p).

This replaces the previous version, which published Twist.linear.y (holonomic
strafing) and therefore (a) violated the unicycle model and (b) never turned
toward the target -- it slid sideways instead.

Division of labour:
  * THIS node positions the robot so the gripper point p reaches the target
    (T1 navigate-to-stick, T3 navigate-to-puck, etc.).
  * stick_grabber_node.py handles the arm height + gripper (T2).

Run (inside the container, robot + vrpn_mocap up):
  python3 move_robot_node.py --ros-args \
    -p robot:=robot3 \
    -p target_topic:=/vrpn_mocap/hockey_sticks_1/pose \
    -p l:=0.20 -p v_max:=0.25
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped


def yaw_from_quat(q) -> float:
    """Extract the planar heading (yaw) from a quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class UnicycleApproxLinController(Node):
    def __init__(self):
        super().__init__('move_robot_node')

        # ---- parameters (defaults match the teammate's tested setup) --------
        robot = self.declare_parameter('robot', 'robot3').value
        self.declare_parameter('cmd_vel_topic', f'/{robot}/cmd_vel')
        self.declare_parameter('robot_pose_topic', f'/vrpn_mocap/dji_robot_3/pose')
        self.declare_parameter('target_topic', f'/vrpn_mocap/hockey_sticks_1/pose')

        # geometry / gains
        self.l = self.declare_parameter('l', 0.20).value          # offset of point p (m)
        self.Kp = self.declare_parameter('Kp', 1.0).value         # point P-gain
        self.v_max = self.declare_parameter('v_max', 0.25).value  # forward speed cap (m/s)
        self.w_max = self.declare_parameter('w_max', 1.5).value   # yaw-rate cap (rad/s)
        self.pos_tol = self.declare_parameter('pos_tol', 0.04).value  # arrival tol (m)

        # optional: rotate in place to a desired final heading after arriving
        self.align_final = self.declare_parameter('align_final_yaw', False).value
        self.goal_yaw = self.declare_parameter('goal_yaw', 0.0).value
        self.yaw_tol = self.declare_parameter('yaw_tol', 0.05).value

        # ---- state ----------------------------------------------------------
        self.robot = None     # np.array([x, y, theta])
        self.target = None    # np.array([x, y])
        self._arrived = False

        mocap_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(PoseStamped, self.get_parameter('robot_pose_topic').value,
                                 self._on_robot, mocap_qos)
        self.create_subscription(PoseStamped, self.get_parameter('target_topic').value,
                                 self._on_target, mocap_qos)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)

        self.timer = self.create_timer(0.05, self.control_loop)   # 20 Hz
        self.get_logger().info('move_robot_node started (unicycle + approx. linearization).')

    # ------------------------------------------------------------------ I/O
    def _on_robot(self, msg: PoseStamped):
        self.robot = np.array([msg.pose.position.x, msg.pose.position.y,
                               yaw_from_quat(msg.pose.orientation)])

    def _on_target(self, msg: PoseStamped):
        self.target = np.array([msg.pose.position.x, msg.pose.position.y])

    def control_point(self) -> np.ndarray:
        """Point p a distance l in front of the robot (= stick tip)."""
        x, y, th = self.robot
        return np.array([x + self.l * math.cos(th), y + self.l * math.sin(th)])

    def stop(self):
        self.cmd_pub.publish(Twist())   # all zeros

    # ------------------------------------------------------- control loop
    def control_loop(self):
        if self.robot is None or self.target is None:
            return

        p = self.control_point()
        dist = float(np.linalg.norm(self.target - p))

        # ---- arrival handling ------------------------------------------
        if dist < self.pos_tol:
            if self.align_final and not self._arrived:
                self.get_logger().info('Reached target -> aligning final heading.')
            self._arrived = True
            if self.align_final:
                self._rotate_to_yaw()
            else:
                self.stop()
            return
        self._arrived = False

        # ---- approximate linearization (eqs. 2-3) ----------------------
        # 1) design point velocity:  p_dot = Kp (p_des - p)
        p_dot = self.Kp * (self.target - p)

        # 2) saturate the *point* speed (keeps v, omega bounded together)
        speed = np.linalg.norm(p_dot)
        if speed > self.v_max:
            p_dot *= self.v_max / speed

        # 3) map back to unicycle inputs:  [v; omega] = L^-1 R^T p_dot
        x, y, th = self.robot
        c, s = math.cos(th), math.sin(th)
        v = c * p_dot[0] + s * p_dot[1]
        w = (-s * p_dot[0] + c * p_dot[1]) / self.l

        tw = Twist()
        tw.linear.x = float(v)
        tw.linear.y = 0.0                       # <-- unicycle: NEVER strafe
        tw.angular.z = float(np.clip(w, -self.w_max, self.w_max))
        self.cmd_pub.publish(tw)

    def _rotate_to_yaw(self):
        """Pure in-place rotation to goal_yaw (still unicycle: v=0)."""
        _, _, th = self.robot
        err = math.atan2(math.sin(self.goal_yaw - th), math.cos(self.goal_yaw - th))
        tw = Twist()
        if abs(err) > self.yaw_tol:
            tw.angular.z = float(np.clip(self.Kp * err, -self.w_max, self.w_max))
        self.cmd_pub.publish(tw)


def main(args=None):
    rclpy.init(args=args)
    node = UnicycleApproxLinController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
