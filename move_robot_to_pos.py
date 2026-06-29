import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Twist

class MecanumBaseController(Node):
    def __init__(self):
        super().__init__('mecanum_base_controller')

        mocap_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot3/cmd_vel', 10)
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.stick_mocap_callback, mocap_qos) 

        # Storage
        self.robot_pose = None
        self.robot_orientation = None
        self.stick_pose = None
        self.stick_orientation = None

        # Subscriptions
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.stick_mocap_callback, mocap_qos) 
        self.create_subscription(PoseStamped, '/vrpn_mocap/dji_robot_3/pose', self.robot_pose_callback, mocap_qos)
        
        # Controller Parameters
        self.Kp_linear = 1.5
        self.Kp_angular = 2.0
        self.MAX_SPEED = 0.3
        self.MAX_ANGULAR_SPEED = 1.0
        self.TARGET_FORWARD_OFFSET = 0.15
        self.LINEAR_THRESHOLD = 0.05
        self.ANGULAR_THRESHOLD = 0.05

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Mecanum base controller node has been started.")

    def robot_pose_callback(self, msg):
        print("SYED-DEBUG: Running callback for robot pose")
        self.robot_pose = msg.pose.position
        self.robot_orientation = msg.pose.orientation 
        print("SYED-DEBUG: Robot pose updated:", self.robot_pose)
        print("SYED-DEBUG: Robot orientation updated:", self.robot_orientation)

    def stick_mocap_callback(self, msg):
        print("SYED-DEBUG: Running callback for stick pose")
        self.stick_pose = msg.pose.position
        self.stick_orientation = msg.pose.orientation
        print("SYED-DEBUG: Stick pose updated:", self.stick_pose)
        print("SYED-DEBUG: Stick orientation updated:", self.stick_orientation)

    def _compute_local_errors(self):
        if self.robot_pose is None or self.stick_pose is None or self.stick_orientation is None:
            print(f"SYED-DEBUG: {self.robot_pose}, {self.stick_pose}, {self.stick_orientation}")
            return None
        
        dx_global = self.stick_pose.x - self.robot_pose.x
        dy_global = self.stick_pose.y - self.robot_pose.y

        q_r = self.robot_orientation
        siny_cosp_r = 2 * (q_r.w * q_r.z + q_r.x * q_r.y)
        cosy_cosp_r = 1 - 2 * (q_r.y * q_r.y + q_r.z * q_r.z)
        robot_yaw = math.atan2(siny_cosp_r, cosy_cosp_r)

        q_s = self.stick_orientation
        siny_cosp_s = 2 * (q_s.w * q_s.z + q_s.x * q_s.y)
        cosy_cosp_s = 1 - 2 * (q_s.y * q_s.y + q_s.z * q_s.z)
        stick_yaw = math.atan2(siny_cosp_s, cosy_cosp_s)

        dx_local = dx_global * math.cos(robot_yaw) + dy_global * math.sin(robot_yaw)
        dy_local = -dx_global * math.sin(robot_yaw) + dy_global * math.cos(robot_yaw)

        error_x = dx_local - self.TARGET_FORWARD_OFFSET
        error_y = dy_local
        error_yaw = stick_yaw - robot_yaw

        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))

        return error_x, error_y, error_yaw

    def control_loop(self):
            print("SYED-DEBUG: Control loop is running")
            errors = self._compute_local_errors()
            if errors is None:
                print("SYED-DEBUG: Error is none")
                return
            error_x, error_y, error_yaw = errors
            if abs(error_x) < self.LINEAR_THRESHOLD and abs(error_y) < self.LINEAR_THRESHOLD and abs(error_yaw) < self.ANGULAR_THRESHOLD:
                print("SYED-DEBUG: Threshold reached")
                self.stop_chassis()
                return
            
            vx = error_x * self.Kp_linear
            vy = error_y * self.Kp_linear
            wz = error_yaw * self.Kp_angular

            vx = np.clip(vx, -self.MAX_SPEED, self.MAX_SPEED)
            vy = np.clip(vy, -self.MAX_SPEED, self.MAX_SPEED)
            wz = np.clip(wz, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)       

            # Direct, clean assignment matching the computed frame velocities
            twist_msg = Twist()
            twist_msg.linear.x = float(vx)   # No inversion!
            twist_msg.linear.y = float(vy)   # No inversion!
            twist_msg.angular.z = float(wz)
            
            print(f"SYED-DEBUG: Computed velocities - vx: {vx:.2f}, vy: {vy:.2f}, wz: {wz:.4f}")
            print(f"SYED-DEBUG: Publishing Twist - Linear: ({twist_msg.linear.x:.2f}, {twist_msg.linear.y:.2f}), Angular: {twist_msg.angular.z:.4f}")
            self.cmd_vel_pub.publish(twist_msg)

    def stop_chassis(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.linear.y = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)
        self.get_logger().info("Chassis stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = MecanumBaseController()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_chassis()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()