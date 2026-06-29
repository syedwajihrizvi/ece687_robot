#!/usr/bin/env python3
"""
stick_grabber_node.py  -- COMPLIANT REWRITE of the teammate's arm/gripper node.

WHY THE OLD VERSION GOT STUCK IN THE AIR
  The EP arm is a 2-link arm that moves the gripper in the robot's
  FORWARD-VERTICAL (x = reach, z = height) plane.  It CANNOT move the gripper
  horizontally side to side.  The old code computed the stick target in the
  HORIZONTAL world x-y plane, ran a 2R forward-kinematics on it, then published
  Vector3(x=err_x, z=err_y) -- feeding a horizontal y-error into the vertical z
  command.  The arm chased a horizontal error it can never null and parked in
  the air.  (The world->body rotation was also wrong.)

CORRECT DIVISION OF LABOUR (matches PDF: "define p to coincide with the stick tip")
  * HORIZONTAL alignment to the stick is the CHASSIS's job -> move_robot_node.py
    drives the gripper point p onto the stick.
  * THIS node only sets the arm HEIGHT/REACH (x,z) to a known grasp posture and
    works the gripper.  The sticks sit at a known height, so the grasp posture
    is a calibrated constant, not something computed from mocap.  No horizontal
    IK on the arm.

SEQUENCE (T2 "pick up a stick"):
  OPEN     : open the gripper
  SET_ARM  : command the arm to the grasp posture (x = reach, z = height)
  WAIT_ARM : wait until the arm reaches it (arm_position feedback, or timeout)
  CLOSE    : close the gripper on the stick
  DONE     : hold

Run (inside the container; align the robot first, or test by hand):
  python3 stick_grabber_node.py --ros-args \
    -p robot:=robot3 -p arm_x:=0.18 -p arm_z:=0.02
Tune arm_x (forward reach) and arm_z (height) until the open gripper sits
around the stick when the chassis has it lined up.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from geometry_msgs.msg import Point, PointStamped
from robomaster_msgs.action import GripperControl


# gripper target_state values (robomaster_ros): 0=pause, 1=open, 2=close
GRIPPER_OPEN = 1
GRIPPER_CLOSE = 2


class StickGrabberNode(Node):
    def __init__(self):
        super().__init__('stick_grabber_node')

        robot = self.declare_parameter('robot', 'robot3').value

        # Grasp posture in the arm's FORWARD-VERTICAL plane (metres).
        #   arm_x = forward reach,  arm_z = height.  CALIBRATE these to the stick.
        self.arm_x = self.declare_parameter('arm_x', 0.18).value
        self.arm_z = self.declare_parameter('arm_z', 0.02).value
        self.arm_tol = self.declare_parameter('arm_tol', 0.02).value     # arrival tol (m)
        self.arm_timeout = self.declare_parameter('arm_timeout', 4.0).value  # s
        self.grip_power = self.declare_parameter('grip_power', 0.7).value

        # ---- interfaces -----------------------------------------------------
        self.cb_group = ReentrantCallbackGroup()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # Absolute arm target in (x=reach, z=height); robomaster_ros arm driver.
        self.arm_pub = self.create_publisher(Point, f'/{robot}/target_arm_position', 10)
        # Arm feedback (x,z); used to detect when the posture is reached.
        self.arm_pos = None
        self.create_subscription(PointStamped, f'/{robot}/arm_position',
                                 self._on_arm_pos, qos)

        self.gripper = ActionClient(self, GripperControl, f'/{robot}/gripper',
                                    callback_group=self.cb_group)

        # ---- state machine --------------------------------------------------
        self.phase = 'OPEN'
        self._busy = False           # a gripper action is in flight
        self._arm_clock = None       # time we first commanded the arm

        self.get_logger().info('Connecting to gripper action server...')
        if not self.gripper.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper server not found; will retry in loop.')

        self.timer = self.create_timer(0.1, self.control_loop)   # 10 Hz
        self.get_logger().info('stick_grabber_node started. Phase OPEN.')

    # ------------------------------------------------------------------ I/O
    def _on_arm_pos(self, msg: PointStamped):
        self.arm_pos = (msg.point.x, msg.point.z)

    def _arm_reached(self) -> bool:
        if self.arm_pos is None:
            return False   # no feedback yet -> rely on timeout instead
        dx = self.arm_pos[0] - self.arm_x
        dz = self.arm_pos[1] - self.arm_z
        return (dx * dx + dz * dz) ** 0.5 < self.arm_tol

    # ------------------------------------------------------- state machine
    def control_loop(self):
        if self._busy:
            return

        if self.phase == 'OPEN':
            self._send_gripper(GRIPPER_OPEN)        # -> advances to SET_ARM

        elif self.phase == 'SET_ARM':
            self.arm_pub.publish(Point(x=float(self.arm_x), y=0.0, z=float(self.arm_z)))
            self._arm_clock = self.get_clock().now()
            self.get_logger().info(f'Arm -> grasp posture (x={self.arm_x}, z={self.arm_z})')
            self.phase = 'WAIT_ARM'

        elif self.phase == 'WAIT_ARM':
            # keep commanding the target so the driver holds it
            self.arm_pub.publish(Point(x=float(self.arm_x), y=0.0, z=float(self.arm_z)))
            elapsed = (self.get_clock().now() - self._arm_clock).nanoseconds * 1e-9
            if self._arm_reached() or elapsed > self.arm_timeout:
                self.get_logger().info('Arm in place -> CLOSE')
                self.phase = 'CLOSE'

        elif self.phase == 'CLOSE':
            self._send_gripper(GRIPPER_CLOSE)       # -> advances to DONE

        elif self.phase == 'DONE':
            pass   # hold posture; nothing to do

    # ------------------------------------------------------- gripper action
    def _send_gripper(self, state_value):
        if not self.gripper.server_is_ready():
            self.gripper.wait_for_server(timeout_sec=1.0)
            return  # try again next tick
        self._busy = True
        goal = GripperControl.Goal()
        goal.target_state = state_value
        goal.power = float(self.grip_power)
        self.get_logger().info(f'Gripper -> {"OPEN" if state_value == GRIPPER_OPEN else "CLOSE"}')
        self.gripper.send_goal_async(goal).add_done_callback(self._goal_response)

    def _goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Gripper goal rejected; retrying.')
            self._busy = False
            return
        handle.get_result_async().add_done_callback(self._goal_done)

    def _goal_done(self, _):
        if self.phase == 'OPEN':
            self.get_logger().info('Gripper open -> SET_ARM')
            self.phase = 'SET_ARM'
        elif self.phase == 'CLOSE':
            self.get_logger().info('Stick grasped -> DONE')
            self.phase = 'DONE'
        self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = StickGrabberNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted; stopping.')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
