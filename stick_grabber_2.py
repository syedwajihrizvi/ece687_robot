import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from geometry_msgs.msg import Vector3, PoseStamped
from sensor_msgs.msg import JointState
from robomaster_msgs.action import GripperControl
import numpy as np
import math

class StickGrabberNode(Node):
    def __init__(self):
        super().__init__('stick_grabber_node')
        self.get_logger().info("Stick Grabber Node initialized.")
        mocap_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=10
            )
        # TODO: Get the robot link dimensions
        # Constants for the robot
        self._a1 = 0.22
        self._a2 = 0.15
        self._error_treshold = 0.04 # 4 cm threshold
        self.sign = -1
        # Initialize variables to store the robot's joint states and the desired pose
        self.joint_positions = None
        self.joint_velocities = None
        self.desired_pose = None
        self.robot_pose = None
        self.robot_pose_orientation = None

        # Initialize robot phase between 0 and 1
        # 0 is INIT_OPEN_GRIPPER
        # 1 is MOVE_TO_STICK
        # 2 is CLOSE_GRIPPER
        # 3 is DONE
        self._currentPhase = 0

        self._gripper_action_running = False
        # Assume the gripper is always closed at the start of the program
        # This way we can ensure the gripper is open
        self.gripper_open = False

        # Create subscription to the stick target position
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.stick_mocap_callback, mocap_qos)  
        # Create subscription to joint states
        self.create_subscription(JointState, '/robot3/joint_states', self.joint_states_callback, mocap_qos)
        # Create subscription to robot base
        self.create_subscription(PoseStamped, '/vrpn_mocap/dji_robot_3/pose', self.robot_pose_callback, mocap_qos)

        # Action Client Configuration
        self.cb_group = ReentrantCallbackGroup()
        self.gripper_client = ActionClient(self, GripperControl, '/robot3/gripper', callback_group=self.cb_group)
    
        # Arm Publisher
        self.arm_pub = self.create_publisher(Vector3, '/robot3/cmd_arm', 10)

        self.get_logger().info("Connecting to gripper action server...")
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper server discovery timed out! Attempting sequence anyway...")

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Closed-Loop Feedback Grabber Node initialized. Awaiting MoCap data stream...")
    
    def _compute_forward_kinematics(self, robot_pose):
        """
        Compute the forward kinematics for the RR Manupilator
        We need the x,y position of the end effector from the Robot Pose
        """
        theta1, theta2 = robot_pose
        c1 = math.cos(theta1)
        c12 = math.cos(theta1 + theta2)
        s1 = math.sin(theta1)
        s12 = math.sin(theta1 + theta2)
        x = self._a1 * c1 + self._a2 * c12
        y = self._a1 * s1 + self._a2 * s12
        return np.array([x, y])
    
    def robot_pose_callback(self, msg):
        self.robot_pose = np.array([msg.pose.position.x, msg.pose.position.y])
        self.robot_pose_orientation = msg.pose.orientation

    def joint_states_callback(self, msg):
        """
        Callback functin to get the joint position and velocities
        """
        try:
            idx1 = msg.name.index('robot3/arm_1_joint')
            idx2 = msg.name.index('robot3/arm_2_joint')
            self.joint_positions = np.array([msg.position[idx1], msg.position[idx2]])
            self.joint_velocities = np.array([msg.velocity[idx1], msg.velocity[idx2]])
            print(f"SYED-DEBUG: Joint Positions: {self.joint_positions}, Joint Velocities: {self.joint_velocities}")
        except ValueError:
            self.get_logger().error("Arm joint names not found in JointState message!")

    def stick_mocap_callback(self, msg):
        """
        Callback function to get the stick's pose, which is also the desired pos for the 
        End Effector
        """
        if self.robot_pose is None:
            self.get_logger().warn("Robot pose not yet received. Waiting for MoCap data...")
            return
        print(f"SYED-DEBUG: Stick Pose: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.orientation}")
        stick_global = np.array([msg.pose.position.x, msg.pose.position.y])
        self.desired_pose = self._convert_to_robot_base_coodinates(stick_global)
        print(f"SYED-DEBUG: Desired Pose: {self.desired_pose}")

    def control_loop(self):
        # Return if the robot pose or desired pose is not yet received
        if self.joint_positions is None or self.joint_velocities is None or self.desired_pose is None:
            if self.joint_positions is None or self.joint_velocities is None:
                self.get_logger().warn("Robot joint states not yet received. Waiting for MoCap data...")
            if self.desired_pose is None:
                self.get_logger().warn("Stick pose not yet received. Waiting for MoCap data...")
            return
        
        # Ensure the gripper is open before moving towards the desired pose
        if self._currentPhase == 0:
            self.get_logger().info("Gripper is closed. Opening gripper before moving towards the desired pose.")
            self._send_gripper_goal(1) # 1 = OPEN
            return

        pos_error = self._compute_error_vectors()
        pos_error_norm = np.linalg.norm(pos_error)
        # print(f"SYED-DEBUG: Pos Error: {pos_error}")
        # msg = Vector3(
        #     x = float(0.35),
        #     y = 0.0,
        #     z = float(self.sign*0.46)
        # )
        # self.sign *= -1
        # self.arm_pub.publish(msg)
        # pos_error = self._compute_error_vectors()
        # pos_error_norm = np.linalg.norm(pos_error)
        print(f"SYED-DEBUG: Current Phase: {self._currentPhase}")
        if self._currentPhase == 1:
            if pos_error_norm < self._error_treshold:
                self.get_logger().info("End Effector is within the error treshold. Stopping the robot and closing the gripper")
                # Manupilator should not be moving since it is at the desired position
                stop_msg = Vector3(x=0.0, y=0.0, z=0.0)
                self.arm_pub.publish(stop_msg)
                self._currentPhase = 2
            else:
                self.get_logger().info("End Effector is outside the error treshold. Moving towards the desired pose")
                print(f"SYED-DEBUG: Sending positions: {pos_error}")
                msg = Vector3(
                    x = float(pos_error[0]),
                    y = 0.0,
                    z = float(pos_error[1])
                )
                self.arm_pub.publish(msg)
                print(f"SYED-DEBUG: Published the command for {msg}")
        elif self._currentPhase == 2:
            self.get_logger().info("End Effector is within the error treshold. Closing the gripper")
            if not self.gripper_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("Gripper server discovery timed out! Attempting sequence anyway...")
            if not self._gripper_action_running:
                self._send_gripper_goal(2)
        elif self._currentPhase == 3:
            self.get_logger().info("Gripper is closed. Task completed.")
            # Stop the robot
            stop_msg = Vector3(x=0.0, y=0.0, z=0.0)
            self.arm_pub.publish(stop_msg)
        else:
            self.get_logger().error(f"Unknown phase: {self._currentPhase}. Resetting to phase 0.")
            self._currentPhase = 0


    def _send_gripper_goal(self, state_value):
        self._gripper_action_running = True
        goal_msg = GripperControl.Goal()
        goal_msg.target_state = state_value
        print(f"SYED-DEBUG: Sending gripper goal with state value: {state_value}")
        send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._gripper_response_callback)

    def _gripper_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper action goal rejected!")
            self._gripper_action_running = False
            return

        self.get_logger().info("Gripper action goal accepted. Waiting for result...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._gripper_result_callback)

    def _gripper_result_callback(self, future):
        result = future.result()
        self.get_logger().info(f"Gripper action result: {result}")
        if self._currentPhase == 0:
            self.get_logger().info("Gripper opened successfully. Moving to phase 1: MOVE_TO_STICK")
            self._currentPhase = 1
        elif self._currentPhase == 2:
            self.get_logger().info("Gripper closed successfully. Task completed. Moving to phase 3: DONE")
            self._currentPhase = 3
        self._gripper_action_running = False

    def _compute_error_vectors(self):
        """
        Compute the error vectors between the desired pose and the current pose
        position error and velocity error. The target velocity is assumed to be zero since we
        want to stop at the desired position
        """
        current_ee_pose = self._compute_forward_kinematics(self.joint_positions)
        position_error = self.desired_pose - current_ee_pose
        return position_error
    
    # convert everything into local coordinates of robot
    def _convert_to_robot_base_coodinates(self, global_pos):
        translation_error = global_pos - self.robot_pose

        q = self.robot_pose_orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        local_x = translation_error[0] * math.sin(yaw) - translation_error[1] * math.cos(yaw)
        local_y = translation_error[0] * math.cos(yaw) + translation_error[1] * math.sin(yaw)
        print(f"SYED-DEBUG: Local Coordinates: {local_x}, {local_y}")
        return np.array([local_x, local_y])

def main(args=None):
    rclpy.init(args=args)
    node = StickGrabberNode() # Make sure this matches your exact class name

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt caught. Stopping arm...")
    finally:
        # 1. Clear out movement while the publisher handle is fully alive and valid
        try:
            stop_msg = Vector3(x=0.0, y=0.0, z=0.0)
            node.arm_pub.publish(stop_msg)
        except Exception as e:
            print(f"Could not send safety stop: {e}")

        # 2. Shutdown the executor threads safely
        executor.shutdown()
        node.destroy_node()
        
        # 3. Only shutdown rclpy if context wasn't wiped out already
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

"""
Creating ros2 package steps
cd /root/ros2_ws/src
ros2 pkg create --build-type ament_python robot_controller --dependencies rclpy geometry_msgs sensor_msgs robomaster_msgs
cd <your_package_name>/<your_package_name>/
touch stick_grabber_node.py
chmod +x stick_grabber_node.py

in setup.py
entry_points={
        'console_scripts': [
            'stick_grabber_node = your_package_name.stick_grabber_node:main',
        ],
    },

cd /root/ros2_ws
colcon build --packages-select <your_package_name> --symlink-install
source install/setup.bash
ros2 run <your_package_name> stick_grabber_node
"""