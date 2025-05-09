#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.duration import Duration
# Add time module for sleep functionality
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, Twist, Pose, TwistStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from pushing_interfaces.action import ApplyPush, MoveRobotTo
from pushing_interfaces.srv import SetPushingEnv
from std_srvs.srv import SetBool
from nav_msgs.msg import Path, Odometry

class PushingController(Node):
    def __init__(self):
        super().__init__('pushing_controller')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('min_turning_radius', 0.47),
                ('safe_distance', 0.15),
                ('robot.distance_success_threshold', 0.02),
                ('robot.x_gain', 1.0),
                ('robot.y_gain', 1.0),
                # ... Add all other parameters
            ]
        )
        
        # State variables
        self.robot_pose = Pose()
        self.object_pose = Pose()
        self.paused = False
        self.current_goal = None
        self.pushing_trajectory = None
        self.trajectory_counter = 0
        
        # Subscribers
        self.robot_pose_sub = self.create_subscription(
            Odometry, 'pushing/robot_pose', self.robot_pose_cb, 10)
        self.object_pose_sub = self.create_subscription(
            PoseStamped, 'pushing/object_pose', self.object_pose_cb, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'pushing/cmd_vel', 10)
        self.ref_path_pub = self.create_publisher(Path, 'pushing/ref_path', 10)
        self.pred_path_pub = self.create_publisher(Path, 'pushing/pred_path', 10)
        
        # Services
        self.pause_srv = self.create_service(
            SetBool, 'pushing/pause', self.pause_cb)
        self.env_srv = self.create_service(
            SetPushingEnv, 'pushing/controller_set_env', self.set_env_cb)
        
        # Action Servers
        self.move_robot_action = ActionServer(
            self, MoveRobotTo, 'pushing/move_robot_to',
            execute_callback=self.execute_move_robot_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb)
            
        self.apply_push_action = ActionServer(
            self, ApplyPush, 'pushing/apply_push',
            execute_callback=self.execute_apply_push_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb)
        
        # Timer
        self.control_timer = self.create_timer(0.1, self.control_cycle)
        
        # MPC Initialization (Placeholder - integrate actual MPC implementation)
        self.init_mpc()

    def init_mpc(self):
        """Initialize MPC parameters and state"""
        self.robot_state = np.zeros(5)
        self.object_state = np.zeros(3)
        self.pushing_state = np.zeros(3)
        
        # Initialize MPC weights and parameters from ROS params
        self.set_weights()

    def set_weights(self):
        """Update MPC weights from parameters"""
        params = self.get_parameters([
            'robot.x_gain', 'robot.y_gain', 
            # ... all other parameters
        ])
        
        # Update MPC weights using parameter values
        # ...

    def control_cycle(self):
        """Main control loop"""
        # Change debug to info so messages are visible by default
        self.get_logger().info('Control cycle triggered')
        
        if self.paused:
            self.get_logger().debug('Control cycle paused')
            self.publish_zero_vel()
            return
            
        if not self.current_goal:
            self.get_logger().debug('No current goal - skipping control cycle')
            self.publish_zero_vel()
            return
                        
        self.get_logger().debug(f'Executing control for goal type: {type(self.current_goal).__name__}')
                    
        if isinstance(self.current_goal, MoveRobotTo.Goal):
            self.execute_robot_control()
        elif isinstance(self.current_goal, ApplyPush.Goal):
            self.execute_push_control()
        else:
            self.get_logger().warn(f'Unknown goal type: {type(self.current_goal).__name__}')

    def execute_robot_control(self):
        """Handle robot movement control"""
        # MPC calculations
        self.update_robot_state()
        cmd_vel = self.calculate_robot_control()
        
        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Check completion
        if self.check_robot_goal_completion():
            self.current_goal.succeed()
            self.current_goal = None

    def execute_push_control(self):
        """Handle object pushing control"""
        # MPC calculations
        self.update_push_state()
        cmd_vel = self.calculate_push_control()
        
        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Check completion
        if self.check_push_goal_completion():
            self.current_goal.succeed()
            self.current_goal = None
            self.trajectory_counter = 0

    def update_robot_state(self):
        """Update robot state from subscriptions"""
        # Convert pose to state vector [x, y, theta, v, omega]
        q = self.robot_pose.orientation
        theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.robot_state = np.array([
            self.robot_pose.position.x,
            self.robot_pose.position.y,
            theta,
            self.robot_state[3],  # Maintain current velocity
            self.robot_state[4]   # Maintain current angular velocity
        ])

    def calculate_robot_control(self):
        """Calculate robot control using MPC"""
        # Placeholder for actual MPC calculation
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.1
        cmd_vel.angular.z = 0.05
        return cmd_vel

    def check_robot_goal_completion(self):
        """Check if robot has reached target"""
        distance = np.linalg.norm(
            self.robot_state[:2] - 
            np.array([self.current_goal.target.position.x,
                      self.current_goal.target.position.y]))
        return distance < self.get_parameter(
            'robot.distance_success_threshold').value

    def update_push_state(self):
        """Update push state from subscriptions"""
        # Convert pose to state vector [x, y, theta]
        q = self.object_pose.orientation
        theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.pushing_state = np.array([
            self.object_pose.position.x,
            self.object_pose.position.y,
            theta
        ])

    def calculate_push_control(self):
        """Calculate push control using MPC"""
        # Placeholder for actual MPC calculation
        cmd_vel = Twist()
        if self.trajectory_counter < len(self.pushing_trajectory):
            sample = self.pushing_trajectory[self.trajectory_counter]
            cmd_vel.linear.x = sample.twist.linear.x
            cmd_vel.angular.z = sample.twist.angular.z
            self.trajectory_counter += 1
        else:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        return cmd_vel

    def check_push_goal_completion(self):
        """Check if push goal has been reached"""
        if self.trajectory_counter >= len(self.pushing_trajectory):
            return True
        return False

    # Callback functions
    def robot_pose_cb(self, msg):
        self.robot_pose = msg.pose

    def object_pose_cb(self, msg):
        self.object_pose = msg.pose

    def pause_cb(self, request, response):
        self.paused = request.data
        response.success = True
        response.message = "Paused" if self.paused else "Resumed"
        return response

    def set_env_cb(self, request, response):
        # Handle environment configuration
        response.success = True
        return response

    # Action server callbacks
    def goal_cb(self, goal_request):
        self.get_logger().info('Received new goal request')
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_move_robot_cb(self, goal_handle):
        self.current_goal = goal_handle
        feedback_msg = MoveRobotTo.Feedback()
        
        # while not self.check_robot_goal_completion():
        #     if goal_handle.is_cancel_requested:
        #         goal_handle.canceled()
        #         self.current_goal = None
        #         return MoveRobotTo.Result()
            
        #     # Publish feedback
        #     feedback_msg.current_pose = self.robot_pose
        #     goal_handle.publish_feedback(feedback_msg)
            
        #     await rclpy.sleep(0.1)
        
        result = MoveRobotTo.Result()
        result.success = True
        return result

    async def execute_apply_push_cb(self, goal_handle):
        self.current_goal = goal_handle
        self.pushing_trajectory = goal_handle.request.action.trajectory
        self.trajectory_counter = 0  # Reset counter when starting new trajectory
        feedback_msg = ApplyPush.Feedback()
        # Create a Path message for feedback
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Let the control_cycle handle the execution
        # Just wait for completion or cancellation
        while not self.check_push_goal_completion():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.current_goal = None
                return ApplyPush.Result()
            
            # Create feedback about the current trajectory point
            if self.trajectory_counter < len(self.pushing_trajectory):
                current_sample = self.pushing_trajectory[self.trajectory_counter]
                pose_stamped = PoseStamped()
                pose_stamped.header = path_msg.header
                pose_stamped.pose = current_sample.pose
                path_msg.poses = [pose_stamped]
                feedback_msg.current_ref = path_msg
                
                # Publish feedback
                goal_handle.publish_feedback(feedback_msg)
            
# Replace incorrect sleep with rate-based sleep
            await self.create_rate(10).sleep()  # Sleep at 10Hz (0.1 seconds)
        
        result = ApplyPush.Result()
        result.success = True
        return result

    def publish_zero_vel(self):
        zero_vel = Twist()
        self.cmd_vel_pub.publish(zero_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = PushingController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()