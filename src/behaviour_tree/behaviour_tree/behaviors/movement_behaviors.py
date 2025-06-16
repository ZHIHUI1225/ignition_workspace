#!/usr/bin/env python3
"""
Movement behavior classes for the behavior tree system.
Contains robot movement and navigation behaviors with MPC-based control.
"""

import py_trees
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
import math
import time
import threading
import tf_transformations as tf
import casadi as ca
import numpy as np
import math
import time
import threading
import tf_transformations as tf
import casadi as ca
import numpy as np


class MobileRobotMPC:
    """MPC controller for robot approach - based on State_switch.py implementation"""
    def __init__(self):
        # MPC parameters
        self.N = 10         # Extended prediction horizon for smoother approach
        self.dt = 0.1        # Time step
        self.wx = 2.0        # Increased position error weight for better position convergence
        self.wtheta = 1.5    # Increased orientation error weight for better alignment
        self.wu = 0.08       # Slightly reduced control effort weight for more responsive control
        
        # Control constraints - further reduced max velocity for even slower approach
        self.v_max = 0.08     # m/s 
        self.w_max = 0.6      # rad/s
        
        # State and control dimensions
        self.nx = 3          # [x, y, theta]
        self.nu = 2          # [v, w]
        
        # Initialize CasADi optimizer
        self.setup_optimizer()
        
    def setup_optimizer(self):
        # Define symbolic variables
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N+1)
        self.U = self.opti.variable(self.nu, self.N)
        
        # Parameters (initial state and reference)
        self.x0 = self.opti.parameter(self.nx)
        self.x_ref = self.opti.parameter(self.nx)
        
        # Cost function
        cost = 0
        for k in range(self.N):
            # Tracking cost with increasing weights as we approach the end
            # Stronger progression factor for more aggressive convergence
            progress_factor = 1.0 + 2.0 * (k + 1) / self.N  # Increases from 1.0 to 3.0
            
            # Position cost - higher weight for xy position tracking
            pos_error = self.X[:2, k] - self.x_ref[:2]
            cost += progress_factor * self.wx * ca.sumsqr(pos_error)
            
            # Orientation cost with angle normalization to handle wraparound
            theta_error = self.X[2, k] - self.x_ref[2]
            # Normalize angle difference to [-pi, pi]
            theta_error = ca.fmod(theta_error + ca.pi, 2*ca.pi) - ca.pi
            cost += progress_factor * self.wtheta * theta_error**2
            
            # Special emphasis on final portion of trajectory
            if k >= self.N - 5:  # Last 5 steps
                # Extra emphasis on final approach
                cost += 1.5 * self.wx * ca.sumsqr(pos_error)
                cost += 2.0 * self.wtheta * theta_error**2
            
            # Control effort cost with smoother transitions
            if k > 0:
                # Penalize control changes for smoother motion
                control_change = self.U[:, k] - self.U[:, k-1]
                cost += 0.1 * ca.sumsqr(control_change)
            
            # Base control effort penalty
            cost += self.wu * ca.sumsqr(self.U[:, k])
        
        # Terminal cost - much stronger to ensure convergence at endpoint
        terminal_pos_error = self.X[:2, self.N] - self.x_ref[:2]
        cost += 10.0 * self.wx * ca.sumsqr(terminal_pos_error)
        
        # Terminal orientation with normalization for angle wraparound
        terminal_theta_error = self.X[2, self.N] - self.x_ref[2]
        terminal_theta_error = ca.fmod(terminal_theta_error + ca.pi, 2*ca.pi) - ca.pi
        cost += 30.0 * self.wtheta * terminal_theta_error**2
        self.opti.minimize(cost)
        
        # Dynamics constraints
        for k in range(self.N):
            x_next = self.X[:, k] + self.robot_model(self.X[:, k], self.U[:, k]) * self.dt
            self.opti.subject_to(self.X[:, k+1] == x_next)
        
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # Control constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max, self.U[0, :], self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.w_max, self.U[1, :], self.w_max))
        
        # Strict terminal constraints to ensure convergence and smooth stopping
        # Terminal velocity constraints - must approach zero at the end
        self.opti.subject_to(self.U[0, -1] <= 0.0005)  # Final linear velocity virtually zero
        self.opti.subject_to(self.U[0, -1] >= 0.0)     # No negative velocity at end
        self.opti.subject_to(self.U[1, -1] <= 0.0005)  # Final angular velocity virtually zero
        self.opti.subject_to(self.U[1, -1] >= -0.0005) # Final angular velocity virtually zero
        
        # Smooth deceleration in last few steps
        for k in range(self.N-3, self.N):
            # Progressive velocity reduction for final steps
            max_vel_factor = (self.N - k) / 4.0  # Ranges from 0.75 to 0.25
            self.opti.subject_to(self.U[0, k] <= self.v_max * max_vel_factor)
        
        # Solver settings with improved convergence parameters
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.tol': 1e-5,           # Even tighter tolerance
            'ipopt.acceptable_tol': 1e-4, # More precise solution
            'ipopt.max_iter': 200,       # More iterations allowed
            'ipopt.warm_start_init_point': 'yes' # Use warm starting for stability
        }
        self.opti.solver('ipopt', opts)
        
    def robot_model(self, x, u):
        # Differential drive kinematics
        dx = ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1]
        )
        return dx
        
    def update_control(self, current_state, target_state):
        # Check how close we are to the target
        dist_to_target = np.sqrt((current_state[0] - target_state[0])**2 + 
                                (current_state[1] - target_state[1])**2)
        
        # Check orientation alignment
        angle_diff = abs((current_state[2] - target_state[2] + np.pi) % (2 * np.pi) - np.pi)
        
        # If we're very close to the target and well-aligned, stop completely
        if dist_to_target < 0.015 and angle_diff < 0.05:  # 1.5cm and ~3 degrees
            return np.array([0.0, 0.0])  # Stop completely
            
        # If we're close but not perfectly aligned, prioritize orientation
        elif dist_to_target < 0.03 and angle_diff > 0.05:
            # Just rotate to align with target, very slowly
            return np.array([0.0, 0.1 * np.sign(target_state[2] - current_state[2])])
        
        # Set initial state and reference
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.x_ref, target_state)
        
        # Solve optimization problem
        try:
            sol = self.opti.solve()
            x_opt = sol.value(self.X)
            u_opt = sol.value(self.U)
            return u_opt[:, 0]  # Return first control input
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None


class ApproachObject(py_trees.behaviour.Behaviour):
    """
    Approach Object behavior - integrates with State_switch approaching_target logic.
    Uses MPC controller to make the robot approach the parcel based on the logic from State_switch.py.
    """

    def __init__(self, name="ApproachObject", robot_namespace="turtlebot0", approach_distance=0.12):
        """
        Initialize the ApproachObject behavior.
        
        Args:
            name: Name of the behavior node
            robot_namespace: The robot namespace (e.g., 'turtlebot0', 'turtlebot1')
            approach_distance: Distance to maintain from the parcel (default 0.12m)
        """
        super(ApproachObject, self).__init__(name)
        self.robot_namespace = robot_namespace
        self.approach_distance = approach_distance
        
        # Extract namespace number for topic subscriptions
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        
        # Pose storage
        self.robot_pose = None
        self.parcel_pose = None
        
        # Setup blackboard access for current_parcel_index with namespace
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/pushing_estimated_time", 
            access=py_trees.common.Access.WRITE
        )
        
        # Set default pushing estimated time (45 seconds)
        setattr(self.blackboard, f"{robot_namespace}/pushing_estimated_time", 45.0)
        
        # State variables
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target_state = np.array([0.0, 0.0, 0.0])   # [x, y, theta]
        self.approaching_target = False
        
        # ROS2 components (will be initialized in setup)
        self.ros_node = None
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        self.cmd_vel_pub = None
        
        # MPC controller
        self.mpc = MobileRobotMPC()
        
        # Control loop timer
        self.control_timer = None
        self.dt = 0.1  # 0.1s timer period for MPC control (10Hz)
        self.control_active = False
        
        # Threading lock for state protection
        self.lock = threading.Lock()
        
    def extract_namespace_number(self, namespace):
        """Extract numerical index from namespace string using regex"""
        import re
        match = re.search(r'\d+', namespace)
        return int(match.group()) if match else 0

    def setup(self, **kwargs):
        """Setup ROS connections"""
        try:
            # Get or create ROS node
            if 'node' in kwargs:
                self.ros_node = kwargs['node']
            else:
                if not rclpy.ok():
                    rclpy.init()
                
                class ApproachObjectNode(Node):
                    def __init__(self):
                        super().__init__(f'approach_object_{self.robot_namespace}')
                
                self.ros_node = ApproachObjectNode()
            
            # Create command velocity publisher
            self.cmd_vel_pub = self.ros_node.create_publisher(
                Twist, f'/turtlebot{self.namespace_number}/cmd_vel', 10)
            
            # Subscribe to robot pose (Odometry)
            self.robot_pose_sub = self.ros_node.create_subscription(
                Odometry, f'/turtlebot{self.namespace_number}/odom_map',
                self.robot_pose_callback, 10)
            
            # Initialize parcel subscription as None - will be set up in update() method
            self.parcel_pose_sub = None
            self._last_parcel_index = None
            
            self.ros_node.get_logger().info(
                f'ApproachObject setup complete for {self.robot_namespace}')
            return True
            
        except Exception as e:
            print(f"ApproachObject setup failed: {e}")
            return False

    def setup_parcel_subscription(self, parcel_index):
        """Set up subscription to the correct parcel topic"""
        if self.ros_node is None:
            return False
            
        # Clean up existing subscription if it exists
        if self.parcel_pose_sub is not None:
            self.ros_node.destroy_subscription(self.parcel_pose_sub)
            self.parcel_pose_sub = None
        
        # Create new subscription to the correct parcel topic
        parcel_topic = f'/parcel{parcel_index}/pose'
        self.parcel_pose_sub = self.ros_node.create_subscription(
            PoseStamped, parcel_topic, self.parcel_pose_callback, 10)
        
        self.ros_node.get_logger().info(
            f'[{self.name}] Subscribed to {parcel_topic}')
        return True

    def robot_pose_callback(self, msg):
        """Callback for robot pose updates (Odometry message)"""
        with self.lock:
            self.robot_pose = msg.pose.pose
            # Update current state for MPC
            self.current_state = np.array([
                self.robot_pose.position.x,
                self.robot_pose.position.y,
                self.quaternion_to_yaw(self.robot_pose.orientation)
            ])
            # if self.ros_node:
            #     self.ros_node.get_logger().info(f'[{self.name}] DEBUG: Received robot pose: x={self.robot_pose.position.x:.3f}, y={self.robot_pose.position.y:.3f}')

    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates (PoseStamped message)"""
        with self.lock:
            self.parcel_pose = msg.pose
            # if self.ros_node:
                # self.ros_node.get_logger().info(f'[{self.name}] DEBUG: Received parcel pose: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}')

    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        # Handle different pose message types
        if hasattr(pose1, 'pose'):
            pose1 = pose1.pose
        if hasattr(pose2, 'pose'):
            pose2 = pose2.pose
            
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx**2 + dy**2)

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        quat_list = [x, y, z, w]
        euler = tf.euler_from_quaternion(quat_list)
        return euler[2]

    def control_timer_callback(self):
        """ROS timer callback for MPC control loop - runs at 10Hz"""
        # Add timing debug info
        current_time = time.time()
        if not hasattr(self, '_last_timer_time'):
            self._last_timer_time = current_time
            self._timer_call_count = 0
        
        time_interval = current_time - self._last_timer_time
        self._timer_call_count += 1
        self._last_timer_time = current_time
        
        # Log timing every 50 calls to avoid spam but verify frequency
        if self._timer_call_count % 50 == 1:
            frequency = 1.0 / time_interval if time_interval > 0 else 0
            print(f"[{self.name}] Timer callback #{self._timer_call_count}, "
                  f"interval: {time_interval:.3f}s, frequency: {frequency:.1f}Hz")
        
        # Only execute control loop if approach is active
        if self.control_active and self.approaching_target:
            # print(f"[{self.name}] DEBUG: Timer executing control loop - control_active={self.control_active}, approaching_target={self.approaching_target}")
            self.control_loop()
        else:
            # Stop the robot when not actively approaching
            if self.cmd_vel_pub:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)

    def control_loop(self):
        """Main MPC control loop for approach behavior"""
        with self.lock:
            # Check if we have the necessary pose data
            if self.robot_pose is None or self.parcel_pose is None:
                return
            
            # Calculate distance to parcel
            distance_to_parcel = self.calculate_distance(self.robot_pose, self.parcel_pose)
            
            # Check if we should continue approaching
            if distance_to_parcel > self.approach_distance and self.approaching_target:
                # Compute target state following State_switch.py logic
                self.target_state = np.array([
                    self.parcel_pose.position.x,
                    self.parcel_pose.position.y,
                    self.quaternion_to_yaw(self.parcel_pose.orientation)
                ])
                
                # Get optimal direction and apply offset
                optimal_direction = self.get_direction(
                    self.current_state[2],
                    self.target_state[2]
                )
                self.target_state[2] = optimal_direction
                self.target_state[0] = self.target_state[0] - (self.approach_distance - 0.2)* math.cos(optimal_direction)
                self.target_state[1] = self.target_state[1] - (self.approach_distance-0.2) * math.sin(optimal_direction)
                
                # Generate and apply control using MPC
                u = self.mpc.update_control(self.current_state, self.target_state)
                
                if u is not None and self.cmd_vel_pub:
                    cmd = Twist()
                    cmd.linear.x = float(u[0])
                    cmd.angular.z = float(u[1])
                    self.cmd_vel_pub.publish(cmd)
                    
                    # Debug output for control commands
                    if self._timer_call_count % 20 == 1:  # Log every 2 seconds
                        print(f"[{self.name}] MPC control: v={cmd.linear.x:.3f}, ω={cmd.angular.z:.3f}, "
                              f"dist={distance_to_parcel:.3f}m")
            else:
                # Stop the robot if we've reached the target or should no longer approach
                if self.cmd_vel_pub:
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    self.cmd_vel_pub.publish(cmd)
                
                # If we reached the target, mark approach as complete
                if distance_to_parcel <= self.approach_distance:
                    self.approaching_target = False
                    self.control_active = False
                    print(f"[{self.name}] Target reached, approach complete. Distance: {distance_to_parcel:.3f}m")

    def get_direction(self, robot_theta, parcel_theta):
        """Get optimal approach direction - from State_switch.py"""
        # Normalize input angles to [-π, π]
        def normalize(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        
        robot_theta = normalize(robot_theta)
        parcel_theta = normalize(parcel_theta)
        
        # Generate candidate angles and normalize
        candidates = [
            parcel_theta,
            normalize(parcel_theta + np.pi/2),  # Turn right 90 degrees
            normalize(parcel_theta - np.pi/2),  # Turn left 90 degrees
            normalize(parcel_theta + np.pi),    # 180 degrees
        ]
        
        # Calculate minimum circular angle difference
        diffs = [abs(normalize(c - robot_theta)) for c in candidates]
        
        index_min = np.argmin(diffs)
        return candidates[index_min]

    def initialise(self):
        """Initialize the behavior when it starts running"""
        self.approaching_target = False
        self.control_active = False
        self.feedback_message = "Initializing approach behavior"
        
        # Create and start ROS timer for control loop at 10Hz (0.1s)
        if self.ros_node:
            # Clean up any existing timer first
            if self.control_timer:
                self.control_timer.cancel()
                self.control_timer = None
                print(f"[{self.name}] DEBUG: Cancelled existing ROS timer")
            
            # Create new timer for 10Hz control frequency
            self.control_timer = self.ros_node.create_timer(self.dt, self.control_timer_callback)
            print(f"[{self.name}] DEBUG: Created ROS timer for control loop at {1/self.dt:.1f} Hz (every {self.dt}s)")
        else:
            print(f"[{self.name}] WARNING: No ROS node available, cannot create control timer")
        
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor

    def update(self):
        """
        Main update method - behavior tree logic only, control runs via timer
        """
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor

        # Get current parcel index and set up subscription if needed
        current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
        
        # Set up parcel subscription if not done yet or if parcel index changed
        if (self.parcel_pose_sub is None or 
            self._last_parcel_index != current_parcel_index):
            
            success = self.setup_parcel_subscription(current_parcel_index)
            if success:
                self._last_parcel_index = current_parcel_index
                print(f"[{self.name}] Now tracking parcel{current_parcel_index}")

        with self.lock:
            # Check if we have the necessary pose data
            if self.robot_pose is None or self.parcel_pose is None:
                if self.ros_node:
                    self.ros_node.get_logger().warn(f'[{self.name}] DEBUG: Missing pose data - robot_pose: {self.robot_pose is not None}, parcel_pose: {self.parcel_pose is not None}')
                self.feedback_message = "Waiting for pose data..."
                return py_trees.common.Status.RUNNING

            # Calculate distance to parcel
            distance_to_parcel = self.calculate_distance(self.robot_pose, self.parcel_pose)
            
            # Check if we need to approach (distance > approach_distance)
            if distance_to_parcel > self.approach_distance:
                if not self.approaching_target:
                    # Start approaching
                    self.approaching_target = True
                    self.control_active = True
                    print(f"[{self.name}] Starting approach to parcel{current_parcel_index}, distance: {distance_to_parcel:.3f}m")
                
                self.feedback_message = f"Approaching parcel{current_parcel_index} - Distance: {distance_to_parcel:.2f}m, parcel pose: {self.parcel_pose.position.x:.2f}, {self.parcel_pose.position.y:.2f}, robot pose: {self.robot_pose.position.x:.2f}, {self.robot_pose.position.y:.2f}"
                
                return py_trees.common.Status.RUNNING
                
            else:
                # Robot reached target position (distance <= approach_distance)
                if self.approaching_target:
                    self.approaching_target = False
                    self.control_active = False
                    self.feedback_message = "Target position reached, approach complete"
                    
                    # The timer callback will handle stopping the robot
                    print(f"[{self.name}] Approach complete! Final distance: {distance_to_parcel:.3f}m")
                    
                    return py_trees.common.Status.SUCCESS
                else:
                    # Already at target, return success immediately
                    self.feedback_message = f"Already at target - Distance: {distance_to_parcel:.2f}m"
                    return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        # Stop control and mark as inactive
        self.control_active = False
        self.approaching_target = False
        
        # Stop the robot
        if self.cmd_vel_pub:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
        
        # Clean up timer
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
            print(f"[{self.name}] DEBUG: Cancelled control timer on terminate")
        
        self.feedback_message = f"ApproachObject terminated with status: {new_status}"
        print(f"[{self.name}] ApproachObject terminated with status: {new_status}")


class MoveBackward(py_trees.behaviour.Behaviour):
    """Move backward behavior - using direct velocity control"""
    
    def __init__(self, name, distance=0.2):
        super().__init__(name)
        self.distance = distance  # meters to move backward
        self.start_time = None
        self.ros_node = None
        self.cmd_vel_pub = None
        self.robot_pose_sub = None
        self.start_pose = None
        self.current_pose = None
        self.move_speed = -0.1  # negative for backward movement
        self.robot_namespace = "turtlebot0"  # Default, will be updated from parameters
        
    def setup(self, **kwargs):
        """Setup ROS connections"""
        if 'node' in kwargs:
            self.ros_node = kwargs['node']
            # Get robot namespace from ROS parameters
            try:
                self.robot_namespace = self.ros_node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "turtlebot0"
            
            # Publisher for cmd_vel
            self.cmd_vel_pub = self.ros_node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            # Subscriber for robot pose
            self.robot_pose_sub = self.ros_node.create_subscription(
                Odometry, f'/turtlebot{self.robot_namespace[-1]}/odom_map', 
                self.robot_pose_callback, 10)
    
    def robot_pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        
    def calculate_distance_moved(self):
        if self.start_pose is None or self.current_pose is None:
            return 0.0
        dx = self.current_pose.position.x - self.start_pose.position.x
        dy = self.current_pose.position.y - self.start_pose.position.y
        return math.sqrt(dx*dx + dy*dy)
    
    def initialise(self):
        self.start_time = time.time()
        self.start_pose = self.current_pose
        self.feedback_message = f"Moving backward {self.distance}m"
        print(f"[{self.name}] Starting to move backward...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # Wait for pose data
        if self.current_pose is None:
            self.feedback_message = "Waiting for pose data..."
            return py_trees.common.Status.RUNNING
        
        # Calculate how far we've moved
        distance_moved = self.calculate_distance_moved()
        self.feedback_message = f"Moving backward... {distance_moved:.2f}/{self.distance:.2f}m"
        
        # Check if we've moved far enough
        if distance_moved >= self.distance:
            # Stop the robot
            if self.cmd_vel_pub:
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
            print(f"[{self.name}] Successfully moved to safe distance! Moved: {distance_moved:.2f}m")
            return py_trees.common.Status.SUCCESS
        
        # Continue moving backward
        if self.cmd_vel_pub:
            cmd_vel = Twist()
            cmd_vel.linear.x = self.move_speed
            cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_vel)
        
        # Safety timeout
        elapsed = time.time() - self.start_time
        if elapsed >= 10.0:  # 10 second timeout
            if self.cmd_vel_pub:
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
            print(f"[{self.name}] Move backward timed out")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates - ensure robot stops"""
        # Stop the robot immediately
        if self.cmd_vel_pub:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_cmd)
        
        self.feedback_message = f"MoveBackward terminated with status: {new_status}"
        print(f"[{self.name}] MoveBackward terminated with status: {new_status} - robot stopped")