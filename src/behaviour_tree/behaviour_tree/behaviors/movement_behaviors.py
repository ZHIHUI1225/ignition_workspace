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
    """MPC controller for robot approach - 2-state position-only controller"""
    def __init__(self):
        # MPC parameters
        self.N = 5         # Extended prediction horizon for smoother approach
        self.dt = 0.1        # Time step
        self.wx = 2.0        # Position error weight for x,y convergence
        self.wu = 0.08       # Control effort weight for smooth control
        
        # Control constraints
        self.vx_max = 0.05   # m/s max velocity in x direction
        self.vy_max = 0.05   # m/s max velocity in y direction
        
        # State and control dimensions - 2-state controller
        self.nx = 2          # [x, y] only
        self.nu = 2          # [vx, vy] - direct x,y velocities
        
        # Initialize CasADi optimizer
        self.setup_optimizer()
        
    def setup_optimizer(self):
        # No persistent optimizer setup for 2-state position-only controller
        # Each solve_mpc call creates fresh optimization problem
        pass
        
    def solve_mpc(self, current_state, target_state):
        """
        Solve MPC optimization problem for 2-state position controller using CasADi.
        Returns: cmd_vel array [vx, vy]
        """
        if ca is None:
            # Fallback to simple proportional control if CasADi is not available
            current_pos = np.array([current_state[0], current_state[1]])
            target_pos = np.array([target_state[0], target_state[1]])
            error = target_pos - current_pos
            gain = 0.5  # Increased gain for better response
            cmd_vel = gain * error
            cmd_vel[0] = np.clip(cmd_vel[0], -self.vx_max, self.vx_max)
            cmd_vel[1] = np.clip(cmd_vel[1], -self.vy_max, self.vy_max)
            print(f"[MPC FALLBACK] Using proportional control: error=[{error[0]:.3f}, {error[1]:.3f}], cmd_vel=[{cmd_vel[0]:.3f}, {cmd_vel[1]:.3f}]")
            return cmd_vel
        
        # Convert poses to 2-state vectors [x, y]
        current_pos = np.array([current_state[0], current_state[1]])
        target_pos = np.array([target_state[0], target_state[1]])
        
        # Create optimization problem
        opti = ca.Opti()
        
        # Decision variables
        U = opti.variable(self.nu, self.N)      # [vx; vy] over horizon
        X = opti.variable(self.nx, self.N + 1)  # [x; y] over horizon
        
        # Objective function
        cost = 0
        for k in range(self.N):
            # Position tracking error
            state_error = X[:, k+1] - target_pos
            cost += self.wx * ca.sumsqr(state_error)
            
            # Control effort
            cost += self.wu * ca.sumsqr(U[:, k])
        
        # Terminal cost
        terminal_error = X[:, -1] - target_pos
        cost += self.wx * ca.sumsqr(terminal_error)
        
        opti.minimize(cost)
        
        # Initial condition constraint
        opti.subject_to(X[:, 0] == current_pos)
        
        # Dynamics constraints (direct velocity integration)
        for k in range(self.N):
            # Simple integration: x[k+1] = x[k] + dt * v[k]
            opti.subject_to(X[:, k+1] == X[:, k] + self.dt * U[:, k])
        
        # Velocity constraints
        for k in range(self.N):
            opti.subject_to(U[0, k] <= self.vx_max)   # vx upper bound
            opti.subject_to(U[0, k] >= -self.vx_max)  # vx lower bound
            opti.subject_to(U[1, k] <= self.vy_max)   # vy upper bound
            opti.subject_to(U[1, k] >= -self.vy_max)  # vy lower bound
        
        # Set solver options for fast execution
        opti.solver('ipopt', {
            'ipopt.print_level': 0,         # IPOPT print level
            'print_time': False,            # CasADi print time option
            'ipopt.sb': 'yes',              # Suppress IPOPT banner
            'ipopt.max_iter': 50,           # Maximum iterations
            'ipopt.tol': 1e-4,              # Tolerance
            'ipopt.acceptable_tol': 1e-3    # Acceptable tolerance
        })
        
        # Solve optimization problem
        try:
            sol = opti.solve()
            # Return first control action
            cmd_vel = sol.value(U[:, 0])
            return cmd_vel
        except Exception as e:
            # If optimization fails, use simple proportional control as fallback
            error = target_pos - current_pos
            gain = 0.8
            cmd_vel = gain * error
            cmd_vel[0] = np.clip(cmd_vel[0], -self.vx_max, self.vx_max)
            cmd_vel[1] = np.clip(cmd_vel[1], -self.vy_max, self.vy_max)
            return cmd_vel
        
    def update_control(self, current_state, target_state, position_achieved=False):
        # SEQUENTIAL APPROACH: Position first, then orientation
        
        # Check distance to target position
        dist_to_target = np.sqrt((current_state[0] - target_state[0])**2 + 
                                (current_state[1] - target_state[1])**2)
        
        # Check orientation alignment
        angle_diff = abs((current_state[2] - target_state[2] + np.pi) % (2 * np.pi) - np.pi)
        
        # If we're very close to the target and well-aligned, stop completely
        if dist_to_target < 0.015 and angle_diff < 0.05:  # 1.5cm and ~3 degrees
            return np.array([0.0, 0.0])  # Stop completely
        
        # PHASE 1: Focus on reaching target position first
        # If position is not yet achieved, use position-only MPC
        if not position_achieved and dist_to_target >= 0.02:  # 2cm threshold for position reaching
            # Use 2-state MPC for position control only
            cmd_vel_2d = self.solve_mpc(current_state, target_state)
            
            # Convert 2D velocity command to [linear_x, angular_z] format
            # For differential drive robot, we need to convert x,y velocities to linear/angular
            vx_global = cmd_vel_2d[0]
            vy_global = cmd_vel_2d[1]
            
            # Convert global velocities to robot frame
            theta = current_state[2]
            vx_robot = vx_global * np.cos(theta) + vy_global * np.sin(theta)
            vy_robot = -vx_global * np.sin(theta) + vy_global * np.cos(theta)
            
            # For differential drive, convert to linear and angular velocity
            # Assuming robot can move sideways (like mecanum drive) or holonomic
            # If purely differential drive, set vy_robot to 0 and convert to angular
            linear_vel = vx_robot
            angular_vel = vy_robot / 0.15  # Convert lateral motion to rotation (increased wheelbase factor for stability)
            
            return np.array([linear_vel, angular_vel])
                
        # PHASE 2: Position achieved, now focus on orientation alignment
        elif position_achieved:
            # Pure rotation control - no linear motion
            angular_error = target_state[2] - current_state[2]
            # Normalize angle error to [-pi, pi]
            while angular_error > np.pi:
                angular_error -= 2 * np.pi
            while angular_error < -np.pi:
                angular_error += 2 * np.pi
            
            # Proportional rotation control with damping - reduced gain for stability
            angular_vel = 0.3 * angular_error  # Reduced proportional gain for smoother control
            angular_vel = np.clip(angular_vel, -0.2, 0.2)  # Reduced rotation speed limit for stability
            
            return np.array([0.0, angular_vel])  # Pure rotation, no linear motion
        
        # Fallback: stop if in between phases
        else:
            return np.array([0.0, 0.0])


class ApproachObject(py_trees.behaviour.Behaviour):
    """
    Approach Object behavior - uses sequential position and orientation control.
    Uses MPC controller to make the robot approach the target with separate position and orientation phases.
    """

    def __init__(self, name="ApproachObject", robot_namespace="turtlebot0", approach_distance=0.14):
        """
        Initialize the ApproachObject behavior.
        
        Args:
            name: Name of the behavior node
            robot_namespace: The robot namespace (e.g., 'turtlebot0', 'turtlebot1')
            approach_distance: Distance to maintain from the parcel
        """
        super(ApproachObject, self).__init__(name)
        self.robot_namespace = robot_namespace
        self.approach_distance = approach_distance
        
        # Extract namespace number for topic subscriptions
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        
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
        
        # Initialize only persistent variables that shouldn't reset
        # State variables will be initialized in initialise() method
        
        # ROS2 components (will be initialized in setup)
        self.ros_node = None
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        self.cmd_vel_pub = None
        
        # MPC controller will be initialized in initialise() method
        self.mpc = None
        
        # Control loop timer
        self.control_timer = None
        self.dt = 0.1  # 0.1s timer period for MPC control (10Hz)
        
        # Threading lock for state protection
        self.lock = threading.Lock()
        
    def _stop_robot(self):
        """Helper method to stop the robot"""
        if self.cmd_vel_pub:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)

    def _calculate_errors(self):
        """Helper method to calculate position and orientation errors"""
        pos_dist = np.sqrt((self.current_state[0] - self.target_state[0])**2 + 
                          (self.current_state[1] - self.target_state[1])**2)
        angle_diff = abs((self.current_state[2] - self.target_state[2] + np.pi) % (2 * np.pi) - np.pi)
        return pos_dist, angle_diff

    def _update_control_flags(self, pos_dist, angle_diff):
        """Helper method to update position and orientation control flags"""
        position_threshold = 0.02  # 2cm for position
        orientation_threshold = 0.05  # ~3 degrees for orientation
        
        # SEQUENTIAL CONTROL: Position first, then orientation
        if pos_dist < position_threshold:
            self.position_control_achieved = True
        
        # Phase 2: Orientation control - only start after position is achieved
        if self.position_control_achieved and angle_diff < orientation_threshold:
            self.orientation_control_achieved = True

    def extract_namespace_number(self, namespace):
        """Extract numerical index from namespace string using regex"""
        import re
        match = re.search(r'\d+', namespace)
        return int(match.group()) if match else 0

    def setup(self, **kwargs):
        """Setup ROS connections"""
        try:
            # Always use the shared ROS node from the behavior tree
            if 'node' in kwargs:
                self.ros_node = kwargs['node']
                self.ros_node.get_logger().info(f'[{self.name}] Using shared ROS node: {self.ros_node.get_name()}')
            else:
                raise RuntimeError("No ROS node provided in kwargs. ApproachObject requires a shared node for proper callback processing.")
            
            # Create command velocity publisher
            self.cmd_vel_pub = self.ros_node.create_publisher(
                Twist, f'/turtlebot{self.namespace_number}/cmd_vel', 10)
            
            # Initialize subscriptions as None - will be set up in initialise() method
            self.robot_pose_sub = None
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

    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates (PoseStamped message)"""
        with self.lock:
            self.parcel_pose = msg.pose

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
        # Only execute control loop if approach is active AND not yet complete
        if self.control_active and not (self.position_control_achieved and self.orientation_control_achieved):
            self.control_loop()
        else:
            # Stop the robot when not actively approaching or when approach is complete
            self._stop_robot()

    def control_loop(self):
        """Control loop for the approaching behavior"""
        with self.lock:
            # Check if we have the necessary pose data
            if self.robot_pose is None or self.parcel_pose is None:
                return
            
            # Calculate target state using the dedicated function
            target_state, distance_to_target_state = self.calculate_target_state()
            if target_state is None:
                return
            
            # Update instance target_state
            self.target_state = target_state
            
            # Calculate position and orientation errors
            pos_dist, angle_diff = self._calculate_errors()
            
            # Update control flags based on errors
            self._update_control_flags(pos_dist, angle_diff)
            
            # Check if both position and orientation control are achieved
            if self.position_control_achieved and self.orientation_control_achieved:
                self._stop_robot()
                # Approach complete, stop control
                self.control_active = False
                print(f"[{self.name}][{self.robot_namespace}] Both position and orientation control achieved! pos: {pos_dist:.3f}m, angle: {angle_diff:.3f}rad")
            else:
                # Continue control if we haven't reached both targets yet
                if self.mpc is None:
                    print(f"[{self.name}][{self.robot_namespace}] WARNING: MPC controller not initialized, skipping control")
                    return
                
                # Generate and apply control using MPC
                u = self.mpc.update_control(self.current_state, self.target_state, self.position_control_achieved)
                
                if u is not None and self.cmd_vel_pub:
                    cmd = Twist()
                    cmd.linear.x = float(u[0])
                    cmd.angular.z = float(u[1])
                    self.cmd_vel_pub.publish(cmd)

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

    def calculate_target_state(self):
        """
        Calculate the target state for the robot based on parcel pose and optimal approach direction.
        
        Returns:
            tuple: (target_state, distance_to_target_state) where target_state is np.array([x, y, theta])
                   and distance_to_target_state is the Euclidean distance to the target position
        """
        if self.robot_pose is None or self.parcel_pose is None:
            return None, float('inf')
        
        # Compute target state following State_switch.py logic
        target_state = np.array([
            self.parcel_pose.position.x,
            self.parcel_pose.position.y,
            self.quaternion_to_yaw(self.parcel_pose.orientation)
        ])
        
        # Get optimal direction and apply offset
        optimal_direction = self.get_direction(
            self.current_state[2],
            target_state[2]
        )
        target_state[2] = optimal_direction
        target_state[0] = target_state[0] - (self.approach_distance) * math.cos(optimal_direction)
        target_state[1] = target_state[1] - (self.approach_distance) * math.sin(optimal_direction)
        
        # Calculate distance to target state
        distance_to_target_state = math.sqrt(
            (self.current_state[0] - target_state[0])**2 + 
            (self.current_state[1] - target_state[1])**2
        )
        
        return target_state, distance_to_target_state

    def initialise(self):
        """Initialize the behavior when it starts running"""
        # Reset state variables every time behavior launches
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target_state = np.array([0.0, 0.0, 0.0])   # [x, y, theta]
        self.control_active = False
        
        # Add separate flags for position and orientation control
        self.position_control_achieved = False
        self.orientation_control_achieved = False
        
        # Add timeout tracking
        self.start_time = time.time()
        self.timeout_duration = 30.0  # 30 second timeout for approach
        
        # Reset ROS2 components
        self.robot_pose = None
        self.parcel_pose = None
        
        self.feedback_message = f"[{self.robot_namespace}] Initializing approach behavior"
        
        # Set default pushing estimated time (45 seconds) every time behavior starts
        setattr(self.blackboard, f"{self.robot_namespace}/pushing_estimated_time", 45.0)
        
        # Reset and create fresh MPC controller every time the node launches
        print(f"[{self.name}][{self.robot_namespace}] Creating fresh MPC controller instance")
        self.mpc = MobileRobotMPC()
        
        # Clean up existing subscriptions and timer
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
        if self.robot_pose_sub is not None:
            self.ros_node.destroy_subscription(self.robot_pose_sub)
            self.robot_pose_sub = None
        if self.parcel_pose_sub is not None:
            self.ros_node.destroy_subscription(self.parcel_pose_sub)
            self.parcel_pose_sub = None
        
        # Set up robot pose subscription
        if self.ros_node:
            self.robot_pose_sub = self.ros_node.create_subscription(
                Odometry, f'/turtlebot{self.namespace_number}/odom_map',
                self.robot_pose_callback, 10)
            print(f"[{self.name}][{self.robot_namespace}] Set up robot pose subscription to /turtlebot{self.namespace_number}/odom_map")
        
            # Set up parcel subscription with current parcel index
            current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
            success = self.setup_parcel_subscription(current_parcel_index)
            if success:
                self._last_parcel_index = current_parcel_index
                print(f"[{self.name}][{self.robot_namespace}] Initialized parcel subscription for parcel{current_parcel_index}")
            else:
                print(f"[{self.name}][{self.robot_namespace}] WARNING: Failed to set up parcel subscription for parcel{current_parcel_index}")
        
            # Create and start ROS timer for control loop at 10Hz
            self.control_timer = self.ros_node.create_timer(self.dt, self.control_timer_callback)
            print(f"[{self.name}][{self.robot_namespace}] DEBUG: Created ROS timer for control loop at {1/self.dt:.1f} Hz (every {self.dt}s)")
        else:
            print(f"[{self.name}][{self.robot_namespace}] WARNING: No ROS node available, cannot create control timer")

    def update(self):
        """Main update method - behavior tree logic only, control runs via timer"""
        # Handle parcel index changes
        current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
        if self._last_parcel_index != current_parcel_index:
            print(f"[{self.name}][{self.robot_namespace}] Parcel index changed from {self._last_parcel_index} to {current_parcel_index} - updating subscription")
            success = self.setup_parcel_subscription(current_parcel_index)
            if success:
                self._last_parcel_index = current_parcel_index

        with self.lock:
            # Check timeout
            elapsed = time.time() - self.start_time
            if elapsed >= self.timeout_duration:
                from .tree_builder import report_node_failure
                error_msg = f"ApproachObject timeout after {elapsed:.1f}s - failed to reach parcel"
                report_node_failure(self.name, error_msg, self.robot_namespace)
                print(f"[{self.name}][{self.robot_namespace}] FAILURE: Approach timeout after {elapsed:.1f}s")
                return py_trees.common.Status.FAILURE
            
            # Check if we have pose data and calculate target state
            if self.robot_pose is None or self.parcel_pose is None:
                self.feedback_message = f"[{self.robot_namespace}] Waiting for pose data..."
                return py_trees.common.Status.RUNNING

            target_state, distance_to_target_state = self.calculate_target_state()
            if target_state is None:
                self.feedback_message = f"[{self.robot_namespace}] Waiting for pose data..."
                return py_trees.common.Status.RUNNING
            
            self.target_state = target_state
            
            # Check if approach is complete
            if self.position_control_achieved and self.orientation_control_achieved and not self.control_active:
                self._stop_robot()
                self.feedback_message = f"[{self.robot_namespace}] Both position and orientation control achieved, approach complete"
                print(f"[{self.name}][{self.robot_namespace}] Approach complete! Both control flags achieved. Distance to target state: {distance_to_target_state:.3f}m")
                return py_trees.common.Status.SUCCESS
            else:
                # Continue approaching the target state
                self.control_active = True
                print(f"[{self.name}][{self.robot_namespace}] Continuing approach to target state, distance: {distance_to_target_state:.3f}m")
                
                self.feedback_message = f"[{self.robot_namespace}] Approaching target state - Distance: {distance_to_target_state:.3f}m, target: ({self.target_state[0]:.2f}, {self.target_state[1]:.2f}), robot: ({self.current_state[0]:.2f}, {self.current_state[1]:.2f}), position_flag: {self.position_control_achieved}, orientation_flag: {self.orientation_control_achieved}"
                
                return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        # Stop control and mark as inactive
        self.control_active = False
        
        # Reset control flags
        self.position_control_achieved = False
        self.orientation_control_achieved = False
        
        # Stop the robot
        self._stop_robot()
        
        # Clean up timer
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
            print(f"[{self.name}][{self.robot_namespace}] DEBUG: Cancelled control timer on terminate")
        
        # Clean up subscriptions
        if self.robot_pose_sub is not None:
            self.ros_node.destroy_subscription(self.robot_pose_sub)
            self.robot_pose_sub = None
            print(f"[{self.name}][{self.robot_namespace}] DEBUG: Destroyed robot pose subscription on terminate")
        
        if self.parcel_pose_sub is not None:
            self.ros_node.destroy_subscription(self.parcel_pose_sub)
            self.parcel_pose_sub = None
            print(f"[{self.name}][{self.robot_namespace}] DEBUG: Destroyed parcel pose subscription on terminate")
        
        self.feedback_message = f"[{self.robot_namespace}] ApproachObject terminated with status: {new_status}"
        print(f"[{self.name}][{self.robot_namespace}] ApproachObject terminated with status: {new_status}")


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
        self.feedback_message = f"[{self.robot_namespace}] Moving backward {self.distance}m"
        print(f"[{self.name}] Starting to move backward...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # Wait for pose data
        if self.current_pose is None:
            self.feedback_message = f"[{self.robot_namespace}] Waiting for pose data..."
            return py_trees.common.Status.RUNNING
        
        # Calculate how far we've moved
        distance_moved = self.calculate_distance_moved()
        self.feedback_message = f"[{self.robot_namespace}] Moving backward... {distance_moved:.2f}/{self.distance:.2f}m"
        
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
        if elapsed >= 15.0:  # 10 second timeout
            if self.cmd_vel_pub:
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
            from .tree_builder import report_node_failure
            error_msg = f"MoveBackward timeout after {elapsed:.1f}s - failed to reach target distance"
            report_node_failure(self.name, error_msg, "turtlebot0")  # MoveBackward doesn't have robot_namespace
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
        
        self.feedback_message = f"[{self.robot_namespace}] MoveBackward terminated with status: {new_status}"
        print(f"[{self.name}] MoveBackward terminated with status: {new_status} - robot stopped")