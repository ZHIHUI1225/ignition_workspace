#!/usr/bin/env python3
"""
Basic behavior classes for the behavior tree system.
Contains utility behaviors like waiting, resetting, and message printing.
"""

import py_trees
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
import math
import threading
import tf_transformations as tf
import casadi as ca
import numpy as np
import threading


class ResetFlags(py_trees.behaviour.Behaviour):
    """Reset system flags behavior"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def update(self):
        print(f"[{self.name}] Resetting flags...")
        return py_trees.common.Status.SUCCESS


class WaitAction(py_trees.behaviour.Behaviour):
    """Wait action behavior - improved version with pose monitoring and proximity checking"""
    
    def __init__(self, name, duration, robot_namespace="tb0", distance_threshold=0.08):
        super().__init__(name)
        self.duration = duration
        self.start_time = 0
        self.robot_namespace = robot_namespace
        self.distance_threshold = distance_threshold
        
        # Extract namespace number for relay point indexing
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        self.relay_number = self.namespace_number  # Relay point is tb{i} -> Relaypoint{i}
        
        # Initialize ROS2 node if not already created
        if not rclpy.ok():
            rclpy.init()
        
        # Create node for subscriptions
        self.node = rclpy.create_node(f'wait_action_{robot_namespace}')
        
        # Pose storage
        self.robot_pose = None
        self.relay_pose = None
        self.parcel_pose = None
        self.current_parcel_index = 0
        
        # Subscriptions
        self.robot_pose_sub = self.node.create_subscription(
            Odometry,
            f'/turtlebot{self.namespace_number}/odom_map',
            self.robot_pose_callback,
            10
        )
        
        self.relay_pose_sub = self.node.create_subscription(
            PoseStamped,
            f'/Relaypoint{self.relay_number}/pose',
            self.relay_pose_callback,
            10
        )
        
        self.parcel_index_sub = self.node.create_subscription(
            Int32,
            f'/{robot_namespace}/current_parcel_index',
            self.current_index_callback,
            10
        )
        
        # Will be updated when parcel index is received
        self.parcel_pose_sub = None
        
        # Create initial parcel subscription for index 0
        self.update_parcel_subscription()
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'tb0' -> 0, 'tb1' -> 1)"""
        import re
        match = re.search(r'tb(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def robot_pose_callback(self, msg):
        """Callback for robot pose updates - handles Odometry message"""
        self.robot_pose = msg.pose.pose
    
    def relay_pose_callback(self, msg):
        """Callback for relay point pose updates"""
        self.relay_pose = msg
    
    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates"""
        self.parcel_pose = msg
        print(f"[{self.name}] Received parcel{self.current_parcel_index} pose: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}")
    
    def current_index_callback(self, msg):
        """Callback for current parcel index updates"""
        old_index = self.current_parcel_index
        self.current_parcel_index = msg.data
        
        # Update parcel subscription if index changed
        if old_index != self.current_parcel_index:
            self.update_parcel_subscription()
    
    def update_parcel_subscription(self):
        """Update parcel pose subscription based on current index"""
        # Destroy old subscription if it exists
        if self.parcel_pose_sub is not None:
            self.node.destroy_subscription(self.parcel_pose_sub)
        
        # Create new subscription for current parcel
        self.parcel_pose_sub = self.node.create_subscription(
            PoseStamped,
            f'/parcel{self.current_parcel_index}/pose',
            self.parcel_pose_callback,
            10
        )
        print(f"[{self.name}] Updated parcel subscription to parcel{self.current_parcel_index}")
    
    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        # Handle different pose message types
        # pose1 could be from Odometry (robot_pose) - extract position directly
        # pose2 could be from PoseStamped (relay/parcel) - extract from .pose.position
        if hasattr(pose1, 'pose'):
            # This is a PoseStamped message
            pos1 = pose1.pose.position
        else:
            # This is already a Pose message (from Odometry.pose.pose)
            pos1 = pose1.position
            
        if hasattr(pose2, 'pose'):
            # This is a PoseStamped message
            pos2 = pose2.pose.position
        else:
            # This is already a Pose message
            pos2 = pose2.position
        
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return math.sqrt(dx*dx + dy*dy)
    
    def check_parcel_in_relay_range(self):
        """Check if parcel is within range of relay point"""
        if self.parcel_pose is None or self.relay_pose is None:
            return False
        
        distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
        return distance <= self.distance_threshold
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Waiting for {self.duration}s and monitoring parcel proximity"
        print(f"[{self.name}] Starting wait for {self.duration}s with parcel monitoring...")
        print(f"[{self.name}] Monitoring robot: tb{self.namespace_number}, relay: Relaypoint{self.relay_number}")
    
    def update(self) -> py_trees.common.Status:
        # Spin ROS2 node to process callbacks
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        elapsed = time.time() - self.start_time
        
        # Check if parcel is in relay range (primary success condition)
        if self.check_parcel_in_relay_range():
            print(f"[{self.name}] SUCCESS: parcel{self.current_parcel_index} is within range of Relaypoint{self.relay_number}!")
            return py_trees.common.Status.SUCCESS
        
        # Check timeout condition
        if elapsed >= self.duration:
            print(f"[{self.name}] TIMEOUT: Wait completed after {self.duration}s, parcel not in range")
            return py_trees.common.Status.FAILURE
        
        # Still running - provide status update
        parcel_relay_dist = self.calculate_distance(self.parcel_pose, self.relay_pose) if self.parcel_pose and self.relay_pose else float('inf')
        print(f"[{self.name}] Waiting... {elapsed:.1f}/{self.duration}s | parcel{self.current_parcel_index} to Relay{self.relay_number} dist: {parcel_relay_dist:.3f}m (threshold: {self.distance_threshold}m)")
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        if hasattr(self, 'node') and self.node:
            self.node.destroy_node()
        super().terminate(new_status)


class ReplanPath(py_trees.behaviour.Behaviour):
    """Path replanning behavior"""
    
    def __init__(self, name, duration=1.5):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Replanning path for {self.duration}s"
        print(f"[{self.name}] Starting to replan path...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.duration, 1.0)
        self.feedback_message = f"Replanning path... {progress*100:.1f}% complete"
        
        if elapsed >= self.duration:
            print(f"[{self.name}] Successfully replanned path!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class StopSystem(py_trees.behaviour.Behaviour):
    """System stop behavior"""
    
    def __init__(self, name, duration=1.0):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Stopping system for {self.duration}s"
        print(f"[{self.name}] Starting to stop system...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            print(f"[{self.name}] System stopped!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class CheckPairComplete(py_trees.behaviour.Behaviour):
    """Check if robot pair operation is complete"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def update(self):
        print(f"[{self.name}] Checking if pair is complete...")
        return py_trees.common.Status.SUCCESS


class IncrementIndex(py_trees.behaviour.Behaviour):
    """Increment current_parcel_index on blackboard"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def update(self):
        blackboard = py_trees.blackboard.Blackboard()
        current_index = getattr(blackboard, 'current_parcel_index', 0)
        blackboard.current_parcel_index = current_index + 1
        print(f"[{self.name}] Incremented current_parcel_index to: {blackboard.current_parcel_index}")
        return py_trees.common.Status.SUCCESS


class PrintMessage(py_trees.behaviour.Behaviour):
    """Print custom message behavior"""
    
    def __init__(self, name, message):
        super().__init__(name)
        self.message = message
    
    def update(self):
        if callable(self.message):
            blackboard = py_trees.blackboard.Blackboard()
            print(self.message(blackboard))
        else:
            print(self.message)
        return py_trees.common.Status.SUCCESS


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
        self.v_max = 0.02     # m/s (reduced from 0.025)
        self.w_max = 0.6      # rad/s (reduced from 0.8)
        
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

    def __init__(self, name="ApproachObject", robot_namespace="tb0", approach_distance=0.12):
        """
        Initialize the ApproachObject behavior.
        
        Args:
            name: Name of the behavior node
            robot_namespace: The robot namespace (e.g., 'tb0', 'tb1')
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
        self.current_parcel_index = 0
        
        # State variables
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target_state = np.array([0.0, 0.0, 0.0])   # [x, y, theta]
        self.approaching_target = False
        
        # ROS2 components (will be initialized in setup)
        self.ros_node = None
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        self.current_index_sub = None
        self.cmd_vel_pub = None
        
        # MPC controller
        self.mpc = MobileRobotMPC()
        
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
                import rclpy
                from rclpy.node import Node
                
                if not rclpy.ok():
                    rclpy.init()
                
                class ApproachObjectNode(Node):
                    def __init__(self):
                        super().__init__(f'approach_object_{self.robot_namespace}')
                
                self.ros_node = ApproachObjectNode()
            
            # Create command velocity publisher
            self.cmd_vel_pub = self.ros_node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            
            # Subscribe to robot pose (Odometry)
            self.robot_pose_sub = self.ros_node.create_subscription(
                Odometry, f'/turtlebot{self.namespace_number}/odom_map',
                self.robot_pose_callback, 10)
            
            # Subscribe to current parcel index
            self.current_index_sub = self.ros_node.create_subscription(
                Int32, f'/{self.robot_namespace}/current_parcel_index',
                self.current_index_callback, 10)
            
            # Initial parcel subscription (will be updated based on current index)
            self.update_parcel_subscription()
            
            self.ros_node.get_logger().info(
                f'ApproachObject setup complete for {self.robot_namespace}')
            return True
            
        except Exception as e:
            print(f"ApproachObject setup failed: {e}")
            return False

    def update_parcel_subscription(self):
        """Update subscription to the correct parcel topic based on current index"""
        if self.ros_node is None:
            return
            
        # Unsubscribe from previous parcel topic if it exists
        if self.parcel_pose_sub is not None:
            self.ros_node.destroy_subscription(self.parcel_pose_sub)
        
        # Subscribe to current parcel topic
        parcel_topic = f'/parcel{self.current_parcel_index}/pose'
        self.parcel_pose_sub = self.ros_node.create_subscription(
            PoseStamped, parcel_topic, self.parcel_pose_callback, 10)
        
        self.ros_node.get_logger().info(f'Updated parcel subscription to: {parcel_topic}')

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

    def current_index_callback(self, msg):
        """Callback for current parcel index updates"""
        new_index = msg.data
        if new_index != self.current_parcel_index:
            with self.lock:
                old_index = self.current_parcel_index
                self.current_parcel_index = new_index
                self.update_parcel_subscription()
                self.ros_node.get_logger().info(
                    f'ApproachObject updated parcel index: {old_index} -> {self.current_parcel_index}')

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
        self.feedback_message = "Initializing approach behavior"
        
        # Spin ROS node once to get latest data
        if self.ros_node:
            import rclpy
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)

    def update(self):
        """
        Main update method - implements the approaching_target logic from State_switch.py
        """
        # Spin ROS node to process callbacks
        if self.ros_node:
            import rclpy
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        with self.lock:
            # Check if we have the necessary pose data
            if self.robot_pose is None or self.parcel_pose is None:
                self.feedback_message = "Waiting for pose data..."
                return py_trees.common.Status.RUNNING

            # Calculate distance to parcel
            distance_to_parcel = self.calculate_distance(self.robot_pose, self.parcel_pose)
            
            # Check if we need to approach (distance > 0.25m as in State_switch.py)
            if distance_to_parcel > 0.25:
                self.approaching_target = True
                self.feedback_message = f"Approaching parcel - Distance: {distance_to_parcel:.2f}m"
                
                # Compute target state following State_switch.py logic
                self.target_state = np.array([
                    self.parcel_pose.position.x,
                    self.parcel_pose.position.y,
                    self.quaternion_to_yaw(self.parcel_pose.orientation)
                ])
                
                # Get optimal direction and apply offset (0.12m as in State_switch.py)
                optimal_direction = self.get_direction(
                    self.current_state[2],
                    self.target_state[2]
                )
                self.target_state[2] = optimal_direction
                self.target_state[0] = self.target_state[0] - self.approach_distance * math.cos(optimal_direction)
                self.target_state[1] = self.target_state[1] - self.approach_distance * math.sin(optimal_direction)
                
                # Generate and apply control using MPC
                u = self.mpc.update_control(self.current_state, self.target_state)
                
                if u is not None and self.cmd_vel_pub:
                    cmd = Twist()
                    cmd.linear.x = float(u[0])
                    cmd.angular.z = float(u[1])
                    self.cmd_vel_pub.publish(cmd)
                
                return py_trees.common.Status.RUNNING
                
            else:
                # Robot reached target position (distance <= 0.25m)
                if self.approaching_target:
                    self.approaching_target = False
                    self.feedback_message = "Target position reached, approach complete"
                    
                    # Stop the robot
                    if self.cmd_vel_pub:
                        cmd = Twist()
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
                        self.cmd_vel_pub.publish(cmd)
                    
                    return py_trees.common.Status.SUCCESS
                else:
                    # Already at target, return success immediately
                    self.feedback_message = f"Already at target - Distance: {distance_to_parcel:.2f}m"
                    return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        # Stop the robot
        if self.cmd_vel_pub:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
        
        self.feedback_message = f"ApproachObject terminated with status: {new_status}"