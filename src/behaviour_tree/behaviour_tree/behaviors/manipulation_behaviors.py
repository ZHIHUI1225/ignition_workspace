#!/usr/bin/env python3
"""
Manipulation behavior classes for the behavior tree system.
Contains object manipulation behaviors like pushing and picking.
"""
import py_trees
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import time
import threading
import json
import numpy as np
import casadi as ca
import os
import math
from tf_transformations import euler_from_quaternion
import re
import copy
import tf_transformations as tf
from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Int32
import math

def extract_namespace_number(namespace):
    """Extract numerical index from robot namespace"""
    match = re.search(r'turtlebot(\d+)', namespace)
    return int(match.group(1)) if match else 0
class MobileRobotMPC:
    def __init__(self):
        # MPC parameters - balanced for tracking accuracy and computation speed
        self.N = 8           # Increased prediction horizon for better tracking
        self.N_c = 3         # Increased control horizon for smoother control
        self.dt = 0.1        # Time step
        # Higher position weights for better tracking precision
        self.Q = np.diag([100.0, 100.0, 10.0, 0.5, 2.0])  # State weights (x, y, theta, v, omega) - increased position weights
        # Lower control input weights to allow more aggressive control for better tracking
        self.R = np.diag([0.05, 0.02])       # Control input weights - reduced for better tracking
        # Much higher terminal position weights for precise goal reaching
        self.F = np.diag([200.0, 200.0, 20.0, 1.0, 5.0])  # Terminal cost weights - much higher position weights
        
        # Velocity constraints
        self.max_vel = 0.25      # m/s
        self.min_vel = 0.0    # m/s (allow reverse)
        self.max_omega = np.pi/2 # rad/s
        self.min_omega = -np.pi/2 # rad/s
        
        # System dimensions
        self.nx = 5   # Number of states (x, y, theta, v, omega)
        self.nu = 2    # Number of controls (v, omega)
        
        # Reference trajectory
        self.ref_traj = None
        
        # Solver cache for warm starting
        self.last_solution = None
        
        # Initialize MPC
        self.setup_mpc()

    def setup_mpc(self):
        # Optimization problem
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N+1)  # State trajectory (N+1 states)
        self.U = self.opti.variable(self.nu, self.N)    # Control trajectory
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)          # Initial state
        self.ref = self.opti.parameter(self.nx, self.N+1)  # Reference trajectory

        # Cost function - fixed matrix operations and added velocity tracking
        cost = 0
        for k in range(self.N):
            # Position error (x,y) - fixed matrix multiplication
            pos_error = self.X[:2, k] - self.ref[:2, k]
            cost += ca.mtimes([pos_error.T, self.Q[:2,:2], pos_error])
            
            # Orientation (theta) - scalar operation
            theta_error = self.X[2, k] - self.ref[2, k]
            cost += self.Q[2,2] * theta_error**2
            
            # Velocity states (v, omega) - track reference velocities
            vel_error = self.X[3:, k] - self.ref[3:, k]
            cost += ca.mtimes([vel_error.T, self.Q[3:,3:], vel_error])
            
            # Control cost - fixed matrix multiplication
            cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
            
            # Dynamics constraints
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost - fixed matrix operations
        pos_term_error = self.X[:2, -1] - self.ref[:2, -1]
        cost += ca.mtimes([pos_term_error.T, self.F[:2,:2], pos_term_error])
        
        # Terminal orientation error
        theta_term_error = self.X[2, -1] - self.ref[2, -1]
        cost += self.F[2,2] * theta_term_error**2
        
        # Terminal velocity errors
        vel_term_error = self.X[3:, -1] - self.ref[3:, -1]
        cost += ca.mtimes([vel_term_error.T, self.F[3:,3:], vel_term_error])

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Control input constraints (velocity and angular velocity)
        self.opti.subject_to(self.opti.bounded(-0.25, self.U[0, :], self.max_vel))  # Allow reverse
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))

        # Solver settings - simplified and robust
        self.opti.minimize(cost)
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 30,       # Reasonable iterations for convergence
            'ipopt.max_cpu_time': 0.1,  # Relaxed time limit
            'ipopt.tol': 1e-3,          # Standard tolerance
            'ipopt.acceptable_tol': 5e-3, # More reasonable tolerance
            'ipopt.acceptable_iter': 5,  # Accept solution after 5 iterations
            'ipopt.warm_start_init_point': 'yes', # Use warm starting
            'ipopt.hessian_approximation': 'limited-memory', # BFGS approximation
            'ipopt.linear_solver': 'mumps' # Reliable linear solver
        }
        
        self.opti.solver('ipopt', opts)

    def robot_model(self, x, u):
        """ System dynamics: x_next = f(x, u) """
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * ca.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt,                 # theta
            u[0],                                  # velocity (directly set)
            u[1]                                   # angular velocity (directly set)
        )

    def set_reference_trajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def update(self, current_state):
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.ref, self.ref_traj)
        
        # Simplified warm starting for speed
        if self.last_solution is not None:
            # Simple shift without expensive propagation
            u_init = np.zeros((self.nu, self.N))
            x_init = np.zeros((self.nx, self.N+1))
            
            # Copy and shift previous solution
            if self.N > 1:
                u_init[:, :self.N-1] = self.last_solution['u'][:, 1:]
                u_init[:, self.N-1] = self.last_solution['u'][:, -1]  # Repeat last control
                
                x_init[:, :self.N] = self.last_solution['x'][:, 1:]
                x_init[:, self.N] = self.last_solution['x'][:, -1]  # Use last state instead of propagation
            
            self.opti.set_initial(self.X, x_init)
            self.opti.set_initial(self.U, u_init)
        else:
            pass  # No previous solution available for warm start
        
        try:
            sol = self.opti.solve()
            
            # Store solution for warm starting next time
            self.last_solution = {
                'u': sol.value(self.U),
                'x': sol.value(self.X)
            }
            
            # Return all N_c control steps
            return sol.value(self.U)[:, :self.N_c]
        except Exception as e:
            print(f"MPC Solver failed: {str(e)}")
            # Clear last solution on failure to avoid using stale data
            self.last_solution = None
            return np.zeros((self.nu, self.N_c))
    
    def robot_model_np(self, x, u):
        """System dynamics in numpy for warm starting"""
        return np.array([
            x[0] + u[0] * np.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * np.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt,                 # theta
            u[0],                                  # velocity (directly set)
            u[1]                                   # angular velocity (directly set)
        ])

    def get_predicted_trajectory(self):
        try:
            return self.opti.debug.value(self.X)
        except:
            return np.zeros((self.nx, self.N+1))

class PushObject(py_trees.behaviour.Behaviour):
    """Push object behavior using MPC controller for trajectory following"""
    
    def __init__(self, name="PushObject", robot_namespace="turtlebot0"):
        super().__init__(name)
        self.node = None
        self.robot_namespace = robot_namespace  # Use provided namespace
        self.case = "simple_maze"  # Default case
        self.number=extract_namespace_number(robot_namespace)
        # Setup blackboard access for namespaced current_parcel_index and pushing_estimated_time
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/pushing_estimated_time", 
            access=py_trees.common.Access.WRITE
        )
        
        # MPC Controller
        self.mpc = MobileRobotMPC()
        
        # State variables
        self.current_state = np.zeros(5)  # [x, y, theta, v, omega]
        self.ref_trajectory = None
        self.trajectory_index = 0
        self.P_HOR = self.mpc.N
        
        # ROS publishers and subscribers
        self.cmd_vel_pub = None
        self.prediction_pub = None
        self.reference_pub = None
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        self.relay_pose_sub = None
        
        # Pose storage for success checking
        self.parcel_pose = None
        self.relay_pose = None
        self.current_parcel_index = 0
        self.last_parcel_index = -1  # Track changes in parcel index
        self.distance_threshold = 0.14  # Distance threshold for success condition
        
        # Path messages for visualization
        self.ref_path = Path()
        self.pred_path = Path()
        
        # Control loop timer
        self.control_timer = None
        
        # Control sequence management (like Follow_controller.py)
        self.control_sequence = None
        self.control_step = 0
        self.dt = 0.1  # 0.1s timer period for MPC control
        
        # State tracking
        self.pushing_active = False
        self.start_time = None
        self.last_time = None
        
        # Threading lock for state protection
        self.state_lock = threading.Lock()
        
        # Control error tracking
        self.last_control_errors = {}
    
    def setup(self, **kwargs):
        """Setup ROS connections and load trajectory"""
        if 'node' in kwargs:
            self.node = kwargs['node']
            
            # Get robot namespace and case from ROS parameters
            try:
                self.robot_namespace = self.node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "turtlebot0"
            
            try:
                self.case = self.node.get_parameter('case').get_parameter_value().string_value
            except:
                self.case = "simple_maze"
            
            # Load reference trajectory from JSON file
            self._load_trajectory()
            
            # Create ROS publishers
            self.cmd_vel_pub = self.node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            
            self.prediction_pub = self.node.create_publisher(
                Path, f'/tb{self._extract_namespace_number()}/pushing/pred_path', 10)
            
            self.reference_pub = self.node.create_publisher(
                Path, f'/tb{self._extract_namespace_number()}/pushing/ref_path', 10)
            
            # Subscribe to robot odometry
            robot_odom_topic = f'/turtlebot{self._extract_namespace_number()}/odom_map'
            self.robot_pose_sub = self.node.create_subscription(
                Odometry, robot_odom_topic, self.robot_pose_callback, 10)
            
            # Subscribe to parcel pose
            self.parcel_pose_sub = self.node.create_subscription(
                PoseStamped, f'/parcel{self.current_parcel_index}/pose', 
                self.parcel_pose_callback, 10)
            
            # Subscribe to relay point pose
            relay_number = self._extract_namespace_number()+1  #  Relaypoint{i+1}
            self.relay_pose_sub = self.node.create_subscription(
                PoseStamped, f'/Relaypoint{relay_number}/pose',
                self.relay_pose_callback, 10)
            
            # Initialize path messages
            self.ref_path.header.frame_id = 'world'
            self.pred_path.header.frame_id = 'world'
            
            print(f"[{self.name}] Setup complete for {self.robot_namespace}, case: {self.case}")
            return True
        
        return False
    
    def _extract_namespace_number(self):
        """Extract numerical index from robot namespace"""
        match = re.search(r'turtlebot(\d+)', self.robot_namespace)
        return int(match.group(1)) if match else 0
    
    def _load_trajectory(self):
        """Load reference trajectory from JSON file"""
        json_file_path = f'/root/workspace/data/{self.case}/tb{self.number}_Trajectory.json'
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            self.ref_trajectory = data['Trajectory']
            print(f"[{self.name}] Loaded trajectory with {len(self.ref_trajectory)} points")
            
            # Set initial pushing estimated time in blackboard
            initial_estimated_time = len(self.ref_trajectory) * self.dt
            setattr(self.blackboard, f"{self.robot_namespace}/pushing_estimated_time", initial_estimated_time)
            print(f"[{self.name}] Set initial pushing_estimated_time: {initial_estimated_time:.2f}s")

    
    def robot_pose_callback(self, msg):
        """Update robot state from odometry message"""
        with self.state_lock:
            # Position
            self.current_state[0] = msg.pose.pose.position.x
            self.current_state[1] = msg.pose.pose.position.y
            
            # Orientation (yaw)
            quat = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            euler = tf.euler_from_quaternion(quat)
            self.current_state[2] = euler[2]  # yaw
            
            # Velocities
            self.current_state[3] = msg.twist.twist.linear.x   # Linear velocity
            self.current_state[4] = msg.twist.twist.angular.z  # Angular velocity
    
    def parcel_pose_callback(self, msg):
        """Update parcel pose from PoseStamped message"""
        with self.state_lock:
            self.parcel_pose = msg.pose
    
    def relay_pose_callback(self, msg):
        """Update relay point pose from PoseStamped message"""
        with self.state_lock:
            self.relay_pose = msg.pose
    
    def _calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx*dx + dy*dy)
    
    def _check_parcel_in_relay_range(self):
        """Check if parcel is within distance threshold of relay point"""
        if self.parcel_pose is None or self.relay_pose is None:
            return False
        
        distance = self._calculate_distance(self.parcel_pose, self.relay_pose)
        is_in_range = distance <= self.distance_threshold
        
        if is_in_range:
            print(f"[{self.name}] SUCCESS: Parcel is within {self.distance_threshold}m of relay point (distance: {distance:.3f}m)")
        
        return is_in_range
    
    def _update_parcel_subscription(self, parcel_index):
        """Update subscription to track the correct parcel"""
        if self.node is None:
            return
        
        # Destroy existing subscription if it exists
        if self.parcel_pose_sub is not None:
            self.node.destroy_subscription(self.parcel_pose_sub)
        
        # Create new subscription for the specified parcel
        self.parcel_pose_sub = self.node.create_subscription(
            PoseStamped, f'/parcel{parcel_index}/pose',
            self.parcel_pose_callback, 10)
        
        self.current_parcel_index = parcel_index
        print(f"[{self.name}] Updated parcel subscription to parcel{parcel_index}")
    
    def _update_pushing_estimated_time(self):
        """Update pushing estimated time in blackboard based on current trajectory index"""
        if self.ref_trajectory:
            if self.trajectory_index < len(self.ref_trajectory):
                remaining_points = len(self.ref_trajectory) - self.trajectory_index
                estimated_time = remaining_points * self.dt
            else:
                # At end of trajectory, minimal estimated time remaining
                estimated_time = 0.5  # Small positive value to indicate near completion
                remaining_points = 0
            
            setattr(self.blackboard, f"{self.robot_namespace}/pushing_estimated_time", estimated_time)
            print(f"[{self.name}] Updated pushing_estimated_time: {estimated_time:.2f}s (remaining points: {remaining_points})")
    
    def _publish_reference_trajectory(self):
        """Publish reference trajectory for visualization"""
        if not self.ref_trajectory:
            return
        self.ref_path.poses.clear()
        self.ref_path.header.stamp = self.node.get_clock().now().to_msg()
        
        # Publish a segment of the reference trajectory around current index
        start_idx = max(0, self.trajectory_index)
        end_idx = min(len(self.ref_trajectory), start_idx + self.P_HOR + 5)
        
        for i in range(start_idx, end_idx):
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'world'
            pose_msg.header.stamp = self.ref_path.header.stamp
            
            pose_msg.pose.position.x = self.ref_trajectory[i][0]
            pose_msg.pose.position.y = self.ref_trajectory[i][1]
            pose_msg.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            yaw = self.ref_trajectory[i][2]
            quat = tf.quaternion_from_euler(0.0, 0.0, yaw)
            pose_msg.pose.orientation.x = quat[1]
            pose_msg.pose.orientation.y = quat[2]
            pose_msg.pose.orientation.z = quat[3]
            pose_msg.pose.orientation.w = quat[0]
            
            self.ref_path.poses.append(pose_msg)
        
        self.reference_pub.publish(self.ref_path)
    
    def _publish_predicted_trajectory(self):
        """Publish predicted trajectory from MPC with enhanced validation"""
        try:
            pred_traj = self.mpc.get_predicted_trajectory()
            if pred_traj is None:
                print(f"[{self.name}] WARNING: get_predicted_trajectory() returned None")
                return
            
            # Use the same timestamp as the reference trajectory for proper alignment
            current_stamp = self.node.get_clock().now().to_msg()
            self.pred_path.header.stamp = current_stamp
            self.pred_path.poses.clear()
            
            # Check for valid prediction trajectory (enhanced validation like Follow_controller.py)
            if isinstance(pred_traj, np.ndarray) and not np.isnan(pred_traj).any():
                for i in range(min(self.mpc.N+1, pred_traj.shape[1])):
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = "world"
                    pose_msg.header.stamp = current_stamp
                    pose_msg.pose.position.x = pred_traj[0, i]
                    pose_msg.pose.position.y = pred_traj[1, i]
                    pose_msg.pose.position.z = 0.0
                    
                    # Ensure we have valid orientation
                    yaw = pred_traj[2, i]
                    if not np.isnan(yaw):
                        quat = tf.quaternion_from_euler(0.0, 0.0, yaw)
                        pose_msg.pose.orientation.x = quat[1]
                        pose_msg.pose.orientation.y = quat[2]
                        pose_msg.pose.orientation.z = quat[3]
                        pose_msg.pose.orientation.w = quat[0]
                    else:
                        # Default orientation if invalid
                        pose_msg.pose.orientation.w = 1.0
                    
                    self.pred_path.poses.append(pose_msg)
                
                # Publish the predicted trajectory
                if len(self.pred_path.poses) > 0:
                    self.prediction_pub.publish(self.pred_path)
                else:
                    print(f"[{self.name}] No valid poses generated for predicted trajectory")
            else:
                print(f"[{self.name}] WARNING: Predicted trajectory contains invalid data or NaN values")
            
        except Exception as e:
            print(f"[{self.name}] Error publishing predicted trajectory: {e}")
    
    def _get_reference_trajectory_segment(self):
        """Get reference trajectory segment for MPC"""
        if not self.ref_trajectory:
            return np.zeros((5, self.P_HOR + 1))
        
        ref_array = np.zeros((5, self.P_HOR + 1))
        
        # Fill reference trajectory
        for i in range(self.P_HOR + 1):
            if self.trajectory_index + i < len(self.ref_trajectory):
                # Use normal trajectory point
                traj_idx = self.trajectory_index + i
                ref_point = self.ref_trajectory[traj_idx]
            else:
                # Trajectory index has reached the end - use last state as target
                ref_point = self.ref_trajectory[-1]  # Last state of trajectory
            
            ref_array[0, i] = ref_point[0]  # x
            ref_array[1, i] = ref_point[1]  # y
            ref_array[2, i] = ref_point[2]  # theta
            ref_array[3, i] = ref_point[3]  # v
            ref_array[4, i] = ref_point[4]  # omega
        
        return ref_array
    
    
    def advance_control_step(self):
        """Advance to the next control step in the stored sequence"""
        if self.control_sequence is None:
            return False
        
        self.control_step += 1
        
        # When using a control step from our sequence, we effectively move forward in our trajectory
        # But don't go beyond the end of the trajectory
        if self.trajectory_index < len(self.ref_trajectory) - 1:
            self.trajectory_index += 1
        else:
            # At end of trajectory, keep using last state as target
            print(f"[{self.name}] At end of trajectory in advance_control_step, maintaining last state as target")
        
        # Update pushing estimated time in blackboard
        self._update_pushing_estimated_time()
        
        # If we've used all our control steps, need a new MPC solve
        if self.control_step >= self.mpc.N_c:
            self.control_step = 0
            self.control_sequence = None
            return False
        
        # Otherwise we can use the next step from our stored sequence
        return True
    
    def apply_stored_control(self):
        """Apply the current step from the stored control sequence"""
        if self.control_sequence is None or self.control_step >= self.mpc.N_c:
            print(f"[{self.name}] DEBUG: Cannot apply stored control - sequence: {self.control_sequence is not None}, step: {self.control_step}/{self.mpc.N_c}")
            return False
        
        cmd_msg = Twist()
        cmd_msg.linear.x = float(self.control_sequence[0, self.control_step])
        cmd_msg.angular.z = float(self.control_sequence[1, self.control_step])
        
        # Debug: Check if publisher exists
        if self.cmd_vel_pub is None:
            print(f"[{self.name}] ERROR: cmd_vel_pub is None, cannot publish command")
            return False
            
        self.cmd_vel_pub.publish(cmd_msg)
        
        print(f"[{self.name}] Applied stored control step {self.control_step+1}/{self.mpc.N_c}: "
              f"v={cmd_msg.linear.x:.2f}, ω={cmd_msg.angular.z:.2f} [PUBLISHED]")
        return True
    
    def control_timer_callback(self):
        """ROS timer callback for MPC control loop - runs at 10Hz"""
        # Add timing debug info
        current_time = time.time()
        if not hasattr(self, '_last_timer_time'):
            self._last_timer_time = current_time
            self._timer_call_count = 0
            print(f"[{self.name}] TIMER DEBUG: First callback, timer period set to {self.dt}s")
        
        time_interval = current_time - self._last_timer_time
        self._timer_call_count += 1
        self._last_timer_time = current_time
        
        # Log timing every single call to see frequency clearly
        frequency = 1.0 / time_interval if time_interval > 0 else 0
        print(f"[{self.name}] Timer callback #{self._timer_call_count}, "
              f"interval: {time_interval:.3f}s, frequency: {frequency:.1f}Hz, "
              f"expected_freq: {1/self.dt:.1f}Hz")
        
        # Only execute control loop if pushing is active
        if self.pushing_active:
            # Use timeout protection to ensure timer doesn't get blocked
            start_time = time.time()
            try:
                self.control_loop()
                execution_time = time.time() - start_time
                if execution_time > 0.1:  # Warn if control loop takes longer than 100ms
                    print(f"[{self.name}] WARNING: Control loop took {execution_time:.3f}s (> 0.1s)")
            except Exception as e:
                print(f"[{self.name}] ERROR in control loop: {e}")
                # Emergency stop on error
                if self.cmd_vel_pub:
                    cmd_msg = Twist()
                    cmd_msg.linear.x = 0.0
                    cmd_msg.angular.z = 0.0
                    self.cmd_vel_pub.publish(cmd_msg)
        else:
            # Stop the robot when not pushing
            if self.cmd_vel_pub:
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)

    def control_loop(self):
        """Main MPC control loop - optimized with reduced solving frequency"""
        # Check if we have a valid stored control sequence we can use
        if self.control_sequence is not None:
            # Use the stored control sequence if available
            if self.apply_stored_control():
                self.advance_control_step()
                return
        
        if not self.ref_trajectory:
            print(f"[{self.name}] Warning: No reference trajectory available")
            return

        with self.state_lock:
            try:
                # Calculate progress to determine if we should continue MPC control
                curr_pos = np.array([self.current_state[0], self.current_state[1]])
                final_pos = np.array([self.ref_trajectory[-1][0], self.ref_trajectory[-1][1]])
                dist_to_final = np.linalg.norm(curr_pos - final_pos)
                
                # Check for success condition first
                pushing_complete = self._check_parcel_in_relay_range()
                
                # Only run MPC if not complete and trajectory index is valid
                if not pushing_complete:
                    print(f"[{self.name}] MPC control - index: {self.trajectory_index}/{len(self.ref_trajectory)}, dist_to_final: {dist_to_final:.3f}m")
                    
                    # Check if we need replanning
                    needs_replanning = (
                        self.control_sequence is None or 
                        self.control_step >= self.mpc.N_c - 1
                    )
                    
                    if needs_replanning:
                        # Initialize variables
                        best_idx = self.trajectory_index
                        min_dist = 0.0
                        
                        # Only find closest reference point during initial planning or when far off track
                        if not hasattr(self, '_initial_index_set') or self.trajectory_index == 0:
                            curr_pos = np.array([self.current_state[0], self.current_state[1]])
                            min_dist = float('inf')
                            
                            # Look through entire trajectory to find the closest reference point (initial load only)
                            for idx in range(len(self.ref_trajectory)):
                                ref_pos = np.array([self.ref_trajectory[idx][0], self.ref_trajectory[idx][1]])
                                dist = np.linalg.norm(curr_pos - ref_pos)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_idx = idx
                            
                            # Update index based on closest point for initial reference
                            self.trajectory_index = best_idx
                            self._initial_index_set = True
                            print(f"[{self.name}] Initial closest reference point found at index {best_idx}, distance: {min_dist:.3f}m")
                            
                            # Update pushing estimated time after trajectory index change
                            self._update_pushing_estimated_time()
                        self._update_pushing_estimated_time()
                        # Prepare reference trajectory arrays
                        ref_array = np.zeros((5, self.mpc.N+1))
                        
                        # Fill reference trajectory directly from original trajectory points
                        for i in range(self.mpc.N + 1):
                            if self.trajectory_index + i < len(self.ref_trajectory):
                                # Use normal trajectory point
                                traj_idx = self.trajectory_index + i
                                ref_point = self.ref_trajectory[traj_idx]
                            else:
                                # Trajectory index has reached the end - use last state as target
                                ref_point = self.ref_trajectory[-1]  # Last state of trajectory
                                print(f"[{self.name}] Using last trajectory state as target for MPC horizon point {i}")
                            
                            ref_array[0, i] = ref_point[0]  # x
                            ref_array[1, i] = ref_point[1]  # y
                            ref_array[2, i] = ref_point[2]  # theta
                            ref_array[3, i] = ref_point[3]  # v
                            ref_array[4, i] = ref_point[4]  # omega
                    
                        current_state = self.current_state.copy()
                        
                        try:
                            self.mpc.set_reference_trajectory(ref_array)
                            u_sequence = self.mpc.update(current_state)
                            
                            # Calculate and output detailed control errors
                            self._calculate_and_output_control_errors(current_state, ref_array)
                            
                            # Check if controls are valid
                            if u_sequence is None or np.isnan(u_sequence).any():
                                print(f"[{self.name}] ERROR: MPC returned invalid control values - using safe fallback controls")
                                print(f"[{self.name}] Control sequence type: {type(u_sequence)}, shape: {u_sequence.shape if u_sequence is not None else 'None'}")
                                if u_sequence is not None:
                                    print(f"[{self.name}] Control sequence values: {u_sequence}")
                                    print(f"[{self.name}] NaN values present: {np.isnan(u_sequence).any()}")
                                cmd_msg = Twist()
                                cmd_msg.linear.x = 0.0
                                cmd_msg.angular.z = 0.0
                                self.cmd_vel_pub.publish(cmd_msg)
                                print(f"[{self.name}] Fallback control applied: v={cmd_msg.linear.x:.2f}, ω={cmd_msg.angular.z:.2f}")
                                return
                            
                            # Store the N_c control steps for future use
                            self.control_sequence = u_sequence
                            self.control_step = 0
                            
                            # Apply first control command 
                            cmd_msg = Twist()
                            cmd_msg.linear.x = float(u_sequence[0, 0])
                            cmd_msg.angular.z = float(u_sequence[1, 0])
                            
                            # Check for goal proximity - slow down when close to final waypoint
                            if dist_to_final < 0.25:
                                slow_factor = max(0.3, dist_to_final / 0.25)
                                cmd_msg.linear.x *= slow_factor
                                cmd_msg.angular.z *= min(0.8, slow_factor * 1.2)
                                print(f"[{self.name}] Close to goal ({dist_to_final:.2f}m), slowing down by factor {slow_factor:.2f}")
                            
                            # Publish command
                            self.cmd_vel_pub.publish(cmd_msg)
                            
                            # Log the control action
                            print(f'[{self.name}] MPC control: v={cmd_msg.linear.x:.3f}, ω={cmd_msg.angular.z:.3f} [PUBLISHED]')
                            
                            # Ensure we make progress, but don't go beyond trajectory length
                            # When at the end, keep trajectory_index at the last valid index
                            if self.trajectory_index < len(self.ref_trajectory) - 1:
                                self.trajectory_index = max(self.trajectory_index + 1, best_idx)
                            else:
                                # At end of trajectory, keep using last state as target
                                print(f"[{self.name}] At end of trajectory, maintaining last state as target")
                            
                            print(f"[{self.name}] Trajectory index: {self.trajectory_index}/{len(self.ref_trajectory)} (max: {len(self.ref_trajectory)-1})")
                            
                            # Update pushing estimated time after trajectory index change
                            self._update_pushing_estimated_time()
                            
                            # Publish predicted trajectory
                            self._publish_predicted_trajectory()
                            
                        except Exception as e:
                            print(f'[{self.name}] Control error: {str(e)}')
                        
                        # Publish reference trajectory for visualization
                        self._publish_reference_trajectory()
                
                else:
                    # If pushing is complete or at end of trajectory, stop the robot
                    cmd_msg = Twist()
                    cmd_msg.linear.x = 0.0
                    cmd_msg.angular.z = 0.0
                    self.cmd_vel_pub.publish(cmd_msg)
                    print(f"[{self.name}] Push complete or trajectory finished - stopping robot")
                
            except Exception as e:
                print(f"[{self.name}] Error in control loop: {e}")
    
    def _calculate_and_output_control_errors(self, current_state, ref_array):
        """Calculate and output control errors between current state and reference state"""
        try:
            # Current position error (robot vs reference at current time step)
            current_pos_error = np.sqrt((current_state[0] - ref_array[0, 0])**2 + 
                                       (current_state[1] - ref_array[1, 0])**2)
            current_angle_error = abs(current_state[2] - ref_array[2, 0])
            # Normalize angle error to [-pi, pi]
            current_angle_error = min(current_angle_error, 2*np.pi - current_angle_error)
            
            # Output only position and angle errors
            print(f"[{self.name}] Control Errors: pos={current_pos_error:.4f}m, θ={current_angle_error:.4f}rad({np.degrees(current_angle_error):.1f}°)")
            
            # Store errors for potential logging or further analysis
            self.last_control_errors = {
                'current_pos_error': current_pos_error,
                'current_angle_error': current_angle_error
            }
            
        except Exception as e:
            print(f"[{self.name}] Error calculating control errors: {e}")
            # Set default error values if calculation fails
            self.last_control_errors = {
                'current_pos_error': float('inf'),
                'current_angle_error': float('inf')
            }
    
    def initialise(self):
        """Initialize the pushing behavior"""
        print(f"[{self.name}] INITIALISE DEBUG: Called at {time.time()}")
        self.start_time = time.time()
        self.last_time = self.node.get_clock().now() if self.node else None
        self.pushing_active = True
        self.trajectory_index = 0
        
        # Update pushing estimated time in blackboard when resetting trajectory index
        self._update_pushing_estimated_time()
        
        # Reset control sequence state
        self.control_sequence = None
        self.control_step = 0
        
        # Reset initial index finding flag to trigger closest reference point search
        self._initial_index_set = False
        
        # Create and start ROS timer for control loop at 10Hz (0.1s)
        if self.node:
            # Clean up any existing timer first
            if self.control_timer:
                self.control_timer.cancel()
                self.control_timer = None
                print(f"[{self.name}] DEBUG: Cancelled existing ROS timer")
            
            # Create callback group for multi-threading
            callback_group = ReentrantCallbackGroup()
            
            # Create new timer for 10Hz control frequency with multi-threading support
            self.control_timer = self.node.create_timer(
                self.dt, 
                self.control_timer_callback,
                callback_group=callback_group
            )
            print(f"[{self.name}] DEBUG: Created multi-threaded ROS timer for control loop at {1/self.dt:.1f} Hz (every {self.dt}s)")
            print(f"[{self.name}] DEBUG: Timer created with period: {self.dt}, expected frequency: {1/self.dt:.1f}Hz")
        else:
            print(f"[{self.name}] WARNING: No ROS node available, cannot create control timer")
        
        self.feedback_message = "Starting push operation with MPC trajectory following..."
        print(f"[{self.name}] Push behavior initialized with ROS timer control")
    
    def update(self):
        """Update pushing behavior status - behavior tree logic only"""
        if self.start_time is None:
            self.start_time = time.time()
        
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor
        # which handles all callbacks including our timer automatically
        
        # Check if parcel index changed in blackboard and update subscription if needed
        try:
            blackboard_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
            if self.last_parcel_index != blackboard_parcel_index:
                self.last_parcel_index = blackboard_parcel_index
                self._update_parcel_subscription(blackboard_parcel_index)
                print(f"[{self.name}] Updated parcel tracking to parcel{blackboard_parcel_index}")
        except Exception as e:
            print(f"[{self.name}] Failed to check blackboard parcel index: {e}")
        
        # PRIMARY SUCCESS CONDITION: Check if parcel has reached relay point
        # This is the ONLY condition for SUCCESS - the push is successful when the parcel
        # reaches the relay point, even if the robot is still moving or the trajectory isn't complete
        if self._check_parcel_in_relay_range():
            print(f"[{self.name}] SUCCESS: Parcel has reached the relay point")
            self.pushing_active = False
            return py_trees.common.Status.SUCCESS
        
        # Always return RUNNING if parcel hasn't reached relay point yet
        # No FAILURE conditions - keep trying until success
        elapsed = time.time() - self.start_time
        
        # Update feedback with distance information
        with self.state_lock:
            if self.parcel_pose and self.relay_pose:
                distance = self._calculate_distance(self.parcel_pose, self.relay_pose)
                trajectory_status = "trajectory complete" if (self.ref_trajectory and self.trajectory_index >= len(self.ref_trajectory) - 1) else f"trajectory index: {self.trajectory_index}"
                self.feedback_message = f"Pushing object... {elapsed:.1f}s elapsed, {trajectory_status}, parcel-relay distance: {distance:.3f}m"
            else:
                self.feedback_message = f"Pushing object... {elapsed:.1f}s elapsed, waiting for pose data"
        
        # Even if trajectory is complete, we continue running until the parcel reaches the relay point
        # The control_timer_callback will maintain a minimal pushing force in the direction of the relay
        if self.ref_trajectory and self.trajectory_index >= len(self.ref_trajectory) - 1:
            # Check if we have pose data to give more detailed feedback
            if self.parcel_pose and self.relay_pose:
                distance = self._calculate_distance(self.parcel_pose, self.relay_pose)
                print(f"[{self.name}] RUNNING: Trajectory complete, continuing to push parcel toward relay point. Distance: {distance:.3f}m")
            else:
                print(f"[{self.name}] RUNNING: Trajectory complete, continuing to push parcel toward relay point")
        
        # Timeout check (adjust based on expected trajectory duration)
        if elapsed >= 120.0: 
            print(f"[{self.name}] FAILURE: Push operation timed out after {elapsed:.1f} seconds")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        self.pushing_active = False
        
        # Stop the robot
        if self.cmd_vel_pub:
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
        
        # Clean up timer
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
        
        print(f"[{self.name}] Push behavior terminated with status: {new_status}")
 


