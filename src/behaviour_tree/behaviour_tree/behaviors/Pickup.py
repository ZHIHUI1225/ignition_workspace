#!/usr/bin/env python3
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
        # MPC parameters - optimized for 3-state control (x, y, theta)
        self.N = 8           # Prediction horizon for good tracking
        self.N_c = 3         # Control horizon for smoother control
        self.dt = 0.1        # Time step
        # Higher position weights for precise tracking
        self.Q = np.diag([100.0, 100.0, 10.0])  # State weights (x, y, theta)
        # Lower control input weights for more aggressive control
        self.R = np.diag([0.05, 0.02])           # Control input weights (v, omega)
        # Much higher terminal position weights for precise goal reaching
        self.F = np.diag([200.0, 200.0, 20.0])  # Terminal cost weights
        
        # Velocity constraints
        self.max_vel = 0.25      # m/s
        self.min_vel = -0.25     # m/s (allow reverse)
        self.max_omega = np.pi/2 # rad/s
        self.min_omega = -np.pi/2 # rad/s
        
        # System dimensions
        self.nx = 3   # Number of states (x, y, theta)
        self.nu = 2   # Number of controls (v, omega)
        
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

        # Cost function - 3-state tracking
        cost = 0
        for k in range(self.N):
            # Position error (x,y)
            pos_error = self.X[:2, k] - self.ref[:2, k]
            cost += ca.mtimes([pos_error.T, self.Q[:2,:2], pos_error])
            
            # Orientation (theta) - scalar operation with angle wrapping
            theta_error = self.X[2, k] - self.ref[2, k]
            # Normalize angle difference to [-pi, pi]
            theta_error = ca.fmod(theta_error + ca.pi, 2*ca.pi) - ca.pi
            cost += self.Q[2,2] * theta_error**2
            
            # Control cost
            cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
            
            # Dynamics constraints
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost - 3-state terminal penalty
        pos_term_error = self.X[:2, -1] - self.ref[:2, -1]
        cost += ca.mtimes([pos_term_error.T, self.F[:2,:2], pos_term_error])
        
        # Terminal orientation error with angle wrapping
        theta_term_error = self.X[2, -1] - self.ref[2, -1]
        theta_term_error = ca.fmod(theta_term_error + ca.pi, 2*ca.pi) - ca.pi
        cost += self.F[2,2] * theta_term_error**2

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Control input constraints (velocity and angular velocity)
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
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
        """ System dynamics for 3-state model: x_next = f(x, u) """
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * ca.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt                  # theta
        )

    def set_reference_trajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def update(self, current_state):
        # Ensure we have a 3-state input
        if len(current_state) != 3:
            current_state = current_state[:3]  # Take only x, y, theta
            
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
        """System dynamics in numpy for warm starting (3-state)"""
        return np.array([
            x[0] + u[0] * np.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * np.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt                  # theta
        ])

    def get_predicted_trajectory(self):
        try:
            return self.opti.debug.value(self.X)
        except:
            return np.zeros((self.nx, self.N+1))

class MobileRobotMPC:
    def __init__(self):
        # MPC parameters - optimized for 3-state control (x, y, theta)
        self.N = 8           # Prediction horizon for good tracking
        self.N_c = 3         # Control horizon for smoother control
        self.dt = 0.1        # Time step
        # Higher position weights for precise tracking
        self.Q = np.diag([100.0, 100.0, 10.0])  # State weights (x, y, theta)
        # Lower control input weights for more aggressive control
        self.R = np.diag([0.05, 0.02])           # Control input weights (v, omega)
        # Much higher terminal position weights for precise goal reaching
        self.F = np.diag([200.0, 200.0, 20.0])  # Terminal cost weights
        
        # Velocity constraints
        self.max_vel = 0.25      # m/s
        self.min_vel = -0.25     # m/s (allow reverse)
        self.max_omega = np.pi/2 # rad/s
        self.min_omega = -np.pi/2 # rad/s
        
        # System dimensions
        self.nx = 3   # Number of states (x, y, theta)
        self.nu = 2   # Number of controls (v, omega)
        
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

        # Cost function - 3-state tracking
        cost = 0
        for k in range(self.N):
            # Position error (x,y)
            pos_error = self.X[:2, k] - self.ref[:2, k]
            cost += ca.mtimes([pos_error.T, self.Q[:2,:2], pos_error])
            
            # Orientation (theta) - scalar operation with angle wrapping
            theta_error = self.X[2, k] - self.ref[2, k]
            # Normalize angle difference to [-pi, pi]
            theta_error = ca.fmod(theta_error + ca.pi, 2*ca.pi) - ca.pi
            cost += self.Q[2,2] * theta_error**2
            
            # Control cost
            cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
            
            # Dynamics constraints
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost - 3-state terminal penalty
        pos_term_error = self.X[:2, -1] - self.ref[:2, -1]
        cost += ca.mtimes([pos_term_error.T, self.F[:2,:2], pos_term_error])
        
        # Terminal orientation error with angle wrapping
        theta_term_error = self.X[2, -1] - self.ref[2, -1]
        theta_term_error = ca.fmod(theta_term_error + ca.pi, 2*ca.pi) - ca.pi
        cost += self.F[2,2] * theta_term_error**2

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Control input constraints (velocity and angular velocity)
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
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
        """ System dynamics for 3-state model: x_next = f(x, u) """
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * ca.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt                  # theta
        )

    def set_reference_trajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def update(self, current_state):
        # Ensure we have a 3-state input
        if len(current_state) != 3:
            current_state = current_state[:3]  # Take only x, y, theta
            
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
        """System dynamics in numpy for warm starting (3-state)"""
        return np.array([
            x[0] + u[0] * np.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * np.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt                  # theta
        ])

    def get_predicted_trajectory(self):
        try:
            return self.opti.debug.value(self.X)
        except:
            return np.zeros((self.nx, self.N+1))

class PickObject(py_trees.behaviour.Behaviour):
    """Pick object behavior using MPC controller for trajectory following"""
    
    def __init__(self, name, robot_namespace="turtlebot0"):
        super().__init__(name)
        self.start_time = None
        self.picking_active = False
        self.node = None
        self.number=extract_namespace_number(robot_namespace)
        self.picking_complete = False
        self.robot_namespace = robot_namespace  # Use provided robot_namespace
        self.case = "simple_maze"  # Default case
        
        # MPC and trajectory following variables
        self.trajectory_data = None
        self.goal = None
        self.target_pose = np.zeros(3)  # x, y, theta
        self.current_state = np.zeros(3)  # x, y, theta
        self.mpc = None
        self.prediction_horizon = 10
        
        # State lock for thread safety
        self.state_lock = threading.Lock()
        
        # Control variables
        self.cmd_vel_pub = None
        self.robot_pose_sub = None
        
        # Control timer for MPC
        self.control_timer = None
        self.dt = 0.1  # 10Hz control frequency
        
        # Relay point tracking
        self.relay_pose_sub = None
        self.relay_pose = None
        self.distance_threshold = 0.14  # Distance threshold for success condition
        
        # Trajectory following variables
        self.trajectory_index = 0  # Current index in trajectory
        self.closest_idx = 0
        self.current_s = 0
        self.lookahead_distance = 0.3
        self._initial_index_set = False  # Flag to track initial closest point finding
        
        # Convergence tracking
        self.parallel_count = 0
        self.last_cross_track_error = 0.0
        
    def setup(self, **kwargs):
        """Setup ROS connections and load trajectory data"""
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
            
            # Setup ROS publishers and subscribers for MPC control
            self._setup_ros_connections()
            
            # Load trajectory data (with replanned trajectory support)
            self._load_trajectory_data()
            
            # Initialize MPC controller
            self._setup_simple_mpc()
            
            print(f"[{self.name}] Setting up pick controller for {self.robot_namespace}, case: {self.case}")
    
    def _setup_ros_connections(self):
        """Setup ROS publishers and subscribers for MPC trajectory following"""
        try:
            # Command velocity publisher
            self.cmd_vel_pub = self.node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            
            # Robot pose subscriber 
            self.robot_pose_sub = self.node.create_subscription(
                Odometry, f'/{self.robot_namespace}/odom_map', 
                self._robot_pose_callback, 10)
            
            # Relay point subscriber
            self.relay_pose_sub = self.node.create_subscription(
                PoseStamped, f'/Relaypoint{self.number}/pose',
                self._relay_pose_callback, 10)
                
            print(f"[{self.name}] ROS connections established for MPC control")
            
        except Exception as e:
            print(f"[{self.name}] Error setting up ROS connections: {e}")
    
    def _relay_pose_callback(self, msg):
        """Update relay point pose"""
        self.relay_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y
        ])
    
    def _robot_pose_callback(self, msg):
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
            yaw = euler_from_quaternion(quat)[2]
            self.current_state[2] = yaw
    
    def _load_trajectory_data(self):
        """Load trajectory data with support for replanned trajectories"""
        try:
            # First try to load replanned trajectory
            replanned_file_path = f'/root/workspace/data/{self.case}/tb{self.number}_Trajectory_replanned.json'
            
            trajectory_file_path = None
            if os.path.exists(replanned_file_path):
                trajectory_file_path = replanned_file_path
                print(f"[{self.name}] Loading replanned trajectory from {replanned_file_path}")
            else:
                print(f"[{self.name}] ERROR: No trajectory file found for {self.robot_namespace}")
                return False
            
            # Load trajectory data
            with open(trajectory_file_path, 'r') as json_file:
                data = json.load(json_file)['Trajectory']
                # Reverse trajectory for picking operation (approach from end)
                self.trajectory_data = data[::-1]
            
            # Set goal as the first point in original trajectory
            self.goal = data[0]
            
            # Calculate target pose (approach position before final goal)
            self.target_pose[0] = self.goal[0] - 0.4 * np.cos(self.goal[2])
            self.target_pose[1] = self.goal[1] - 0.4 * np.sin(self.goal[2])
            self.target_pose[2] = self.goal[2]
            
            # Add interpolation points for smoother approach
            num_interp_points = 5
            interp_points = []
            for i in range(num_interp_points):
                alpha = i / (num_interp_points - 1)  # Interpolation factor (0 to 1)
                interp_x = self.target_pose[0] * alpha + self.goal[0] * (1 - alpha)
                interp_y = self.target_pose[1] * alpha + self.goal[1] * (1 - alpha)
                interp_theta = self.target_pose[2]  # Keep orientation constant
                
                interp_points.append([interp_x, interp_y, interp_theta])
            
            # Add interpolation points to the trajectory (executed last in reverse)
            self.trajectory_data = self.trajectory_data + interp_points
            
            print(f"[{self.name}] Loaded {len(self.trajectory_data)} trajectory points for picking")
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error loading trajectory data: {e}")
            return False
    
    def _setup_simple_mpc(self):
        """Setup full MobileRobotMPC controller for trajectory following"""
        # Create the full MPC controller from manipulation_behaviors.py
        self.mpc = MobileRobotMPC()
        self.prediction_horizon = self.mpc.N
        
        # Initialize control sequence management for N_c control horizon
        self.control_sequence = None
        self.control_step = 0
    
    def initialise(self):
        """Initialize the picking behavior"""
        self.start_time = time.time()
        self.picking_active = True
        self.picking_complete = False
        self.feedback_message = "Starting pick operation..."
        
        # Reset trajectory index and initial index finding flag
        self.trajectory_index = 0
        self._initial_index_set = False
        
        # Set up control timer for MPC
        if self.node and not self.control_timer:
            self.control_timer = self.node.create_timer(self.dt, self.control_timer_callback)
            
        print(f"[{self.name}] Starting to pick object...")
    
    def control_timer_callback(self):
        """Timer callback for MPC control execution"""
        if not self.picking_active or self.picking_complete:
            return
            
        try:
            # Execute MPC control if trajectory following is active
            self._follow_trajectory_with_mpc()
        except Exception as e:
            print(f"[{self.name}] Error in control timer callback: {e}")
    
    def update(self):
        """Update picking behavior - simplified to check completion"""
        if self.start_time is None:
            self.start_time = time.time()
        
        if not self.picking_active:
            return py_trees.common.Status.FAILURE
        
        elapsed = time.time() - self.start_time
        self.feedback_message = f"Following trajectory to pick object... {elapsed:.1f}s elapsed"
        
        # Check if robot is close to target
        with self.state_lock:
            current_state = self.current_state.copy()
            
        distance_to_target = np.sqrt((current_state[0] - self.target_pose[0])**2 + 
                                   (current_state[1] - self.target_pose[1])**2)
        
        # Check if out of relay point range (if relay point exists)
        out_of_range = True
        if self.relay_pose is not None:
            distance_to_relay = np.sqrt((current_state[0] - self.relay_pose[0])**2 + 
                                      (current_state[1] - self.relay_pose[1])**2)
            out_of_range = distance_to_relay > self.distance_threshold
        
        # Success condition: close to target and out of relay range
        if distance_to_target <= 0.05 and out_of_range:
            # Stop robot
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
            
            self.picking_complete = True
            print(f"[{self.name}] Successfully reached pick target!")
            return py_trees.common.Status.SUCCESS
        
        # Timeout check
        if elapsed >= 50.0:
            print(f"[{self.name}] Pick operation timed out")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING

    
    def _mpc_trajectory_following(self):
        """MPC-based trajectory following for picking operation"""
        try:
            # Calculate distance to target
            with self.state_lock:
                current_state = self.current_state.copy()
            
            distance_error = np.sqrt((current_state[0] - self.target_pose[0])**2 + 
                                   (current_state[1] - self.target_pose[1])**2)
            
            # Calculate orientation error
            angle_error = abs(current_state[2] - self.target_pose[2])
            angle_error = min(angle_error, 2*np.pi - angle_error)  # Normalize to [0, pi]
            
            # Define stopping criteria
            position_tolerance = 0.015  # 5cm position tolerance
            orientation_tolerance = 0.1  # ~5.7 degrees orientation tolerance
            
            # Check if picking is complete
            position_reached = distance_error <= position_tolerance
            orientation_reached = angle_error <= orientation_tolerance
            
            if position_reached and orientation_reached:
                # Stop robot
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
                
                self.picking_complete = True
                print(f"[{self.name}] Successfully reached pick target and completed picking!")
                return py_trees.common.Status.SUCCESS
            
            # Continue trajectory following if not at target
            if not self._follow_trajectory_with_mpc():
                print(f"[{self.name}] MPC trajectory following failed")
                return py_trees.common.Status.FAILURE
            
            # Timeout check
            elapsed = time.time() - self.start_time
            if elapsed >= 30.0:  # Extended timeout for trajectory following
                print(f"[{self.name}] Pick operation timed out")
                return py_trees.common.Status.FAILURE
            
            return py_trees.common.Status.RUNNING
            
        except Exception as e:
            print(f"[{self.name}] Error in MPC trajectory following: {e}")
            return py_trees.common.Status.FAILURE
    
    def _follow_trajectory_with_mpc(self):
        """Follow trajectory using MPC controller with control sequence management (similar to PushObject)"""
        try:
            # Check if we have a valid stored control sequence we can use
            if self.control_sequence is not None:
                # Use the stored control sequence if available
                if self._apply_stored_control():
                    self._advance_control_step()
                    return True
            
            # If no valid control sequence or need replanning, run MPC
            with self.state_lock:
                current_state = self.current_state.copy()
            
            # Find closest point initially or when far off track (similar to PushObject)
            if not hasattr(self, '_initial_index_set') or not self._initial_index_set:
                curr_pos = np.array([current_state[0], current_state[1]])
                min_dist = float('inf')
                best_idx = 0
                
                # Look through entire trajectory to find the closest reference point (initial load only)
                for idx in range(len(self.trajectory_data)):
                    ref_pos = np.array([self.trajectory_data[idx][0], self.trajectory_data[idx][1]])
                    dist = np.linalg.norm(curr_pos - ref_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                
                # Update index based on closest point for initial reference
                self.trajectory_index = best_idx
                self._initial_index_set = True
                print(f"[{self.name}] Initial closest reference point found at index {best_idx}, distance: {min_dist:.3f}m")
            
            # Generate reference trajectory from trajectory data using current index
            ref_array = np.zeros((3, self.prediction_horizon + 1))
            
            # Fill reference trajectory directly from trajectory points starting from current index
            for i in range(self.prediction_horizon + 1):
                if self.trajectory_index + i < len(self.trajectory_data):
                    # Use normal trajectory point
                    traj_idx = self.trajectory_index + i
                    point = self.trajectory_data[traj_idx]
                else:
                    # Trajectory index has reached the end - use last state as target
                    point = self.trajectory_data[-1]  # Use final point
                
                ref_array[0, i] = point[0]  # x
                ref_array[1, i] = point[1]  # y
                ref_array[2, i] = point[2]  # theta
            
            # Apply cross-track error correction (like PickupController and PushObject)
            cross_track_gain = 0.5
            if self.trajectory_index < len(self.trajectory_data) - 1:
                closest_point = self.trajectory_data[self.trajectory_index]
                next_point = self.trajectory_data[min(self.trajectory_index + 1, len(self.trajectory_data) - 1)]
                
                # Path direction vector
                path_dx = next_point[0] - closest_point[0]
                path_dy = next_point[1] - closest_point[1]
                path_length = np.sqrt(path_dx**2 + path_dy**2)
                
                if path_length > 0.01:  # Avoid division by zero
                    # Normalized path direction
                    path_dx /= path_length
                    path_dy /= path_length
                    
                    # Vector from path point to robot
                    robot_dx = current_state[0] - closest_point[0]
                    robot_dy = current_state[1] - closest_point[1]
                    
                    # Cross-track error (perpendicular distance from path)
                    cross_track_error = robot_dx * (-path_dy) + robot_dy * path_dx
                    
                    # Correction vector (perpendicular to path, towards path)
                    correction_x = -cross_track_error * (-path_dy) * cross_track_gain
                    correction_y = -cross_track_error * path_dx * cross_track_gain
                    
                    # Limit correction magnitude
                    max_correction = 0.2  # meters
                    correction_magnitude = np.sqrt(correction_x**2 + correction_y**2)
                    if correction_magnitude > max_correction:
                        correction_x *= max_correction / correction_magnitude
                        correction_y *= max_correction / correction_magnitude
                    
                    # Apply correction to first reference point
                    ref_array[0, 0] += correction_x
                    ref_array[1, 0] += correction_y
            
            # Update MPC with reference and get control sequence
            self.mpc.set_reference_trajectory(ref_array)
            u_sequence = self.mpc.update(current_state)
            
            if u_sequence is not None and not np.isnan(u_sequence).any():
                # Store the N_c control steps for future use
                self.control_sequence = u_sequence
                self.control_step = 0
                
                # Apply first control command with velocity scaling for approach
                if self._apply_stored_control():
                    # Advance to next control step for next iteration
                    self._advance_control_step()
                    return True
                else:
                    print(f"[{self.name}] Failed to apply stored control")
                    return False
            else:
                print(f"[{self.name}] MPC returned invalid control")
                return False
                
        except Exception as e:
            print(f"[{self.name}] Error in trajectory following: {e}")
            return False

    def _apply_stored_control(self):
        """Apply the current step from the stored control sequence"""
        if self.control_sequence is None or self.control_step >= self.mpc.N_c:
            return False
        
        # Get current control input
        raw_v = float(self.control_sequence[0, self.control_step])
        raw_omega = float(self.control_sequence[1, self.control_step])
        
        # Apply velocity scaling for approach
        distance_to_target = np.sqrt((self.current_state[0] - self.target_pose[0])**2 + 
                                   (self.current_state[1] - self.target_pose[1])**2)
        
        # Scale velocities based on distance to target
        approach_distance = 0.15
        fine_approach_distance = 0.08
        
        if distance_to_target <= fine_approach_distance:
            scale_factor = 0.1 + 0.1 * (distance_to_target / fine_approach_distance)
        elif distance_to_target <= approach_distance:
            scale_factor = 0.2 + 0.4 * (distance_to_target / approach_distance)
        else:
            scale_factor = 1.0
        
        # Create and publish command
        cmd_msg = Twist()
        cmd_msg.linear.x = raw_v * scale_factor
        cmd_msg.angular.z = raw_omega * scale_factor
        
        self.cmd_vel_pub.publish(cmd_msg)
        
        print(f'[{self.name}] MPC control step {self.control_step+1}/{self.mpc.N_c}: '
              f'v={cmd_msg.linear.x:.3f}, ω={cmd_msg.angular.z:.3f}, '
              f'dist_to_target={distance_to_target:.3f}m, traj_idx={self.trajectory_index}/{len(self.trajectory_data)}')
        
        return True

    def _advance_control_step(self):
        """Advance to the next control step in the stored sequence"""
        if self.control_sequence is None:
            return False
        
        self.control_step += 1
        
        # When using a control step from our sequence, we effectively move forward in our trajectory
        # But don't go beyond the end of the trajectory
        if self.trajectory_index < len(self.trajectory_data) - 1:
            self.trajectory_index += 1
        else:
            # At end of trajectory, keep using last state as target
            print(f"[{self.name}] At end of trajectory in advance_control_step, maintaining last state as target")
        
        # If we've used all our control steps, need a new MPC solve
        if self.control_step >= self.mpc.N_c:
            self.control_step = 0
            self.control_sequence = None
            return False
        
        # Otherwise we can use the next step from our stored sequence
        return True
    
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        self.picking_active = False
        
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
        
        print(f"[{self.name}] Pick behavior terminated with status: {new_status}")