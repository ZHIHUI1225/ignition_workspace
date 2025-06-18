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

def report_node_failure(node_name, error_info, robot_namespace):
    """Report node failure to blackboard"""
    try:
        blackboard_client = py_trees.blackboard.Client(name="failure_reporter")
        blackboard_client.register_key(
            key=f"{robot_namespace}/failure_context",
            access=py_trees.common.Access.WRITE
        )
        
        failure_context = {
            "failed_node": node_name,
            "error_info": error_info,
            "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
        
        blackboard_client.set(f"{robot_namespace}/failure_context", failure_context)
        print(f"[FAILURE] {node_name}: {error_info}")
        
    except Exception as e:
        print(f"[ERROR] Failed to report failure for {node_name}: {e}")

class MobileRobotMPC:
    def __init__(self):
        # MPC parameters - 5-state control (x, y, theta, v, omega)
        self.N = 8           # Prediction horizon for good tracking
        self.N_c = 3         # Control horizon for smoother control
        self.dt = 0.1        # Time step
        
        # Balanced weights for 5-state formulation
        self.Q = np.diag([50.0, 50.0, 5.0, 1.0, 1.0])   # State weights (x, y, theta, v, omega)
        self.R = np.diag([1.0, 1.0])                     # Control input weights (v, omega)
        self.F = np.diag([100.0, 100.0, 10.0, 2.0, 2.0]) # Terminal cost weights
        
        # Conservative velocity constraints
        self.max_vel = 0.25      # m/s
        self.min_vel = -0.1      # m/s (limited reverse)
        self.max_omega = np.pi/3 # rad/s (conservative angular velocity)
        self.min_omega = -np.pi/3 # rad/s
        
        # System dimensions
        self.nx = 5   # Number of states (x, y, theta, v, omega)
        self.nu = 2   # Number of controls (v, omega)
        
        # Reference trajectory
        self.ref_traj = None
        
        # Solver cache for warm starting
        self.last_solution = None
        
        # Initialize MPC
        self.setup_mpc()

    def setup_mpc(self):
        # Robust optimization problem with numerical stability
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N+1)  # State trajectory
        self.U = self.opti.variable(self.nu, self.N)    # Control trajectory
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)          # Initial state
        self.ref = self.opti.parameter(self.nx, self.N+1)  # Reference trajectory

        # Robust cost function with numerical stability
        cost = 0
        for k in range(self.N):
            # State tracking cost - robust formulation
            state_error = self.X[:, k] - self.ref[:, k]
            
            # Position error (x, y) - standard quadratic
            pos_error = state_error[:2]
            cost += ca.mtimes([pos_error.T, self.Q[:2,:2], pos_error])
            
            # Orientation error with angle wrapping
            theta_error = state_error[2]
            # Robust angle wrapping using atan2 for numerical stability
            theta_error_wrapped = ca.atan2(ca.sin(theta_error), ca.cos(theta_error))
            cost += self.Q[2,2] * theta_error_wrapped**2
            
            # Velocity errors - add small regularization for stability
            vel_error = state_error[3:5]
            cost += ca.mtimes([vel_error.T, self.Q[3:5,3:5], vel_error])
            
            # Control cost with regularization
            u_reg = self.U[:, k] + 1e-6  # Small regularization to prevent singularity
            cost += ca.mtimes([u_reg.T, self.R, u_reg])
            
            # Dynamics constraints
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost with robust angle handling
        terminal_error = self.X[:, -1] - self.ref[:, -1]
        
        # Terminal position cost
        pos_term_error = terminal_error[:2]
        cost += ca.mtimes([pos_term_error.T, self.F[:2,:2], pos_term_error])
        
        # Terminal orientation cost with wrapping
        theta_term_error = terminal_error[2]
        theta_term_wrapped = ca.atan2(ca.sin(theta_term_error), ca.cos(theta_term_error))
        cost += self.F[2,2] * theta_term_wrapped**2
        
        # Terminal velocity costs
        vel_term_error = terminal_error[3:5]
        cost += ca.mtimes([vel_term_error.T, self.F[3:5,3:5], vel_term_error])

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Control input constraints
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))
        
        # State constraints for numerical stability
        self.opti.subject_to(self.opti.bounded(-2.0, self.X[3, :], 2.0))  # Linear velocity bounds
        self.opti.subject_to(self.opti.bounded(-np.pi, self.X[4, :], np.pi))  # Angular velocity bounds

        # Robust solver settings
        self.opti.minimize(cost)
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 100,
            'ipopt.max_cpu_time': 0.15,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 10,
            'ipopt.mu_strategy': 'monotone',
            'ipopt.linear_solver': 'mumps',
            'ipopt.hessian_approximation': 'limited-memory'
        }
        
        self.opti.solver('ipopt', opts)

    def robot_model(self, x, u):
        """ 5-state system dynamics: x_next = f(x, u) """
        return ca.vertcat(
            x[0] + x[3] * ca.cos(x[2]) * self.dt,  # x position
            x[1] + x[3] * ca.sin(x[2]) * self.dt,  # y position
            x[2] + x[4] * self.dt,                 # theta
            u[0],                                  # linear velocity (direct control)
            u[1]                                   # angular velocity (direct control)
        )

    def set_reference_trajectory(self, ref_traj):
        self.ref_traj = ref_traj
    
    def set_target_pose(self, target_pose):
        """Set the target pose for precision positioning"""
        self.target_pose = target_pose

    def update(self, current_state):
        # Convert 3-state input to 5-state for 5-state MPC
        if len(current_state) == 3:
            # Add current velocities (assume zero if not provided)
            current_state_5d = np.zeros(5)
            current_state_5d[:3] = current_state  # x, y, theta
            current_state_5d[3] = 0.0  # v (will be estimated or use last known)
            current_state_5d[4] = 0.0  # omega (will be estimated or use last known)
            current_state = current_state_5d
        
        # Set parameters
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.ref, self.ref_traj)
        
        # Simplified warm starting for stability
        if self.last_solution is not None:
            try:
                # Simple shift without expensive propagation
                u_init = np.zeros((self.nu, self.N))
                x_init = np.zeros((self.nx, self.N+1))
                
                # Copy and shift previous solution
                if self.N > 1:
                    u_init[:, :self.N-1] = self.last_solution['u'][:, 1:]
                    u_init[:, self.N-1] = self.last_solution['u'][:, -1]  # Repeat last control
                    
                    x_init[:, :self.N] = self.last_solution['x'][:, 1:]
                    x_init[:, self.N] = self.last_solution['x'][:, -1]  # Use last state
                
                self.opti.set_initial(self.X, x_init)
                self.opti.set_initial(self.U, u_init)
            except:
                # If warm start fails, continue with cold start
                pass
        
        try:
            sol = self.opti.solve()
            
            # Validate solution - check for NaN or extreme values
            u_opt = sol.value(self.U)[:, :self.N_c]
            if np.isnan(u_opt).any() or np.abs(u_opt[0]).max() > 2.0 or np.abs(u_opt[1]).max() > np.pi:
                print(f"MPC solution validation failed - using fallback controller")
                self.last_solution = None  # Clear stale solution
                return np.zeros((self.nu, self.N_c))
            
            # Store solution for warm starting next time
            self.last_solution = {
                'u': sol.value(self.U),
                'x': sol.value(self.X)
            }
            
            # Return all N_c control steps
            return u_opt
            
        except Exception as e:
            print(f"MPC Solver failed: {str(e)}")
            # Clear last solution on failure to avoid using stale data
            self.last_solution = None
            return np.zeros((self.nu, self.N_c))
    
    def get_predicted_trajectory(self):
        try:
            return self.opti.debug.value(self.X)
        except:
            return np.zeros((self.nx, self.N+1))

class PickObject(py_trees.behaviour.Behaviour):
    """Pick object behavior using MPC controller for trajectory following"""
    
    def __init__(self, name, robot_namespace="turtlebot0", timeout=100.0):
        super().__init__(name)
        self.start_time = None
        self.picking_active = False
        self.node = None
        self.number=extract_namespace_number(robot_namespace)
        self.picking_complete = False
        self.robot_namespace = robot_namespace  # Use provided robot_namespace
        self.case = "simple_maze"  # Default case
        self.timeout = timeout  # Timeout in seconds
        
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
            if not self._load_trajectory_data():
                print(f"[{self.name}] CRITICAL ERROR: Failed to load trajectory data for {self.robot_namespace}")
                print(f"[{self.name}] Expected file: /root/workspace/data/{self.case}/tb{self.number}_Trajectory_replanned.json")
                return False
            
            # Initialize MPC controller
            self._setup_simple_mpc()
            
            print(f"[{self.name}] Setting up pick controller for {self.robot_namespace}, case: {self.case}")
            return True
        
        print(f"[{self.name}] ERROR: No ROS node provided in setup")
        return False
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
            dt = 0.1  # Time interval in seconds
            
            for i in range(num_interp_points):
                alpha = i / (num_interp_points - 1)  # Interpolation factor (0 to 1)
                interp_x = self.target_pose[0] * alpha + self.goal[0] * (1 - alpha)
                interp_y = self.target_pose[1] * alpha + self.goal[1] * (1 - alpha)
                interp_theta = self.target_pose[2]  # Keep orientation constant
                
                # Estimate velocity based on distance to next point (delta x, delta y)
                if i < num_interp_points - 1:
                    # Calculate next point for velocity estimation
                    next_alpha = (i + 1) / (num_interp_points - 1)
                    next_x = self.target_pose[0] * next_alpha + self.goal[0] * (1 - next_alpha)
                    next_y = self.target_pose[1] * next_alpha + self.goal[1] * (1 - next_alpha)
                    
                    # Calculate velocity as distance / time
                    delta_x = next_x - interp_x
                    delta_y = next_y - interp_y
                    distance = np.sqrt(delta_x**2 + delta_y**2)
                    estimated_v = distance / dt
                else:
                    # For the last point, use zero velocity (stationary)
                    estimated_v = 0.0
                
                # Add trajectory point with 5 elements (x, y, theta, v, omega)
                # omega = 0 for straight line interpolation
                interp_points.append([interp_x, interp_y, interp_theta, estimated_v, 0.0])
            
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
        
        # Pass target pose to MPC for precision positioning
        if hasattr(self, 'target_pose'):
            self.mpc.set_target_pose(self.target_pose)
        
        # Initialize control sequence management for N_c control horizon
        self.control_sequence = None
        self.control_step = 0
    
    def _find_initial_closest_trajectory_index(self):
        """Find the closest trajectory point to robot's current position at initialization"""
        try:
            # Get current robot position
            with self.state_lock:
                current_state = self.current_state.copy()
            
            if not self.trajectory_data or len(self.trajectory_data) == 0:
                print(f"[{self.name}] No trajectory data available for initial closest index finding")
                return 0
            
            curr_pos = np.array([current_state[0], current_state[1]])
            min_dist = float('inf')
            closest_idx = 0
            
            # Search through entire trajectory to find the closest point
            for idx in range(len(self.trajectory_data)):
                ref_pos = np.array([self.trajectory_data[idx][0], self.trajectory_data[idx][1]])
                dist = np.linalg.norm(curr_pos - ref_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx
            
            print(f"[{self.name}] Initial closest trajectory point found at index {closest_idx}/{len(self.trajectory_data)-1}, distance: {min_dist:.3f}m")
            print(f"[{self.name}] Robot position: ({current_state[0]:.3f}, {current_state[1]:.3f})")
            print(f"[{self.name}] Closest trajectory point: ({self.trajectory_data[closest_idx][0]:.3f}, {self.trajectory_data[closest_idx][1]:.3f})")
            
            return closest_idx
            
        except Exception as e:
            print(f"[{self.name}] Error finding initial closest trajectory index: {e}")
            return 0

    def _wait_for_initial_pose(self, timeout=5.0):
        """Wait for initial robot pose to be received before finding closest trajectory point"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.state_lock:
                # Check if we have received a valid pose (not all zeros)
                if not np.allclose(self.current_state, [0.0, 0.0, 0.0]):
                    print(f"[{self.name}] Initial robot pose received: ({self.current_state[0]:.3f}, {self.current_state[1]:.3f}, {self.current_state[2]:.3f})")
                    return True
            time.sleep(0.1)  # Wait 100ms before checking again
        
        print(f"[{self.name}] Warning: Timeout waiting for initial robot pose, using default position")
        return False

    def initialise(self):
        """Initialize the picking behavior"""
        self.start_time = time.time()
        self.picking_active = True
        self.picking_complete = False
        self.feedback_message = f"[{self.robot_namespace}] Starting pick operation..."
        
        # Reset trajectory index and initial index finding flag
        self.trajectory_index = 0
        self._initial_index_set = False
        
        # Set up control timer for MPC
        if self.node and not self.control_timer:
            self.control_timer = self.node.create_timer(self.dt, self.control_timer_callback)
        
        # Find initial closest trajectory index
        self._wait_for_initial_pose(timeout=5.0)  # Wait for initial pose
        self.closest_idx = self._find_initial_closest_trajectory_index()
        self.trajectory_index = self.closest_idx  # Start at closest index
        
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
        
        # Check if trajectory data was loaded successfully
        if self.trajectory_data is None:
            report_node_failure(self.name, "No trajectory data available - cannot proceed with picking", self.robot_namespace)
            print(f"[{self.name}] FAILURE: No trajectory data available - cannot proceed with picking")
            return py_trees.common.Status.FAILURE
        
        if not self.picking_active:
            report_node_failure(self.name, "Picking is not active", self.robot_namespace)
            print(f"[{self.name}] FAILURE: Picking is not active")
            return py_trees.common.Status.FAILURE
        
        # Check if setup was successful
        if not hasattr(self, 'mpc') or self.mpc is None:
            report_node_failure(self.name, "MPC controller not initialized", self.robot_namespace)
            print(f"[{self.name}] FAILURE: MPC controller not initialized")
            return py_trees.common.Status.FAILURE
        
        if not hasattr(self, 'target_pose') or self.target_pose is None:
            report_node_failure(self.name, "Target pose not set", self.robot_namespace)
            print(f"[{self.name}] FAILURE: Target pose not set")
            return py_trees.common.Status.FAILURE
        
        elapsed = time.time() - self.start_time
        self.feedback_message = f"[{self.robot_namespace}] Following trajectory to pick object... {elapsed:.1f}s elapsed"
        
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
        if distance_to_target <= 0.03 and out_of_range:
            # Stop robot
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
            
            self.picking_complete = True
            print(f"[{self.name}] Successfully reached pick target!")
            return py_trees.common.Status.SUCCESS
        
        # Timeout check
        if elapsed >= self.timeout:
            report_node_failure(self.name, f"PickObject timeout after {self.timeout}s - failed to reach target", self.robot_namespace)
            print(f"[{self.name}] Pick operation timed out after {self.timeout}s")
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
            position_tolerance = 0.025 # 2cm position tolerance
            orientation_tolerance = 0.2  # ~17 degrees orientation tolerance (increased from ~5.7 degrees)
            
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
            if elapsed >= self.timeout:  # Extended timeout for trajectory following
                print(f"[{self.name}] Pick operation timed out after {self.timeout}s")
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
            
            # Always find the closest trajectory point to ensure accurate reference
            curr_pos = np.array([current_state[0], current_state[1]])
            min_dist = float('inf')
            closest_idx = 0
            
            # Search for closest point in a local window around current index for efficiency
            search_window = 20  # Search +/- 20 points around current index
            start_idx = max(0, self.trajectory_index - search_window)
            end_idx = min(len(self.trajectory_data), self.trajectory_index + search_window + 1)
            
            # First search in local window
            for idx in range(start_idx, end_idx):
                ref_pos = np.array([self.trajectory_data[idx][0], self.trajectory_data[idx][1]])
                dist = np.linalg.norm(curr_pos - ref_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx
            
            # If we're far from trajectory (> 0.5m), search entire trajectory
            if min_dist > 0.5:
                print(f"[{self.name}] Robot far from trajectory ({min_dist:.3f}m), searching entire trajectory")
                for idx in range(len(self.trajectory_data)):
                    ref_pos = np.array([self.trajectory_data[idx][0], self.trajectory_data[idx][1]])
                    dist = np.linalg.norm(curr_pos - ref_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx
            
            # Update trajectory index to closest point
            self.trajectory_index = closest_idx
            
            # Log trajectory index updates for debugging
            if not hasattr(self, '_last_logged_index') or abs(self.trajectory_index - self._last_logged_index) > 5:
                print(f"[{self.name}] Updated trajectory index to {self.trajectory_index}, distance to closest point: {min_dist:.3f}m")
                self._last_logged_index = self.trajectory_index
            
            # Generate reference trajectory for 5-state MPC from trajectory data using updated closest index
            ref_array = np.zeros((5, self.prediction_horizon + 1))
            
            # Check if we're close to target - use stationary reference for precision
            distance_to_target = np.sqrt((current_state[0] - self.target_pose[0])**2 + 
                                       (current_state[1] - self.target_pose[1])**2)
            
            if distance_to_target <= 0.15:  # Within 15cm, use stationary target reference
                # Use target pose for entire prediction horizon for precision
                for i in range(self.prediction_horizon + 1):
                    ref_array[0, i] = self.target_pose[0]  # x
                    ref_array[1, i] = self.target_pose[1]  # y  
                    ref_array[2, i] = self.target_pose[2]  # theta
                    ref_array[3, i] = 0.0  # v (stationary)
                    ref_array[4, i] = 0.0  # omega (stationary)
                print(f"[{self.name}] Using stationary target reference for precision positioning")
            else:
                # Fill reference trajectory from trajectory points starting from closest index
                for i in range(self.prediction_horizon + 1):
                    # Calculate trajectory index for this prediction step
                    traj_idx = self.trajectory_index + i
                    
                    if traj_idx < len(self.trajectory_data):
                        # Use trajectory point
                        point = self.trajectory_data[traj_idx]
                        ref_array[0, i] = point[0]  # x
                        ref_array[1, i] = point[1]  # y
                        ref_array[2, i] = point[2]  # theta
                        # Use actual velocity values from trajectory data
                        ref_array[3, i] = point[3]  # v
                        ref_array[4, i] = point[4]  # omega
                    else:
                        # Trajectory index has reached the end - use target pose
                        ref_array[0, i] = self.target_pose[0]  # x
                        ref_array[1, i] = self.target_pose[1]  # y
                        ref_array[2, i] = self.target_pose[2]  # theta
                        ref_array[3, i] = 0.0  # v (stationary)
                        ref_array[4, i] = 0.0  # omega (stationary)
            
            # Apply cross-track error correction using the closest trajectory point
            cross_track_gain = 0.2  # Reduced from 0.5 to make orientation less aggressive
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
                    
                    # Vector from closest path point to robot
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
                    
                    # Apply correction to first reference point (only if not using stationary reference)
                    if distance_to_target > 0.15:
                        ref_array[0, 0] += correction_x
                        ref_array[1, 0] += correction_y
                        
                    # Log cross-track error for debugging
                    if abs(cross_track_error) > 0.05:  # Only log significant errors
                        print(f"[{self.name}] Cross-track error: {cross_track_error:.3f}m, correction: ({correction_x:.3f}, {correction_y:.3f})")
            
            # Update MPC with reference and get control sequence
            self.mpc.set_reference_trajectory(ref_array)
            # Update target pose in MPC for distance-based reference switching
            self.mpc.set_target_pose(self.target_pose)
            u_sequence = self.mpc.update(current_state)
            
            # Check if MPC provided valid solution
            if u_sequence is not None and not np.isnan(u_sequence).any() and not np.allclose(u_sequence, 0.0):
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
                    return self._apply_fallback_control(current_state)
            else:
                print(f"[{self.name}] MPC returned invalid control, using fallback controller")
                return self._apply_fallback_control(current_state)
                
        except Exception as e:
            print(f"[{self.name}] Error in trajectory following: {e}")
            return False
    
    def _apply_fallback_control(self, current_state):
        """Enhanced fallback controller for pickup behavior when MPC fails"""
        try:
            # Get target from current trajectory index or target pose
            if self.trajectory_index < len(self.trajectory_data):
                target_x = self.trajectory_data[self.trajectory_index][0]
                target_y = self.trajectory_data[self.trajectory_index][1]
                target_theta = self.trajectory_data[self.trajectory_index][2]
            else:
                target_x = self.target_pose[0]
                target_y = self.target_pose[1]
                target_theta = self.target_pose[2]
            
            # Calculate errors
            dx = target_x - current_state[0]
            dy = target_y - current_state[1]
            distance = np.sqrt(dx**2 + dy**2)
            bearing_to_target = np.arctan2(dy, dx)
            
            # Normalize angle difference
            angle_error = bearing_to_target - current_state[2]
            while angle_error > np.pi:
                angle_error -= 2*np.pi
            while angle_error < -np.pi:
                angle_error += 2*np.pi
            
            # Orientation error to target orientation
            orientation_error = target_theta - current_state[2]
            while orientation_error > np.pi:
                orientation_error -= 2*np.pi
            while orientation_error < -np.pi:
                orientation_error += 2*np.pi
            
            # Distance-based control strategy
            if distance < 0.05:
                # Very close - just align orientation
                v_cmd = 0.0
                w_cmd = 2.0 * orientation_error
                w_cmd = np.clip(w_cmd, -np.pi/4, np.pi/4)
            elif distance < 0.15:
                # Close approach - precise positioning
                kp_linear = 1.5
                kp_angular = 2.5
                
                # Move toward target with orientation correction
                v_cmd = kp_linear * distance
                w_cmd = kp_angular * angle_error + 0.5 * orientation_error
                
                v_cmd = np.clip(v_cmd, 0.0, 0.15)
                w_cmd = np.clip(w_cmd, -np.pi/3, np.pi/3)
            else:
                # Far approach - trajectory following mode
                kp_linear = 1.0
                kp_angular = 1.5
                
                # Align with path direction first if large angle error
                if abs(angle_error) > np.pi/3:
                    v_cmd = 0.05  # Very slow forward motion while turning
                    w_cmd = kp_angular * angle_error
                else:
                    v_cmd = kp_linear * distance * np.cos(angle_error)
                    w_cmd = kp_angular * angle_error
                
                v_cmd = np.clip(v_cmd, 0.0, 0.4)
                w_cmd = np.clip(w_cmd, -np.pi/2, np.pi/2)
            
            # Publish control command
            cmd_msg = Twist()
            cmd_msg.linear.x = float(v_cmd)
            cmd_msg.angular.z = float(w_cmd)
            self.cmd_vel_pub.publish(cmd_msg)
            
            print(f"[{self.name}] Fallback control: dist={distance:.3f}m, angle_err={angle_error:.3f}rad, v={v_cmd:.3f}, w={w_cmd:.3f}")
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error in fallback controller: {e}")
            # Emergency stop
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
            return False
    
    
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
    
    def _apply_stored_control(self):
        """Apply the current step from the stored control sequence"""
        if self.control_sequence is None or self.control_step >= self.mpc.N_c:
            return False
        
        # Get current control input
        raw_v = float(self.control_sequence[0, self.control_step])
        raw_omega = float(self.control_sequence[1, self.control_step])
        
        # Apply velocity scaling for approach
        with self.state_lock:
            current_state = self.current_state.copy()
            
        distance_to_target = np.sqrt((current_state[0] - self.target_pose[0])**2 + 
                                   (current_state[1] - self.target_pose[1])**2)
        
        # Calculate distance to closest trajectory point for monitoring
        if self.trajectory_index < len(self.trajectory_data):
            closest_traj_point = self.trajectory_data[self.trajectory_index]
            distance_to_traj = np.sqrt((current_state[0] - closest_traj_point[0])**2 + 
                                     (current_state[1] - closest_traj_point[1])**2)
        else:
            distance_to_traj = 0.0
        
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
        
        # Enhanced logging with trajectory following info
        print(f'[{self.name}] MPC control step {self.control_step+1}/{self.mpc.N_c}: '
              f'v={cmd_msg.linear.x:.3f}, ω={cmd_msg.angular.z:.3f}, '
              f'dist_to_target={distance_to_target:.3f}m, dist_to_traj={distance_to_traj:.3f}m, '
              f'traj_idx={self.trajectory_index}/{len(self.trajectory_data)}')
        
        return True

    def _advance_control_step(self):
        """Advance to the next control step in the stored sequence"""
        if self.control_sequence is None:
            return False
        
        self.control_step += 1
        
        # Advanced trajectory progression: only advance index if robot has moved forward along trajectory
        if self.control_step == 1:  # Only check on first control step application
            with self.state_lock:
                current_state = self.current_state.copy()
            
            curr_pos = np.array([current_state[0], current_state[1]])
            
            # Look ahead in trajectory to see if robot should advance
            lookahead_distance = 0.1  # 10cm lookahead
            max_lookahead_idx = min(self.trajectory_index + 5, len(self.trajectory_data) - 1)
            
            for check_idx in range(self.trajectory_index + 1, max_lookahead_idx + 1):
                check_pos = np.array([self.trajectory_data[check_idx][0], self.trajectory_data[check_idx][1]])
                distance_to_check = np.linalg.norm(curr_pos - check_pos)
                
                # If robot is close to a forward point, advance to that point
                if distance_to_check < lookahead_distance:
                    self.trajectory_index = check_idx
                    print(f"[{self.name}] Advanced trajectory index to {self.trajectory_index} (robot close to forward point)")
                    break
        
        # If we've used all our control steps, need a new MPC solve
        if self.control_step >= self.mpc.N_c:
            self.control_step = 0
            self.control_sequence = None
            return False
        
        # Otherwise we can use the next step from our stored sequence
        return True