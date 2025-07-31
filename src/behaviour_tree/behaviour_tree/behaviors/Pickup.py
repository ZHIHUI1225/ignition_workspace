#!/usr/bin/env python3
import py_trees
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Int32, Float64
import time
import json
import numpy as np
import casadi as ca
import os
import math
import threading
import traceback  # Add traceback import for error handling
import re
import copy
import tf_transformations as tf
from tf_transformations import euler_from_quaternion, quaternion_from_euler
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
    def __init__(self, dt=None):
        # MPC parameters - optimized for real-time performance (2Hz control)
        self.N = 6           # Reduced prediction horizon from 8 for faster computation
        self.N_c = 2         # Reduced control horizon from 3 for faster solve
        self.dt = dt if dt is not None else 0.5  # Use provided dt or default to 0.5s (2Hz)
        
        # Weights for 3-state tracking (x, y, theta) with 5-state dynamics
        self.Q = np.diag([50.0, 50.0, 1.0])             # State weights (x, y, theta only)
        self.R = np.diag([1.0, 1.0])                     # Control input weights (v, omega)
        self.F = np.diag([100.0, 100.0, 2.0])           # Terminal cost weights (x, y, theta only)
        
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

        # Robust cost function with numerical stability - 3-state tracking only
        cost = 0
        for k in range(self.N):
            # State tracking cost - only track position and orientation (x, y, theta)
            state_error = self.X[:3, k] - self.ref[:3, k]  # Only use first 3 states
            
            # Position error (x, y) - standard quadratic
            pos_error = state_error[:2]
            cost += ca.mtimes([pos_error.T, self.Q[:2,:2], pos_error])
            
            # Orientation error with angle wrapping
            theta_error = state_error[2]
            # Robust angle wrapping using atan2 for numerical stability
            theta_error_wrapped = ca.atan2(ca.sin(theta_error), ca.cos(theta_error))
            cost += self.Q[2,2] * theta_error_wrapped**2
            
            # Control cost with regularization
            u_reg = self.U[:, k] + 1e-6  # Small regularization to prevent singularity
            cost += ca.mtimes([u_reg.T, self.R, u_reg])
            
            # Dynamics constraints
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost with robust angle handling - 3-state tracking only
        terminal_error = self.X[:3, -1] - self.ref[:3, -1]  # Only use first 3 states
        
        # Terminal position cost
        pos_term_error = terminal_error[:2]
        cost += ca.mtimes([pos_term_error.T, self.F[:2,:2], pos_term_error])
        
        # Terminal orientation cost with wrapping
        theta_term_error = terminal_error[2]
        theta_term_wrapped = ca.atan2(ca.sin(theta_term_error), ca.cos(theta_term_error))
        cost += self.F[2,2] * theta_term_wrapped**2

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Control input constraints
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))
        
        # State constraints for numerical stability
        self.opti.subject_to(self.opti.bounded(-2.0, self.X[3, :], 2.0))  # Linear velocity bounds
        self.opti.subject_to(self.opti.bounded(-np.pi, self.X[4, :], np.pi))  # Angular velocity bounds

        # Optimized solver settings for real-time performance (target <100ms for 2Hz control)
        self.opti.minimize(cost)
        
        # 🔧 CRITICAL: Set environment variables to control BLAS/LAPACK threading for consistent timing
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1' 
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 100,              
            'ipopt.max_cpu_time': 0.4,        
            'ipopt.tol': 5e-3,                
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 5,        
            'ipopt.mu_strategy': 'monotone',
            'ipopt.linear_solver': 'mumps',
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.warm_start_init_point': 'yes',  # Enable warm starting for speed
            'ipopt.nlp_scaling_method': 'none',    # Disable scaling for speed
            'ipopt.constr_viol_tol': 1e-2,        # Relaxed constraint tolerance
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
    
    def __init__(self, name, robot_namespace="robot0", timeout=100.0, estimated_time=55.0, dt=0.5, case="simple_maze"):
        super().__init__(name)
        self.start_time = None
        self.picking_active = False
        self.node = None
        self.number=extract_namespace_number(robot_namespace)
        self.picking_complete = False
        self.robot_namespace = robot_namespace  # Use provided robot_namespace
        self.case = case  # Use provided case instead of hardcoded default
        self.timeout = timeout  # Timeout in seconds
        self.estimated_time = estimated_time  # Estimated time for pushing operation in seconds
        
        # Initialize thread-safe state lock
        self.state_lock = threading.Lock()
        
        # MPC and trajectory following variables
        self.trajectory_data = None
        self.goal = None
        self.target_pose = np.zeros(3)  # x, y, theta
        self.current_state = np.zeros(3)  # x, y, theta
        self.mpc = None
        self.prediction_horizon = 10
        
        # Control variables
        self.cmd_vel_pub = None
        self.robot_pose_sub = None
        self.estimated_time_pub = None  # Publisher for pushing estimated time
        
        # Replace dedicated control thread with ROS timer
        self.control_timer = None
        self.control_thread_active = False  # Keeping for compatibility with existing code
        self.dt = dt  # Control frequency (default: 0.5s for 2Hz)
        
        # Relay point tracking
        self.relay_pose_sub = None
        self.relay_pose = None
        self.distance_threshold = 0.10 # Distance threshold for success condition
        
        # Trajectory following variables
        self.trajectory_index = 0  # Current index in trajectory
        self.closest_idx = 0
        self.current_s = 0
        self.lookahead_distance = 0.3
        self._initial_index_set = False  # Flag to track initial closest point finding
        
        # Convergence tracking
        self.parallel_count = 0
        self.last_cross_track_error = 0.0
        
        # Estimated time publishing tracking
        self.last_estimated_time_publish = 0.0
        self.estimated_time_publish_interval = 1.0  # Publish every 5 seconds
        
        # State for the behavior
        self.state = 'idle'  # idle, moving_to_parcel, moving_to_relay, finished
        
        # Freshness checking variables
        self._last_robot_callback_time = None
        self._robot_callback_count = 0
        self._freshness_check_count = 0
        
    def start_control_thread(self):
        """Start the control timer for 2Hz control loop using ROS timer"""
        if self.control_thread_active or self.control_timer is not None:
            return  # Already running
        
        # Verify prerequisites before starting timer
        trajectory_valid = hasattr(self, 'trajectory_data') and self.trajectory_data is not None
        mpc_valid = hasattr(self, 'mpc') and self.mpc is not None
        current_state_valid = not np.allclose(self.current_state, [0.0, 0.0, 0.0])
            
        self.control_thread_active = True
        
        # Use the shared callback group from the node to create a timer
        callback_group = None
        if hasattr(self, 'control_callback_group'):
            callback_group = self.control_callback_group
            print(f"[{self.name}] Using self.control_callback_group")
        elif hasattr(self.node, 'shared_callback_manager'):
            callback_group = self.node.shared_callback_manager.get_group('control')
            print(f"[{self.name}] Using node.shared_callback_manager.get_group('control')")
        elif hasattr(self.node, 'robot_dedicated_callback_group'):
            callback_group = self.node.robot_dedicated_callback_group
            print(f"[{self.name}] Using node.robot_dedicated_callback_group")
        else:
            print(f"[{self.name}] ⚠️ No suitable callback group found!")
        
        try:
            # Create a ROS timer with the appropriate callback group
            print(f"[{self.name}] Creating control timer with period {self.dt}s")
            self.control_timer = self.node.create_timer(
                self.dt,  # 2Hz frequency = 0.5s period
                self.control_step,  # Use the existing control_step as callback
                callback_group=callback_group
            )
            print(f"[{self.name}] ✅ Control timer created and started successfully")
        except Exception as e:
            self.control_thread_active = False
            print(f"[{self.name}] ❌ Failed to create control timer: {e}")
            import traceback
            print(f"[{self.name}] Exception traceback: {traceback.format_exc()}")
        
    def stop_control_thread(self):
        """Stop the control timer"""
        self.control_thread_active = False
        if self.control_timer is not None:
            self.node.destroy_timer(self.control_timer)
            self.control_timer = None
    
                    
    def control_step(self):
        """Single control step - now called directly by ROS timer"""
        import time  # Import time for timing diagnostics
        
        # Start timing for entire control step
        control_step_start_time = time.time()
        
        # Enhanced debugging for second launch issues
        if not hasattr(self, '_control_step_call_count'):
            self._control_step_call_count = 0
            print(f"[{self.name}] 🚀 First control step called!")
        self._control_step_call_count += 1
        
        # Track control step timing
        if not hasattr(self, '_last_control_step_time'):
            self._last_control_step_time = control_step_start_time
        
        time_since_last_control = control_step_start_time - self._last_control_step_time
        
        # Skip debug print for control step initialization
        
        if not self.control_thread_active or not self.picking_active:
            print(f"[{self.name}] ⚠️ Control inactive: control_thread_active={self.control_thread_active}, picking_active={self.picking_active}")
            return
            
        # Debug: Check critical prerequisites
        if not hasattr(self, 'mpc') or self.mpc is None:
            print(f"[{self.name}] Control aborted: MPC controller not initialized")
            return
            
        if not hasattr(self, 'trajectory_data') or self.trajectory_data is None:
            print(f"[{self.name}] Control aborted: No trajectory data available")
            return
            
        if not hasattr(self, 'cmd_vel_pub') or self.cmd_vel_pub is None:
            print(f"[{self.name}] Control aborted: cmd_vel publisher not initialized")
            return
            
        # Try to acquire state lock non-blocking
        lock_start_time = time.time()
        state_acquired = self.state_lock.acquire(blocking=False)
        if not state_acquired:
            print(f"[{self.name}] Could not acquire state lock, skipping control step")
            return
        
        lock_acquired_time = time.time()
        lock_wait_duration = lock_acquired_time - lock_start_time

        try:
            # Check if we have valid robot pose data
            current_state = self.current_state.copy()
            if np.allclose(current_state, [0.0, 0.0, 0.0]):
                print(f"[{self.name}] No valid robot pose data yet")
                return
            
            # Debug current state every 10 calls for better visibility
            if not hasattr(self, '_debug_call_count'):
                self._debug_call_count = 0
            self._debug_call_count += 1
            
            # Check topic freshness (only outputs when there are issues)
            self._check_topic_freshness()
            
            # Ensure trajectory index is initialized before starting MPC control
            if not getattr(self, '_initial_index_set', False):
                print(f"[{self.name}] Finding initial closest trajectory index...")
                self._find_closest_trajectory_index(set_as_initial=True, wait_for_pose=False)
                print(f"[{self.name}] Initial trajectory index set: {self.trajectory_index}")
            
            # Execute MPC control if trajectory following is active
            mpc_start_time = time.time()
            if not self.picking_complete:
                success = self._follow_trajectory_with_mpc()
                # print(f"[{self.name}] MPC control step result: {'success' if success else 'failed'}")
            mpc_end_time = time.time()
            mpc_duration = mpc_end_time - mpc_start_time
                
        except Exception as e:
            print(f"[{self.name}] Error in control step: {str(e)}")
            print(f"[{self.name}] Traceback: {traceback.format_exc()}")
        finally:
            self.state_lock.release()
            
        # Calculate total control step duration
        control_step_end_time = time.time()
        control_step_duration = control_step_end_time - control_step_start_time
        
        # Update timing tracking
        self._last_control_step_time = control_step_start_time
        
        # Print timing diagnostics when control step duration > target dt or periodically
        if (control_step_duration > 0.6 or  # Alert if > 120% of target dt (0.5s)
            time_since_last_control > 1.0 or  # Alert if gap > 2x target dt
            self._control_step_call_count % 20 == 0):  # Periodic status
            
            print(f"[{self.name}] 🕒 Control Step Timing (Call #{self._control_step_call_count}):")
            print(f"   ⏰ Gap since last: {time_since_last_control:.3f}s (target: 0.5s)")
            print(f"   🔒 Lock wait: {lock_wait_duration:.3f}s")
            print(f"   🧮 MPC exec: {mpc_duration:.3f}s")
            print(f"   📊 Total duration: {control_step_duration:.3f}s")
            if time_since_last_control > 1.0:
                print(f"   ⚠️ WARNING: Control gap {time_since_last_control:.3f}s > 1.0s (should be ~0.5s)")
            if control_step_duration > 0.6:
                print(f"   ⚠️ WARNING: Control step {control_step_duration:.3f}s > 0.6s (should be <0.5s)")
        
        
    def setup(self, **kwargs):
        """Setup ROS connections and load trajectory data"""
        
        if 'node' in kwargs:
            self.node = kwargs['node']
            
            # 🔧 CRITICAL FIX: Use shared callback groups to prevent proliferation
            if hasattr(self.node, 'shared_callback_manager'):
                self.pose_callback_group = self.node.shared_callback_manager.get_group('sensor')
                self.control_callback_group = self.node.shared_callback_manager.get_group('control')
            elif hasattr(self.node, 'robot_dedicated_callback_group'):
                self.pose_callback_group = self.node.robot_dedicated_callback_group
                self.control_callback_group = self.node.robot_dedicated_callback_group
            else:
                return False
                
            # Get robot namespace and case from ROS parameters
            try:
                self.robot_namespace = self.node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "robot0"
            
            try:
                self.case = self.node.get_parameter('case').get_parameter_value().string_value
            except:
                self.case = "simple_maze"
            
            # Setup ROS publishers and subscribers for MPC control
            # Note: Subscriptions will be created in initialise(), not here
            # self._setup_ros_connections()
            
            # Load trajectory data (with replanned trajectory support)
            try:
                if not self._load_trajectory_data():
                    error_msg = f"Failed to load trajectory data for {self.robot_namespace}"
                    # Report the failure to blackboard
                    try:
                        report_node_failure(self.name, error_msg, self.robot_namespace)
                    except Exception as blackboard_err:
                        pass
                    return False
                else:
                    # Calculate closest trajectory index after trajectory is loaded
                    self._find_closest_trajectory_index(set_as_initial=True, wait_for_pose=True)
                    
            except Exception as e:
                error_msg = f"Exception during trajectory loading: {str(e)}"
                try:
                    report_node_failure(self.name, error_msg, self.robot_namespace)
                except Exception as blackboard_err:
                    pass
                return False
            
            # Initialize MPC controller
            try:
                self._setup_simple_mpc()
            except Exception as e:
                error_msg = f"Failed to initialize MPC controller: {str(e)}"
                try:
                    report_node_failure(self.name, error_msg, self.robot_namespace)
                except Exception as blackboard_err:
                    pass
                return False
            
            return True
        
        error_msg = "No ROS node provided in setup"
        try:
            report_node_failure(self.name, error_msg, self.robot_namespace)
        except:
            pass
        return False
    def setup_robot_subscription(self):
        """Set up robot pose subscription - consistent with other behaviors"""
        if self.node is None:
            return False
            
        try:
            # Clean up existing subscription if it exists
            if self.robot_pose_sub is not None:
                self.node.destroy_subscription(self.robot_pose_sub)
                self.robot_pose_sub = None
            
            # Create new subscription with shared callback group
            self.robot_pose_sub = self.node.create_subscription(
                Odometry,
                f'/{self.robot_namespace}/odom_map',
                self._robot_pose_callback,
                10,
                callback_group=self.pose_callback_group  # Use shared callback group
            )
            
            return True
        except Exception as e:
            return False

    def setup_relay_subscription(self):
        """Set up relay point pose subscription"""
        if self.node is None:
            return False
            
        try:
            # If already have relay point data, no need to resubscribe
            if self.relay_pose is not None:
                return True
            
            # Clean up existing subscription if it exists
            if self.relay_pose_sub is not None:
                self.node.destroy_subscription(self.relay_pose_sub)
                self.relay_pose_sub = None
            
            # Subscribe to relay point pose (one-time static data reading)
            self.relay_pose_sub = self.node.create_subscription(
                PoseStamped, 
                f'/Relaypoint{self.number}/pose',
                self._relay_pose_callback, 
                10,
                callback_group=self.pose_callback_group  # Use the already defined pose_callback_group
            )
            
            return True
            
        except Exception as e:
            return False

    def setup_publishers(self):
        """Set up publishers - these are created once and reused"""
        try:
            # Command velocity publisher (only create if not exists)
            if self.cmd_vel_pub is None:
                self.cmd_vel_pub = self.node.create_publisher(
                    Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            
            # Estimated time publisher (only create if not exists)
            if self.estimated_time_pub is None:
                self.estimated_time_pub = self.node.create_publisher(
                    Float64, f'/{self.robot_namespace}/pushing_estimated_time', 10)
            
            # Publish the estimated time immediately upon setup
            self._publish_estimated_time()
                
            print(f"[{self.name}] Publishers established, estimated time published: {self.estimated_time}s")
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error setting up publishers: {e}")
            return False
    
    def _relay_pose_callback(self, msg):
        """Update relay point pose - optimized for non-blocking and static data reading"""
        # Fast update of pose data (minimize lock holding time)
        try:
            with self.state_lock:
                self.relay_pose = np.array([
                    msg.pose.position.x,
                    msg.pose.position.y
                ])
            
            # Static data read, destroy subscription
            if self.relay_pose_sub is not None:
                try:
                    self.node.destroy_subscription(self.relay_pose_sub)
                    self.relay_pose_sub = None
                except Exception as destroy_e:
                    pass
                    
        except Exception as e:
            pass
    
    def _robot_pose_callback(self, msg):
        """Update robot state from odometry message - optimized for non-blocking with freshness tracking"""
        try:
            # Track callback timing for freshness checking
            current_time = time.time()
            self._last_robot_callback_time = current_time
            
            # Initialize debug counter if not exists
            if not hasattr(self, '_pose_callback_count'):
                self._pose_callback_count = 0
            self._pose_callback_count += 1
            
            # Update robot callback count for freshness checking
            if not hasattr(self, '_robot_callback_count'):
                self._robot_callback_count = 0
            self._robot_callback_count += 1
            
            # Use non-blocking lock acquisition to avoid blocking callbacks
            if self.state_lock.acquire(blocking=False):
                try:
                    # Position
                    old_state = self.current_state.copy()
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
                    
                    # # Debug output every 50 callbacks or if state significantly changed
                    # if (self._pose_callback_count % 50 == 0 or 
                    #     np.linalg.norm(self.current_state - old_state) > 0.1):
                    #     print(f"[{self.name}] POSE CALLBACK #{self._pose_callback_count}: "
                    #           f"pos=({self.current_state[0]:.3f}, {self.current_state[1]:.3f}, {self.current_state[2]:.3f})")
                              
                finally:
                    self.state_lock.release()
            else:
                # If lock can't be acquired, skip this update to avoid blocking
                if self._pose_callback_count % 100 == 0:
                    print(f"[{self.name}] POSE CALLBACK #{self._pose_callback_count}: SKIPPED (lock busy)")
                    
        except Exception as e:
            print(f"[{self.name}] ERROR in robot_pose_callback: {e}")
            import traceback
            traceback.print_exc()
    
    def _publish_estimated_time(self):
        """Publish estimated time for pushing operation"""
        if self.estimated_time_pub is not None:
            msg = Float64()
            msg.data = float(self.estimated_time)
            self.estimated_time_pub.publish(msg)
            print(f"[{self.name}] Published pushing estimated time: {self.estimated_time}s to /{self.robot_namespace}/pushing_estimated_time")
    
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
            
            # Add interpolation points for smoother approach (3-state only)
            num_interp_points = 5
            interp_points = []
            
            for i in range(num_interp_points):
                alpha = i / (num_interp_points - 1)  # Interpolation factor (0 to 1)
                interp_x = self.target_pose[0] * alpha + self.goal[0] * (1 - alpha)
                interp_y = self.target_pose[1] * alpha + self.goal[1] * (1 - alpha)
                interp_theta = self.target_pose[2]  # Keep orientation constant
                
                # Add trajectory point with 3 elements (x, y, theta) - velocities not needed
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
        try:
            # Create the full MPC controller and pass the dt parameter from PickObject
            self.mpc = MobileRobotMPC(dt=self.dt)
            
            print(f"[{self.name}] MPC initialized with dt={self.dt}s (control frequency: {1/self.dt:.1f}Hz)")
            print(f"[{self.name}] MPC tracking 3 states (x, y, theta) with 5-state dynamics model")
            
            self.prediction_horizon = self.mpc.N
            
            # Pass target pose to MPC for precision positioning
            if hasattr(self, 'target_pose'):
                self.mpc.set_target_pose(self.target_pose)
            
            # Initialize control sequence management for N_c control horizon
            self.control_sequence = None
            self.control_step_index = 0
            
        except Exception as e:
            print(f"[{self.name}] MPC setup failed: {e}")
            import traceback
            print(f"[{self.name}] MPC setup exception details: {traceback.format_exc()}")
            raise e
    
    def _find_closest_trajectory_index(self, set_as_initial=False, wait_for_pose=False):
        """Find the closest trajectory point to robot's current position - unified method for setup and runtime
        
        Args:
            set_as_initial (bool): If True, set this as the initial trajectory index and mark as set
            wait_for_pose (bool): If True, wait briefly for robot pose before calculating
        """
        try:
            # Wait for robot pose if requested (typically during setup)
            if wait_for_pose:
                self._wait_for_initial_pose(timeout=2.0)
            
            # Get current robot position
            with self.state_lock:
                current_state = self.current_state.copy()
            
            # If no valid pose received, use a default position near trajectory start
            if np.allclose(current_state, [0.0, 0.0, 0.0]):
                print(f"[{self.name}] No robot pose available")
            if not self.trajectory_data or len(self.trajectory_data) == 0:
                print(f"[{self.name}] No trajectory data available for closest index calculation")
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
            
            # Set as initial index if requested
            if set_as_initial:
                self._safe_update_trajectory_index(closest_idx, "initial_closest_point_search")
                self.closest_idx = closest_idx
                self._initial_index_set = True
            
            return closest_idx
            
        except Exception as e:
            print(f"[{self.name}] Error finding closest trajectory index: {e}")
            if set_as_initial:
                self._safe_update_trajectory_index(0, "error_fallback")
                self.closest_idx = 0
                self._initial_index_set = True
            return 0

    def _wait_for_initial_pose(self, timeout=5.0):
        """Non-blocking check for initial robot pose"""
        with self.state_lock:
            # Check if we have received a valid pose (not all zeros)
            if not np.allclose(self.current_state, [0.0, 0.0, 0.0]):
                print(f"[{self.name}] Initial robot pose received: ({self.current_state[0]:.3f}, {self.current_state[1]:.3f}, {self.current_state[2]:.3f})")
                return True
        
        print(f"[{self.name}] Warning: No initial robot pose available yet, will retry in next cycle")
        return False

    def initialise(self):
        """Initialize the picking behavior - complete setup equivalent to setup() method"""
        self.start_time = time.time()
        self.picking_active = True
        self.picking_complete = False
        self.feedback_message = f"[{self.robot_namespace}] Starting pick operation..."
        
        # Initialize debug counters
        self._debug_call_count = 0
        self._pose_callback_count = 0
        
        # RESET: Clear all state for fresh initialization
        self.trajectory_data = None
        self.goal = None
        self.target_pose = np.zeros(3)
        self._initial_index_set = False
        self._safe_update_trajectory_index(0, "initialization_reset")
        self.closest_idx = 0
        
        # CRITICAL: Reset current_state to ensure fresh pose acquisition on second launch
        self.current_state = np.zeros(3)
        
        # Reset pose callback related variables for fresh start
        self._last_robot_callback_time = None
        self._robot_callback_count = 0
        self._freshness_check_count = 0

         # 2: Setup ROS connections (REQUIRES callback groups from STEP 1)
        # The setup_robot_subscription() and setup_relay_subscription() methods 
        # use self.pose_callback_group which was set up in STEP 1
        print(f"[{self.name}] Setting up ROS subscriptions...")
        robot_sub_ok = self.setup_robot_subscription()
        relay_sub_ok = self.setup_relay_subscription()
        print(f"[{self.name}] Setting up publishers...")
        self.setup_publishers()
          # Wait for subscriptions to establish and data to be received
        print(f"[{self.name}] Waiting for subscriptions to establish...")
        time.sleep(0.8)  # Wait for topic connectivity
        self.check_topic_connectivity()

        # Check for critical data availability
        print(f"[{self.name}] Checking for critical data availability...")
        robot_has_data = (self.current_state is not None and not np.allclose(self.current_state, [0.0, 0.0, 0.0]))
        
        if robot_has_data:
            print(f"[{self.name}] ✅ Critical data is ready.")
        else:
            print(f"[{self.name}] ⚠️ Waiting for robot pose data - control will start when data arrives.")

        print(f"[{self.name}] Subscription status: robot={robot_sub_ok}, relay={relay_sub_ok}")

        #3: Load trajectory data
        print(f"[{self.name}] Loading trajectory data...")
        try:
            if not self._load_trajectory_data():
                error_msg = f"Failed to load trajectory data for {self.robot_namespace}"
                print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
                try:
                    report_node_failure(self.name, error_msg, self.robot_namespace)
                except Exception as blackboard_err:
                    print(f"[{self.name}] Cannot report failure to blackboard: {blackboard_err}")
                self.picking_active = False
                return
            else:
                print(f"[{self.name}] Trajectory data loaded successfully!")
                print(f"[{self.name}] Target pose: ({self.target_pose[0]:.3f}, {self.target_pose[1]:.3f}, {self.target_pose[2]:.3f})")
                print(f"[{self.name}] Goal position: ({self.goal[0]:.3f}, {self.goal[1]:.3f}, {self.goal[2]:.3f})")
                    
        except Exception as e:
            error_msg = f"Exception during trajectory loading: {str(e)}"
            print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
            try:
                report_node_failure(self.name, error_msg, self.robot_namespace)
            except Exception as blackboard_err:
                print(f"[{self.name}] Cannot report failure to blackboard: {blackboard_err}")
            self.picking_active = False
            return
            
        # Compute initial closest trajectory index AFTER trajectory data is loaded
        if robot_has_data:
            print(f"[{self.name}] Computing initial closest trajectory index...")
            self._find_closest_trajectory_index(set_as_initial=True, wait_for_pose=False)

        #4: Initialize MPC controller
        print(f"[{self.name}] Initializing MPC controller...")
        try:
            self._setup_simple_mpc()
            print(f"[{self.name}] MPC controller initialized successfully!")
        except Exception as e:
            error_msg = f"Failed to initialize MPC controller: {str(e)}"
            print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
            try:
                report_node_failure(self.name, error_msg, self.robot_namespace)
            except Exception as blackboard_err:
                print(f"[{self.name}] Cannot report failure to blackboard: {blackboard_err}")
            self.picking_active = False
            return

        # 5. STEP 5: Start control timer
        print(f"[{self.name}] Starting control thread with dt={self.dt}s...")
        self.start_control_thread()
        
        # Verify that the control thread was successfully started
        if self.control_thread_active and self.control_timer is not None:
            print(f"[{self.name}] ✅ Control timer successfully created and control thread is active")
        else:
            print(f"[{self.name}] ⚠️ Control setup incomplete: thread_active={self.control_thread_active}, timer_exists={self.control_timer is not None}")
        
        print(f"[{self.name}] PickObject initialization complete for robot: {self.robot_namespace}")
        print(f"[INITIALIZE] ✓ PickObject successfully initialized: {self.name}")
    
    def check_topic_connectivity(self):
        """Verify topic data flow connectivity (non-blocking)."""
        topics = [
            f'/{self.robot_namespace}/odom_map',
            f'/Relaypoint{self.number}/pose'
        ]
        
        print(f"[{self.name}] 📊 Topic Connectivity Check (Node: {self.node.get_name() if self.node else 'None'})")
        
        for topic in topics:
            try:
                publishers = self.node.count_publishers(topic)
                subscribers = self.node.count_subscribers(topic)
                if publishers == 0:
                    print(f"⚠️ Topic {topic} has no publishers!")
                else:
                    if "Relaypoint" in topic:
                        if self.relay_pose is not None:
                            print(f"✅ Relay data acquired {topic} (Pubs: {publishers}, Subs: {subscribers}) [Static]")
                        else:
                            print(f"✅ Connected to {topic} (Pubs: {publishers}, Subs: {subscribers}) [Waiting for static data]")
                    else:
                        print(f"✅ Connected to {topic} (Pubs: {publishers}, Subs: {subscribers})")
            except Exception as e:
                print(f"❌ Topic check failed for {topic}: {str(e)}")
                
        # Additional debug: Check if callbacks are actually being triggered
        print(f"[{self.name}] 📊 Callback Status Check:")
        print(f"   Robot State: {self.current_state is not None and not np.allclose(self.current_state, [0.0, 0.0, 0.0])} (Callbacks: {getattr(self, '_robot_callback_count', 0)})")
        print(f"   Relay Pose: {self.relay_pose is not None}")
        
        # Check subscription objects
        print(f"[{self.name}] 📊 Subscription Object Status:")
        print(f"   robot_pose_sub: {self.robot_pose_sub is not None}")
        print(f"   relay_pose_sub: {self.relay_pose_sub is not None}")
        
        # Force a topic list check to see what topics actually exist
        try:
            topic_names_and_types = self.node.get_topic_names_and_types()
            available_topics = [name for name, _ in topic_names_and_types]
            print(f"[{self.name}] 📊 System has {len(available_topics)} available topics.")
            
            for topic in topics:
                if topic in available_topics:
                    print(f"   ✅ Topic exists: {topic}")
                else:
                    print(f"   ❌ Topic does NOT exist: {topic}")
        except Exception as e:
            print(f"[{self.name}] ❌ Could not check system topic list: {e}")

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
        
        # Periodically publish estimated time
        current_time = time.time()
        if current_time - self.last_estimated_time_publish >= self.estimated_time_publish_interval:
            self._publish_estimated_time()
            self.last_estimated_time_publish = current_time
        
        # Check if robot is close to target
        # Use non-blocking lock acquisition to avoid blocking callbacks
        if self.state_lock.acquire(blocking=False):
            try:
                current_state = self.current_state.copy()
            finally:
                self.state_lock.release()
        else:
            # If lock can't be acquired, use last known state
            current_state = getattr(self, '_last_known_state', np.zeros(3))
            
        # Store last known state for fallback
        self._last_known_state = current_state
            
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
        if elapsed >= self.timeout:
            report_node_failure(self.name, f"PickObject timeout after {self.timeout}s - failed to reach target", self.robot_namespace)
            print(f"[{self.name}] Pick operation timed out after {self.timeout}s")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING


    
    def _follow_trajectory_with_mpc(self):
        """Follow trajectory using MPC controller with control sequence management (similar to PushObject)"""
        import time  # Import time for timing diagnostics
        
        try:
            # Start timing for entire function
            function_start_time = time.time()
            
            # Ensure initial closest trajectory index is set before first controller calculation
            if not getattr(self, '_initial_index_set', False):
                print(f"[{self.name}] Initial trajectory index not set, finding closest point before first controller calculation...")
                self._find_closest_trajectory_index(set_as_initial=True, wait_for_pose=False)
            
            # Check if we have a valid stored control sequence we can use
            if self.control_sequence is not None:
                # Use the stored control sequence if available
                if self._apply_stored_control():
                    self._advance_control_step()
                    function_end_time = time.time()
                    function_duration = function_end_time - function_start_time
                    if function_duration > 0.1:  # Only print if > 100ms
                        print(f"[{self.name}] ⏱️ _follow_trajectory_with_mpc (stored): {function_duration:.3f}s")
                    return True
           
            # If no valid control sequence or need replanning, run MPC
            # NOTE: State lock is already held by control_step(), so we don't need to acquire it again
            current_state = self.current_state.copy()
            
            # Timing: Closest point search
            search_start_time = time.time()
            
            # Always find the closest trajectory point to ensure accurate reference
            curr_pos = np.array([current_state[0], current_state[1]])
            min_dist = float('inf')
            closest_idx = 0
            
            # Search in local window around current index for efficiency
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
            
            search_end_time = time.time()
            search_duration = search_end_time - search_start_time
            
            # Update trajectory index to closest point with timing logging
            self._safe_update_trajectory_index(closest_idx, "MPC_trajectory_following")
            
            # Timing: Reference trajectory preparation
            ref_prep_start_time = time.time()
            
            # Prepare reference trajectory for MPC horizon - only position and orientation
            ref_traj = np.zeros((5, self.prediction_horizon + 1))  # 5 states: x, y, theta, v, omega
            for i in range(self.prediction_horizon + 1):
                idx = min(closest_idx + i, len(self.trajectory_data) - 1)
                ref_traj[:3, i] = self.trajectory_data[idx][:3]  # x, y, theta only
                # Leave velocity states (ref_traj[3:5, i]) as zeros since we don't track them in cost
            
            ref_prep_end_time = time.time()
            ref_prep_duration = ref_prep_end_time - ref_prep_start_time
            
            # Set reference trajectory in MPC
            self.mpc.set_reference_trajectory(ref_traj)
            
            # Timing: MPC optimization (this is likely the bottleneck)
            mpc_start_time = time.time()
            
            # Run MPC optimization
            control_sequence = self.mpc.update(current_state)
            
            mpc_end_time = time.time()
            mpc_duration = mpc_end_time - mpc_start_time
            
            if control_sequence is None or np.any(np.isnan(control_sequence)):
                print(f"[{self.name}] ERROR: MPC optimization failed")
                return False
                
            # Store new control sequence
            self.control_sequence = control_sequence
            self.control_step_index = 0
            
            # Timing: Command preparation and publishing
            cmd_start_time = time.time()
            
            # Apply first control input
            v_cmd = float(control_sequence[0, 0])  # Linear velocity
            omega_cmd = float(control_sequence[1, 0])  # Angular velocity
            
            # Create and publish velocity command
            cmd_vel = Twist()
            cmd_vel.linear.x = v_cmd
            cmd_vel.angular.z = omega_cmd
            
            self.cmd_vel_pub.publish(cmd_vel)
            
            cmd_end_time = time.time()
            cmd_duration = cmd_end_time - cmd_start_time
            
            # Calculate total function duration
            function_end_time = time.time()
            function_duration = function_end_time - function_start_time
            
            # Only print commands and timing periodically to reduce output
            if not hasattr(self, '_cmd_print_count'):
                self._cmd_print_count = 0
            self._cmd_print_count += 1
            
            # Calculate distance to target for printing
            dist_to_target = np.sqrt(
                (current_state[0] - self.target_pose[0])**2 + 
                (current_state[1] - self.target_pose[1])**2
            )
            
            # Print timing diagnostics when total time > 0.1s or every 10 calls
            if function_duration > 0.1 or self._cmd_print_count % 10 == 0:
                print(f"[{self.name}] ⏱️ MPC Timing Breakdown:")
                print(f"   🔍 Search: {search_duration:.3f}s")
                print(f"   📝 Ref prep: {ref_prep_duration:.3f}s") 
                print(f"   🧮 MPC opt: {mpc_duration:.3f}s ← BOTTLENECK")
                print(f"   📤 Cmd pub: {cmd_duration:.3f}s")
                print(f"   ⏰ TOTAL: {function_duration:.3f}s (target: <0.5s for 2Hz)")
                print(f"   🎯 CMD: v={v_cmd:.3f} m/s, ω={omega_cmd:.3f} rad/s | dist={dist_to_target:.3f}m | idx={self.trajectory_index}")
            elif self._cmd_print_count % 10 == 0:
                print(f"[{self.name}] CMD: v={v_cmd:.3f} m/s, ω={omega_cmd:.3f} rad/s | "
                      f"dist={dist_to_target:.3f}m | idx={self.trajectory_index} | t={function_duration:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error in trajectory following: {str(e)}")
            print(f"[{self.name}] Traceback: {traceback.format_exc()}")
            return False
    
    def terminate(self, new_status):
        """Cleanup when behavior is terminated"""
        try:
            print(f"[{self.name}] Terminating picking behavior with status: {new_status}")
            
            # Stop any active movement
            if self.cmd_vel_pub is not None:
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                stop_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(stop_msg)
                print(f"[{self.name}] Published zero velocity command")
            
            # Stop control thread/timer if active
            if self.control_thread_active or self.control_timer is not None:
                print(f"[{self.name}] Stopping control thread...")
                self.stop_control_thread()
            
            # Clean up subscribers
            if self.robot_pose_sub is not None:
                self.node.destroy_subscription(self.robot_pose_sub)
                self.robot_pose_sub = None
            
            if self.relay_pose_sub is not None:
                self.node.destroy_subscription(self.relay_pose_sub)
                self.relay_pose_sub = None
            
            # Reset control and state variables
            self.control_thread_active = False
            self.picking_active = False
            self.picking_complete = False
            self._initial_index_set = False
            
            print(f"[{self.name}] Cleanup complete")
            
        except Exception as e:
            print(f"[{self.name}] Error during termination: {str(e)}")
            print(f"[{self.name}] Termination error details: {traceback.format_exc()}")
        finally:
            # Always ensure these flags are reset
            self.control_thread_active = False
            self.picking_active = False

    def _apply_stored_control(self):
        """Apply a stored control input from the control sequence"""
        try:
            if self.control_sequence is None or self.control_step_index >= self.mpc.N_c:
                return False
                
            # Get control input for current step
            v_cmd = float(self.control_sequence[0, self.control_step_index])
            omega_cmd = float(self.control_sequence[1, self.control_step_index])
            
            # Create and publish velocity command
            cmd_vel = Twist()
            cmd_vel.linear.x = v_cmd
            cmd_vel.angular.z = omega_cmd
            
            # Only print commands periodically
            if not hasattr(self, '_stored_cmd_print_count'):
                self._stored_cmd_print_count = 0
            self._stored_cmd_print_count += 1
            if self._stored_cmd_print_count % 10 == 0:
                print(f"[{self.name}] Stored CMD: v={v_cmd:.3f} m/s, ω={omega_cmd:.3f} rad/s")
            
            self.cmd_vel_pub.publish(cmd_vel)
            
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error applying stored control: {str(e)}")
            return False
            
    def _advance_control_step(self):
        """Advance to the next control step in the sequence"""
        self.control_step_index += 1
        if self.control_step_index >= self.mpc.N_c:
            self.control_sequence = None
            self.control_step_index = 0
    
    def _check_topic_freshness(self):
        """Check if topic data is fresh (non-blocking)"""
        try:
            # Only check every 50 calls to avoid spam
            if not hasattr(self, '_freshness_check_count'):
                self._freshness_check_count = 0
            self._freshness_check_count += 1
            
            if self._freshness_check_count % 50 != 0:
                return
                
            current_time = time.time()
            
            # Check robot pose freshness
            if self._last_robot_callback_time is not None:
                time_since_last_robot = current_time - self._last_robot_callback_time
                if time_since_last_robot > 0.5:  # Over 500ms old
                    print(f"[{self.name}] ERROR: Robot pose data stale ({time_since_last_robot:.2f}s old)")
            else:
                print(f"[{self.name}] ERROR: No robot pose data received")
                
            # No need to check relay pose freshness as it's static data
            
        except Exception as e:
            # Don't let freshness checking crash the control loop
            if self._freshness_check_count % 100 == 0:  # Limit error reporting
                print(f"[{self.name}] Error checking topic freshness: {str(e)}")
    
    def _safe_update_trajectory_index(self, new_index, context="unknown"):
        """Safely update trajectory index with timing logging (similar to PushObject)"""
        import time  # Import time for timing calculations
        
        old_index = self.trajectory_index
        
        # Validate new index bounds
        if new_index < 0:
            new_index = 0
        elif new_index >= len(self.trajectory_data) if self.trajectory_data else 0:
            new_index = len(self.trajectory_data) - 1 if self.trajectory_data else 0
        
        # Update trajectory index
        self.trajectory_index = new_index
        
        # Log all trajectory index changes with timing information
        if new_index != old_index:
            current_time = time.time()
            if not hasattr(self, '_last_trajectory_update_time'):
                self._last_trajectory_update_time = current_time
                self._trajectory_update_count = 0
            
            time_since_last = current_time - self._last_trajectory_update_time
            self._trajectory_update_count += 1
            
            # Log trajectory index changes with timing information
            if time_since_last < self.dt:  # If updates are happening faster than 5Hz
                print(f"[{self.name}] ⚠️ FAST trajectory index update: {old_index} -> {new_index} in '{context}' (Δt={time_since_last:.3f}s)")
            else:
                print(f"[{self.name}] ✅ Normal trajectory index update: {old_index} -> {new_index} in '{context}' (Δt={time_since_last:.3f}s)")
            
            self._last_trajectory_update_time = current_time
            
            return True  # Index updated
        else:
            return False  # Index unchanged