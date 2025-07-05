#!/usr/bin/env python3
import py_trees
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import time
import json
import numpy as np
import casadi as ca
import os
import math
import threading
from tf_transformations import euler_from_quaternion
import re
import copy
import tf_transformations as tf
from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Int32, Float32, Float64
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
        self.Q = np.diag([50.0, 50.0, 1.0, 1.0, 1.0])   # State weights (x, y, theta, v, omega)
        self.R = np.diag([1.0, 1.0])                     # Control input weights (v, omega)
        self.F = np.diag([100.0, 100.0, 2.0, 2.0, 2.0]) # Terminal cost weights
        
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
    
    def __init__(self, name, robot_namespace="turtlebot0", timeout=100.0, estimated_time=55.0):
        super().__init__(name)
        self.start_time = None
        self.picking_active = False
        self.node = None
        self.number=extract_namespace_number(robot_namespace)
        self.picking_complete = False
        self.robot_namespace = robot_namespace  # Use provided robot_namespace
        self.case = "simple_maze"  # Default case
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
        self.dt = 0.1  # 10Hz control frequency
        
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
        """Start the control timer for 10Hz control loop using ROS timer"""
        if self.control_thread_active or self.control_timer is not None:
            return  # Already running
            
        self.control_thread_active = True
        
        # Use the shared callback group from the node to create a timer
        callback_group = None
        if hasattr(self.node, 'shared_callback_manager'):
            callback_group = self.node.shared_callback_manager.get_group('control')
            print(f"[{self.name}] Using shared control callback group for timer: {id(callback_group)}")
        elif hasattr(self.node, 'robot_dedicated_callback_group'):
            callback_group = self.node.robot_dedicated_callback_group
            print(f"[{self.name}] Using robot dedicated callback group for timer: {id(callback_group)}")
        
        # Create a ROS timer with the appropriate callback group
        self.control_timer = self.node.create_timer(
            self.dt,  # 10Hz frequency = 0.1s period
            self.control_step,  # Use the existing control_step as callback
            callback_group=callback_group
        )
        print(f"[{self.name}] ROS control timer started for Pickup behavior at {1.0/self.dt}Hz")
        
    def stop_control_thread(self):
        """Stop the control timer"""
        self.control_thread_active = False
        if self.control_timer is not None:
            self.node.destroy_timer(self.control_timer)
            self.control_timer = None
            print(f"[{self.name}] ROS control timer stopped")
    
    # This method is no longer needed - replaced by direct timer callback to control_step
    # def control_loop_thread(self):
    #    """Dedicated control thread running at 10Hz"""
    #    import time
    #    
    #    while self.control_thread_active and (not self.node or rclpy.ok()):
    #        try:
    #            self.control_step()
    #            time.sleep(0.1)  # 10Hz control frequency
    #        except Exception as e:
    #            print(f"[{self.name}] Error in control thread: {e}")
                    
    def control_step(self):
        """Single control step - now called directly by ROS timer"""
        if not self.control_thread_active or not self.picking_active:
            return
            
        # Debug: Check critical prerequisites
        if not hasattr(self, 'mpc') or self.mpc is None:
            print(f"[{self.name}] ERROR: MPC controller not initialized!")
            return
            
        if not hasattr(self, 'trajectory_data') or self.trajectory_data is None:
            print(f"[{self.name}] ERROR: No trajectory data loaded!")
            return
            
        if not hasattr(self, 'cmd_vel_pub') or self.cmd_vel_pub is None:
            print(f"[{self.name}] ERROR: cmd_vel_pub not initialized!")
            return
            
        # Try to acquire state lock non-blocking
        state_acquired = self.state_lock.acquire(blocking=False)
        if not state_acquired:
            # Use last known state if lock cannot be acquired
            print(f"[{self.name}] Using last known state for control step - lock is BUSY/HELD by another process")
            return

            
        try:
            # Check if we have valid robot pose data
            current_state = self.current_state.copy()
            if np.allclose(current_state, [0.0, 0.0, 0.0]):
                # Also check if callback is receiving data
                callback_count = getattr(self, '_pose_callback_count', 0)
                print(f"[{self.name}] WARNING: No valid robot pose data - current_state is all zeros! "
                      f"Callback count: {callback_count}, Subscription active: {self.robot_pose_sub is not None}")
                return
            
            # Debug current state every 50 calls
            if not hasattr(self, '_debug_call_count'):
                self._debug_call_count = 0
            self._debug_call_count += 1
            
            if self._debug_call_count % 50 == 0:
                print(f"[{self.name}] DEBUG Control Step #{self._debug_call_count}: "
                      f"pos=({current_state[0]:.3f}, {current_state[1]:.3f}, {current_state[2]:.3f}), "
                      f"trajectory_idx={getattr(self, 'trajectory_index', 'N/A')}/{len(self.trajectory_data) if self.trajectory_data else 'N/A'}")
            
            # Check topic freshness (only outputs when there are issues)
            self._check_topic_freshness()
            
            # Ensure trajectory index is initialized before starting MPC control
            if not getattr(self, '_initial_index_set', False):
                print(f"[{self.name}] Trajectory index not initialized in control step, finding closest point...")
                self._find_closest_trajectory_index(set_as_initial=True, wait_for_pose=False)
            
            # Execute MPC control if trajectory following is active
            if not self.picking_complete:
                success = self._follow_trajectory_with_mpc()
                if not success:
                    print(f"[{self.name}] MPC trajectory following returned False!")
            else:
                print(f"[{self.name}] Picking complete - stopping control")
                
        except Exception as e:
            print(f"[{self.name}] Error in control step: {e}")
            import traceback
            print(f"[{self.name}] Control step traceback: {traceback.format_exc()}")
        finally:
            self.state_lock.release()
        
    def setup(self, **kwargs):
        """Setup ROS connections and load trajectory data"""
        
        if 'node' in kwargs:
            self.node = kwargs['node']
            
            # 🔧 CRITICAL FIX: Use shared callback groups to prevent proliferation
            if hasattr(self.node, 'shared_callback_manager'):
                self.pose_callback_group = self.node.shared_callback_manager.get_group('sensor')
                self.control_callback_group = self.node.shared_callback_manager.get_group('control')
                print(f"[{self.name}] ✅ Using shared callback group manager: sensor={id(self.pose_callback_group)}, control={id(self.control_callback_group)}")
            elif hasattr(self.node, 'robot_dedicated_callback_group'):
                self.pose_callback_group = self.node.robot_dedicated_callback_group
                self.control_callback_group = self.node.robot_dedicated_callback_group
                print(f"[{self.name}] ✅ Using robot dedicated callback group: {id(self.pose_callback_group)}")
            else:
                print(f"[{self.name}] ❌ Error: No shared_callback_manager found, cannot use shared callback groups")
                return False
                
            # Get robot namespace and case from ROS parameters
            try:
                self.robot_namespace = self.node.get_parameter('robot_namespace').get_parameter_value().string_value
                print(f"[{self.name}] 获取到机器人命名空间: {self.robot_namespace}")
            except:
                self.robot_namespace = "turtlebot0"
                print(f"[{self.name}] 使用默认机器人命名空间: {self.robot_namespace}")
            
            try:
                self.case = self.node.get_parameter('case').get_parameter_value().string_value
                print(f"[{self.name}] 获取到案例: {self.case}")
            except:
                self.case = "simple_maze"
                print(f"[{self.name}] 使用默认案例: {self.case}")
            
            # Setup ROS publishers and subscribers for MPC control
            # Note: Subscriptions will be created in initialise(), not here
            # self._setup_ros_connections()
            
            # Load trajectory data (with replanned trajectory support)
            print(f"[{self.name}] 开始加载轨迹数据...")
            try:
                if not self._load_trajectory_data():
                    error_msg = f"Failed to load trajectory data for {self.robot_namespace}"
                    print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
                    print(f"[{self.name}] Expected file: /root/workspace/data/{self.case}/tb{self.number}_Trajectory_replanned.json")
                    # Report the failure to blackboard
                    try:
                        report_node_failure(self.name, error_msg, self.robot_namespace)
                    except Exception as blackboard_err:
                        print(f"[{self.name}] 无法报告失败到黑板: {blackboard_err}")
                    return False
                else:
                    print(f"[{self.name}] 轨迹数据加载成功!")
                    
                    # Calculate closest trajectory index after trajectory is loaded
                    print(f"[{self.name}] 计算初始最近轨迹索引...")
                    self._find_closest_trajectory_index(set_as_initial=True, wait_for_pose=True)
                    
            except Exception as e:
                error_msg = f"Exception during trajectory loading: {str(e)}"
                print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
                print(f"[{self.name}] Traceback: {e}")
                try:
                    report_node_failure(self.name, error_msg, self.robot_namespace)
                except Exception as blackboard_err:
                    print(f"[{self.name}] 无法报告失败到黑板: {blackboard_err}")
                return False
            
            # Initialize MPC controller
            print(f"[{self.name}] 初始化 MPC 控制器...")
            try:
                self._setup_simple_mpc()
                print(f"[{self.name}] MPC 控制器初始化成功!")
            except Exception as e:
                error_msg = f"Failed to initialize MPC controller: {str(e)}"
                print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
                print(f"[{self.name}] MPC Traceback: {e}")
                try:
                    report_node_failure(self.name, error_msg, self.robot_namespace)
                except Exception as blackboard_err:
                    print(f"[{self.name}] 无法报告失败到黑板: {blackboard_err}")
                return False
            
            print(f"[{self.name}] PickObject 设置完成，机器人: {self.robot_namespace}, 案例: {self.case}")
            print(f"[SETUP] ✓ PickObject 设置成功: {self.name}")
            return True
        
        error_msg = "No ROS node provided in setup"
        print(f"[{self.name}] ERROR: {error_msg}")
        try:
            report_node_failure(self.name, error_msg, self.robot_namespace)
        except:
            print(f"[{self.name}] 无法报告失败到黑板")
        return False
    def setup_robot_subscription(self):
        """Set up robot pose subscription - consistent with other behaviors"""
        if self.node is None:
            print(f"[{self.name}] WARNING: Cannot setup robot subscription - no ROS node")
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
            
            print(f"[{self.name}] ✅ Robot pose subscription created using shared callback group")
            return True
        except Exception as e:
            print(f"[{self.name}] Error setting up robot pose subscription: {e}")
            return False

    def setup_relay_subscription(self):
        """Set up relay point pose subscription"""
        if self.node is None:
            print(f"[{self.name}] WARNING: Cannot setup relay subscription - no ROS node")
            return False
            
        try:
            # 如果已经有中继点数据，无需重新订阅
            if self.relay_pose is not None:
                print(f"[{self.name}] ✅ 中继点数据已存在，跳过订阅")
                return True
            
            # Clean up existing subscription if it exists
            if self.relay_pose_sub is not None:
                self.node.destroy_subscription(self.relay_pose_sub)
                self.relay_pose_sub = None
            
            # Subscribe to relay point pose (一次性读取静态数据)
            self.relay_pose_sub = self.node.create_subscription(
                PoseStamped, 
                f'/Relaypoint{self.number}/pose',
                self._relay_pose_callback, 
                10,
                callback_group=self.pose_callback_group  # Use the already defined pose_callback_group
            )
            
            print(f"[{self.name}] ✅ 成功订阅中继话题: /Relaypoint{self.number}/pose (中继点: {self.number}) [一次性读取]")
            return True
            
        except Exception as e:
            print(f"[{self.name}] ERROR: Failed to setup relay subscription: {e}")
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
                
            print(f"[{self.name}] 发布器已建立，估计时间已发布: {self.estimated_time}s")
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error setting up publishers: {e}")
            return False
    
    def _relay_pose_callback(self, msg):
        """Update relay point pose - optimized for non-blocking and static data reading"""
        # 快速更新姿态数据（最小化锁持有时间）
        try:
            with self.state_lock:
                self.relay_pose = np.array([
                    msg.pose.position.x,
                    msg.pose.position.y
                ])
            
            # 静态数据已读取，销毁订阅 (Static relay data read, destroy subscription)
            if self.relay_pose_sub is not None:
                try:
                    self.node.destroy_subscription(self.relay_pose_sub)
                    self.relay_pose_sub = None
                    print(f"[{self.name}] ✅ 中继点静态数据已读取，订阅已销毁: ({self.relay_pose[0]:.3f}, {self.relay_pose[1]:.3f})")
                except Exception as destroy_e:
                    print(f"[{self.name}] WARNING: Failed to destroy relay subscription: {destroy_e}")
                    
        except Exception as e:
            print(f"[{self.name}] ERROR in relay_pose_callback: {e}")
    
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
                    
                    # Debug output every 50 callbacks or if state significantly changed
                    if (self._pose_callback_count % 50 == 0 or 
                        np.linalg.norm(self.current_state - old_state) > 0.1):
                        print(f"[{self.name}] POSE CALLBACK #{self._pose_callback_count}: "
                              f"pos=({self.current_state[0]:.3f}, {self.current_state[1]:.3f}, {self.current_state[2]:.3f})")
                              
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
            print(f"[{self.name}] 发布推送估计时间: {self.estimated_time}秒 到 /{self.robot_namespace}/pushing_estimated_time")
    
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
        try:
            print(f"[{self.name}] 创建 MobileRobotMPC 实例...")
            # Create the full MPC controller from manipulation_behaviors.py
            self.mpc = MobileRobotMPC()
            print(f"[{self.name}] MobileRobotMPC 创建成功")
            
            self.prediction_horizon = self.mpc.N
            print(f"[{self.name}] 预测地平线设置为: {self.prediction_horizon}")
            
            # Pass target pose to MPC for precision positioning
            if hasattr(self, 'target_pose'):
                self.mpc.set_target_pose(self.target_pose)
                print(f"[{self.name}] 目标姿态已设置到 MPC")
            
            # Initialize control sequence management for N_c control horizon
            self.control_sequence = None
            self.control_step_index = 0
            print(f"[{self.name}] 控制序列管理已初始化")
            
        except Exception as e:
            print(f"[{self.name}] MPC 设置失败: {e}")
            import traceback
            print(f"[{self.name}] MPC 设置异常详情: {traceback.format_exc()}")
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
                print(f"[{self.name}] No robot pose available, using trajectory start as reference")
                # Use first trajectory point as reference
                if self.trajectory_data and len(self.trajectory_data) > 0:
                    current_state = np.array([
                        self.trajectory_data[0][0], 
                        self.trajectory_data[0][1], 
                        self.trajectory_data[0][2]
                    ])
                else:
                    print(f"[{self.name}] ERROR: No trajectory data available for closest index calculation")
                    return 0
            
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
            
            print(f"[{self.name}] Closest trajectory point found at index {closest_idx}/{len(self.trajectory_data)-1}")
            print(f"[{self.name}] Distance to closest point: {min_dist:.3f}m")
            print(f"[{self.name}] Robot position: ({current_state[0]:.3f}, {current_state[1]:.3f})")
            print(f"[{self.name}] Closest trajectory point: ({self.trajectory_data[closest_idx][0]:.3f}, {self.trajectory_data[closest_idx][1]:.3f})")
            
            # Set as initial index if requested
            if set_as_initial:
                self.trajectory_index = closest_idx
                self.closest_idx = closest_idx
                self._initial_index_set = True
                print(f"[{self.name}] MPC controller will start from trajectory index {closest_idx}")
            
            return closest_idx
            
        except Exception as e:
            print(f"[{self.name}] Error finding closest trajectory index: {e}")
            if set_as_initial:
                self.trajectory_index = 0
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
        print(f"[{self.name}] Resetting all state for fresh initialization...")
        self.trajectory_data = None
        self.goal = None
        self.target_pose = np.zeros(3)  # Reset target pose to zeros
        self._initial_index_set = False  # Reset trajectory index flag
        self.trajectory_index = 0
        self.closest_idx = 0
        
        # RELOAD: Load fresh trajectory data and recalculate goal position
        print(f"[{self.name}] Reloading trajectory data during initialization...")
        try:
            if not self._load_trajectory_data():
                error_msg = f"Failed to load trajectory data for {self.robot_namespace} during initialization"
                print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
                print(f"[{self.name}] Expected file: /root/workspace/data/{self.case}/tb{self.number}_Trajectory_replanned.json")
                # Report the failure to blackboard
                try:
                    report_node_failure(self.name, error_msg, self.robot_namespace)
                except Exception as blackboard_err:
                    print(f"[{self.name}] 无法报告失败到黑板: {blackboard_err}")
                self.picking_active = False
                return
            else:
                print(f"[{self.name}] 轨迹数据加载成功!")
                print(f"[{self.name}] New target pose: ({self.target_pose[0]:.3f}, {self.target_pose[1]:.3f}, {self.target_pose[2]:.3f})")
                print(f"[{self.name}] New goal position: ({self.goal[0]:.3f}, {self.goal[1]:.3f}, {self.goal[2]:.3f})")
                
                # Calculate closest trajectory index after trajectory is loaded - like in setup
                print(f"[{self.name}] 计算初始最近轨迹索引...")
                self._find_closest_trajectory_index(set_as_initial=True, wait_for_pose=True)
                    
        except Exception as e:
            error_msg = f"Exception during trajectory loading in initialization: {str(e)}"
            print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
            print(f"[{self.name}] Traceback: {e}")
            try:
                report_node_failure(self.name, error_msg, self.robot_namespace)
            except Exception as blackboard_err:
                print(f"[{self.name}] 无法报告失败到黑板: {blackboard_err}")
            self.picking_active = False
            return
        
        # 3. Initialize/Re-initialize MPC controller - like in setup
        print(f"[{self.name}] 初始化 MPC 控制器...")
        try:
            self._setup_simple_mpc()
            print(f"[{self.name}] MPC 控制器初始化成功!")
        except Exception as e:
            error_msg = f"Failed to initialize MPC controller during initialization: {str(e)}"
            print(f"[{self.name}] CRITICAL ERROR: {error_msg}")
            print(f"[{self.name}] MPC Traceback: {e}")
            try:
                report_node_failure(self.name, error_msg, self.robot_namespace)
            except Exception as blackboard_err:
                print(f"[{self.name}] 无法报告失败到黑板: {blackboard_err}")
            self.picking_active = False
            return
        
        # 4. Setup ROS connections (subscriptions and publishers)
        print(f"[{self.name}] Setting up robot subscription...")
        robot_sub_ok = self.setup_robot_subscription()
        print(f"[{self.name}] Setting up relay subscription...")
        relay_sub_ok = self.setup_relay_subscription()
        print(f"[{self.name}] Setting up publishers...")
        self.setup_publishers()
        
        # 5. Debug: Check if topic exists
        if robot_sub_ok and self.node is not None:
            try:
                topic_name = f'/{self.robot_namespace}/odom_map'
                topic_list = self.node.get_topic_names_and_types()
                topic_exists = any(topic_name in topic for topic, _ in topic_list)
                print(f"[{self.name}] Topic {topic_name} exists: {topic_exists}")
                if not topic_exists:
                    print(f"[{self.name}] Available topics: {[topic for topic, _ in topic_list if 'odom' in topic]}")
            except Exception as e:
                print(f"[{self.name}] Could not check topic list: {e}")
        
        # 6. Start control timer
        self.start_control_thread()
        
        print(f"[{self.name}] PickObject 完整初始化完成，机器人: {self.robot_namespace}, 案例: {self.case}")
        print(f"[INITIALIZE] ✓ PickObject 完整初始化成功: {self.name}")
        print(f"[{self.name}] Starting to pick object with control timer...")
    
    # Note: control_timer_callback removed - using ROS timer instead
    
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
            # Ensure initial closest trajectory index is set before first controller calculation
            if not getattr(self, '_initial_index_set', False):
                print(f"[{self.name}] Initial trajectory index not set, finding closest point before first controller calculation...")
                self._find_closest_trajectory_index(set_as_initial=True, wait_for_pose=False)
            
            # Check if we have a valid stored control sequence we can use
            if self.control_sequence is not None:
                # Use the stored control sequence if available
                if self._apply_stored_control():
                    self._advance_control_step()
                    return True
           
            # If no valid control sequence or need replanning, run MPC
            # NOTE: State lock is already held by control_step(), so we don't need to acquire it again
            current_state = self.current_state.copy()
            
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
                self.control_step_index = 0
                
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
        """Clean up when behavior terminates - with proper resource cleanup"""
        print(f"[{self.name}] 开始终止拾取行为，状态: {new_status}")
        
        # Step 1: Set termination flags
        self.picking_active = False
        
        # Step 2: Stop the robot immediately
        try:
            if self.cmd_vel_pub and hasattr(self, 'node') and self.node:
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
                print(f"[{self.name}] 机器人已停止 [发布停止命令]")
        except Exception as e:
            print(f"[{self.name}] 警告: 停止机器人时出错: {e}")
        
        # Step 3: Clean up timer
        if hasattr(self, 'control_timer') and self.control_timer:
            try:
                self.control_timer.cancel()
                self.control_timer = None
                print(f"[{self.name}] 控制定时器已清理")
            except Exception as e:
                print(f"[{self.name}] 警告: 定时器清理错误: {e}")
        
        # Step 4: Thread-safe subscription cleanup using non-blocking locks
        if self.state_lock.acquire(blocking=False):
            try:
                subscription_errors = []
                
                # Clean up robot pose subscription
                if hasattr(self, 'robot_pose_sub') and self.robot_pose_sub is not None:
                    try:
                        self.node.destroy_subscription(self.robot_pose_sub)
                        self.robot_pose_sub = None
                    except Exception as e:
                        subscription_errors.append(f"robot_pose_sub: {e}")
                
                # Clean up relay pose subscription (静态数据，应该已销毁)
                if hasattr(self, 'relay_pose_sub') and self.relay_pose_sub is not None:
                    try:
                        self.node.destroy_subscription(self.relay_pose_sub)
                        self.relay_pose_sub = None
                    except Exception as e:
                        subscription_errors.append(f"relay_pose_sub: {e}")
                
                # Note: Publishers (cmd_vel_pub, estimated_time_pub) are kept for reuse
                # They will be destroyed when the node is destroyed
                
                if subscription_errors:
                    print(f"[{self.name}] 资源清理警告: {subscription_errors}")
            finally:
                self.state_lock.release()
        else:
            print(f"[{self.name}] 警告: 无法获取锁进行资源清理，跳过订阅清理")
        
        print(f"[{self.name}] 拾取行为终止完成，状态: {new_status}")
    
    def _apply_stored_control(self):
        """Apply the current step from the stored control sequence"""
        if self.control_sequence is None or self.control_step_index >= self.mpc.N_c:
            print(f"[{self.name}] DEBUG: No control sequence or index out of bounds: "
                  f"sequence={self.control_sequence is not None}, "
                  f"index={self.control_step_index}, N_c={self.mpc.N_c if self.mpc else 'N/A'}")
            return False
        
        # Get current control input
        raw_v = float(self.control_sequence[0, self.control_step_index])
        raw_omega = float(self.control_sequence[1, self.control_step_index])
     
        
        # Apply velocity scaling for approach
        # NOTE: State lock is already held by control_step(), so we don't need to acquire it again
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
        print(f'[{self.name}] MPC control step {self.control_step_index+1}/{self.mpc.N_c}: '
              f'v={cmd_msg.linear.x:.3f}, ω={cmd_msg.angular.z:.3f}, '
              f'dist_to_target={distance_to_target:.3f}m, dist_to_traj={distance_to_traj:.3f}m, '
              f'traj_idx={self.trajectory_index}/{len(self.trajectory_data)}')
        
        return True

    def _advance_control_step(self):
        """Advance to the next control step in the stored sequence"""
        if self.control_sequence is None:
            return False
        
        self.control_step_index += 1
        
        # Advanced trajectory progression: only advance index if robot has moved forward along trajectory
        if self.control_step_index == 1:  # Only check on first control step application
            # NOTE: State lock is already held by control_step(), so we don't need to acquire it again
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
        if self.control_step_index >= self.mpc.N_c:
            self.control_step_index = 0
            self.control_sequence = None
            return False
        
        # Otherwise we can use the next step from our stored sequence
        return True
    
    def _check_topic_freshness(self):
        """Check freshness of robot topic data and output debug when stale"""
        current_time = time.time()
        
        # Initialize freshness check counter
        if not hasattr(self, '_freshness_check_count'):
            self._freshness_check_count = 0
        self._freshness_check_count += 1
        
        # Check robot data freshness
        robot_data_fresh = True
        robot_data_exists = (self._last_robot_callback_time is not None and 
                           hasattr(self, '_robot_callback_count') and 
                           self._robot_callback_count > 0)
        
        if robot_data_exists:
            robot_data_age = current_time - self._last_robot_callback_time
            robot_data_fresh = robot_data_age < 3.0  # 3 second timeout
            
            # Only output when data is stale (freshness is False)
            if not robot_data_fresh:
                print(f"[{self.name}] ❌ ROBOT TOPIC FRESHNESS ERROR:")
                print(f"   Topic: /{self.robot_namespace}/odom_map")
                print(f"   Data age: {robot_data_age:.2f}s (threshold: 3.0s)")
                print(f"   Last callback count: {self._robot_callback_count}")
                print(f"   Subscription active: {self.robot_pose_sub is not None}")
        else:
            robot_data_fresh = False
            # Only output when no data received (freshness is False)
            if self._freshness_check_count % 20 == 1:  # Every 2 seconds at 10Hz
                print(f"[{self.name}] ❌ ROBOT TOPIC DATA ERROR:")
                print(f"   Topic: /{self.robot_namespace}/odom_map")
                print(f"   No data received yet")
                print(f"   Callback count: {getattr(self, '_robot_callback_count', 0)}")
                print(f"   Subscription active: {self.robot_pose_sub is not None}")
        
        # Note: This PickObject behavior follows pre-planned trajectory, no parcel subscription needed
        
        # Only output summary when there are freshness issues
        if self._freshness_check_count % 50 == 0 and not robot_data_fresh:
            print(f"[{self.name}] 📊 TOPIC FRESHNESS SUMMARY #{self._freshness_check_count}:")
            print(f"   Robot odometry: ✗ STALE (/{self.robot_namespace}/odom_map)")
            print(f"   Robot callbacks: {getattr(self, '_robot_callback_count', 0)}")
            print(f"   Behavior: PickObject (trajectory-following, no parcel subscription)")
        
        return robot_data_fresh