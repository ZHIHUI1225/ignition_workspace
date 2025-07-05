#!/usr/bin/env python3
"""
Manipulation behavior classes for the behavior tree system.
Contains object manipulation behaviors like pushing and picking.
"""
import py_trees
import rclpy
import rclpy.executors
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
import re
import traceback
import threading  # Keep full threading module for compatibility
from threading import Lock  # Also import Lock directly for clarity
from tf_transformations import euler_from_quaternion
import tf_transformations as tf
import copy

def extract_namespace_number(namespace):
    """Extract numerical index from robot namespace"""
    match = re.search(r'turtlebot(\d+)', namespace)
    return int(match.group(1)) if match else 0
class MobileRobotMPC:
    def __init__(self):
        # MPC parameters - optimized for position convergence
        self.N = 8           # Longer horizon for better convergence
        self.N_c = 3         # Longer control horizon for smoother control
        self.dt = 0.1        # Time step
        
        # Increased weights for better position convergence
        self.Q = np.diag([150.0, 150.0, 20.0, 1.0, 1.0])  # Higher position weights (x, y, theta, v, omega)
        self.R = np.diag([0.5, 0.5])                       # Lower control weights for more aggressive control
        self.F = np.diag([300.0, 300.0, 50.0, 2.0, 2.0])  # Much higher terminal cost weights for convergence
        
        # Conservative velocity constraints for numerical stability
        self.max_vel = 0.2       # m/s
        self.min_vel = 0.0    # m/s - minimal reverse
        self.max_omega = np.pi/3 # rad/s
        self.min_omega = -np.pi/3 # rad/s
        
        # Numerical stability parameters
        self.eps = 1e-6          # Small regularization term
        self.max_angle_diff = np.pi/2  # Maximum angle difference to prevent wrapping issues
        
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
        self.X = self.opti.variable(self.nx, self.N+1)  # State trajectory
        self.U = self.opti.variable(self.nu, self.N)    # Control trajectory
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)          # Initial state
        self.ref = self.opti.parameter(self.nx, self.N+1)  # Reference trajectory

        # Simplified, numerically stable cost function
        cost = 0
        
        for k in range(self.N):
            # Progressive weighting - higher weights later in the horizon for convergence
            progress_factor = 1.0 + 2.0 * k / self.N  # Increases from 1.0 to 3.0
            
            # Position tracking - higher weighted with progressive emphasis
            pos_error_x = self.X[0, k] - self.ref[0, k]
            pos_error_y = self.X[1, k] - self.ref[1, k] 
            cost += progress_factor * self.Q[0,0] * pos_error_x**2 
            cost += progress_factor * self.Q[1,1] * pos_error_y**2
            
            # Orientation tracking - simplified without complex angle wrapping
            theta_error = self.X[2, k] - self.ref[2, k]
            # Simple bounded theta cost to prevent numerical issues
            cost += self.Q[2,2] * ca.fmin(ca.fmax(theta_error**2, 0), 10.0)
            
            # Velocity tracking - bounded to prevent extreme values
            v_error = ca.fmin(ca.fmax(self.X[3, k] - self.ref[3, k], -5.0), 5.0)
            w_error = ca.fmin(ca.fmax(self.X[4, k] - self.ref[4, k], -5.0), 5.0)
            cost += self.Q[3,3] * v_error**2 + self.Q[4,4] * w_error**2
            
            # Control effort - bounded
            u_v = ca.fmin(ca.fmax(self.U[0, k], -2.0), 2.0)
            u_w = ca.fmin(ca.fmax(self.U[1, k], -2.0), 2.0)
            cost += self.R[0,0] * u_v**2 + self.R[1,1] * u_w**2
            
            # Dynamics constraints with bounds checking
            x_next = self.robot_model_safe(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost - simplified
        pos_error_x_f = self.X[0, -1] - self.ref[0, -1]
        pos_error_y_f = self.X[1, -1] - self.ref[1, -1]
        theta_error_f = self.X[2, -1] - self.ref[2, -1]
        
        # Very strong terminal position cost to force convergence
        cost += 5.0 * self.F[0,0] * pos_error_x_f**2 
        cost += 5.0 * self.F[1,1] * pos_error_y_f**2 
        cost += self.F[2,2] * ca.fmin(ca.fmax(theta_error_f**2, 0), 10.0)
        
        # Additional strong terminal position constraint for final convergence
        terminal_pos_error = ca.sqrt(pos_error_x_f**2 + pos_error_y_f**2)
        cost += 1000.0 * terminal_pos_error**2  # Very high weight for final position

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Input constraints with safety margins
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))
        
        # State bounds for numerical stability
        self.opti.subject_to(self.opti.bounded(-10.0, self.X[0, :], 10.0))  # x bounds
        self.opti.subject_to(self.opti.bounded(-10.0, self.X[1, :], 10.0))  # y bounds
        self.opti.subject_to(self.opti.bounded(-2*np.pi, self.X[2, :], 2*np.pi))  # theta bounds
        self.opti.subject_to(self.opti.bounded(-1.0, self.X[3, :], 1.0))   # v bounds
        self.opti.subject_to(self.opti.bounded(-np.pi, self.X[4, :], np.pi)) # omega bounds

        # Solver settings - optimized for speed and numerical stability with THREAD CONTROL
        self.opti.minimize(cost)
        
        # 🔧 CRITICAL: Set environment variables to control BLAS/LAPACK threading
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1' 
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 40,        # Reduced iterations for speed
            'ipopt.max_cpu_time': 0.08,  # Much shorter time limit for responsiveness
            'ipopt.tol': 5e-2,           # Relaxed tolerance for speed
            'ipopt.acceptable_tol': 1e-1, # Very relaxed acceptable tolerance
            'ipopt.acceptable_iter': 2,   # Accept quickly if reasonable
            'ipopt.warm_start_init_point': 'no', # Disable warm start initially for stability
            'ipopt.hessian_approximation': 'limited-memory', # BFGS approximation
            'ipopt.linear_solver': 'mumps', # Try single-threaded linear solver
            'ipopt.mu_strategy': 'monotone', # Stable strategy
            'ipopt.nlp_scaling_method': 'none', # Disable scaling to prevent issues
            'ipopt.constr_viol_tol': 5e-2, # Relaxed constraint violation
        }
        
        self.opti.solver('ipopt', opts)

    def robot_model_safe(self, x, u):
        """Numerically safe system dynamics for 5-state model"""
        # Bound inputs to prevent numerical issues
        u_safe = ca.fmax(ca.fmin(u, [2.0, 2.0]), [-2.0, -2.0])
        
        # Safe trigonometric functions with bounded angles
        theta_safe = ca.fmax(ca.fmin(x[2], 2*np.pi), -2*np.pi)
        cos_theta = ca.cos(theta_safe)
        sin_theta = ca.sin(theta_safe)
        
        return ca.vertcat(
            x[0] + u_safe[0] * cos_theta * self.dt,   # x position
            x[1] + u_safe[0] * sin_theta * self.dt,   # y position  
            x[2] + u_safe[1] * self.dt,               # theta
            u_safe[0],                                # v (control input becomes state)
            u_safe[1]                                 # omega (control input becomes state)
        )

    def robot_model(self, x, u):
        """System dynamics for 5-state model: x_next = f(x, u)"""
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * ca.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt,                 # theta
            u[0],                                  # v (control input becomes state)
            u[1]                                   # omega (control input becomes state)
        )
        

    def set_reference_trajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def update(self, current_state):
        # Simplified, robust state validation
        current_state = np.array(current_state[:5], dtype=np.float64)  # Ensure 5-state
        
        # Basic validation - reject clearly invalid states
        if np.any(np.isnan(current_state)) or np.any(np.isinf(current_state)):
            print(f"MPC: Invalid current state, using fallback")
            return np.zeros((self.nu, self.N_c))
        
        # Basic bounds checking
        current_state[2] = np.clip(current_state[2], -2*np.pi, 2*np.pi)  # theta
        current_state[3] = np.clip(current_state[3], -1.0, 1.0)          # v
        current_state[4] = np.clip(current_state[4], -np.pi, np.pi)      # omega
        
        # Validate reference trajectory
        if self.ref_traj is None:
            return np.zeros((self.nu, self.N_c))
        
        ref_traj = np.array(self.ref_traj)
        if np.any(np.isnan(ref_traj)) or np.any(np.isinf(ref_traj)):
            print(f"MPC: Invalid reference trajectory")
            return np.zeros((self.nu, self.N_c))
        
        # Set problem parameters
        try:
            self.opti.set_value(self.x0, current_state)
            self.opti.set_value(self.ref, ref_traj)
        except Exception as e:
            print(f"MPC: Error setting parameters: {e}")
            return np.zeros((self.nu, self.N_c))
        
        # Simple cold start initialization for stability
        u_init = np.zeros((self.nu, self.N))
        x_init = np.tile(current_state.reshape(-1, 1), (1, self.N+1))
        
        try:
            self.opti.set_initial(self.X, x_init)
            self.opti.set_initial(self.U, u_init)
        except:
            pass  # Continue without initialization if it fails
        
        # Solve with robust error handling
        try:
            sol = self.opti.solve()
            
            # Extract and validate solution
            u_opt = sol.value(self.U)[:, :self.N_c]
            
            # Validate solution
            if np.any(np.isnan(u_opt)) or np.any(np.isinf(u_opt)):
                print(f"MPC: Solution contains NaN/Inf")
                return np.zeros((self.nu, self.N_c))
            
            # Bound solution to reasonable values
            u_opt[0, :] = np.clip(u_opt[0, :], self.min_vel, self.max_vel)
            u_opt[1, :] = np.clip(u_opt[1, :], self.min_omega, self.max_omega)
            
            # Store solution for potential future use
            self.last_solution = {
                'u': sol.value(self.U),
                'x': sol.value(self.X)
            }
            
            return u_opt
            
        except Exception as e:
            print(f"MPC Solver failed: {str(e)}")
            # Clear any stale solution
            self.last_solution = None
            
            # Fallback: Simple proportional control towards reference
            if self.ref_traj is not None and self.ref_traj.shape[1] > 0:
                # Get target position from reference
                target_x = self.ref_traj[0, 0]
                target_y = self.ref_traj[1, 0]
                target_theta = self.ref_traj[2, 0]
                
                # Calculate errors
                dx = target_x - current_state[0]
                dy = target_y - current_state[1]
                dtheta = target_theta - current_state[2]
                
                # Normalize angle error
                while dtheta > np.pi:
                    dtheta -= 2*np.pi
                while dtheta < -np.pi:
                    dtheta += 2*np.pi
                
                # Simple proportional control
                distance = np.sqrt(dx*dx + dy*dy)
                desired_v = min(0.15, max(0.05, 0.3 * distance))  # Speed proportional to distance
                desired_omega = np.clip(2.0 * dtheta, -0.8, 0.8)  # Angular velocity proportional to angle error
                
                # Create fallback control sequence
                fallback_u = np.zeros((self.nu, self.N_c))
                for i in range(self.N_c):
                    fallback_u[0, i] = desired_v
                    fallback_u[1, i] = desired_omega
                
                print(f"MPC: 使用比例控制回退 - v={desired_v:.3f}, ω={desired_omega:.3f}")
                return fallback_u
            
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

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range to prevent numerical issues"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle

    def get_predicted_trajectory(self):
        try:
            return self.opti.debug.value(self.X)
        except:
            return np.zeros((self.nx, self.N+1))

class PushObject(py_trees.behaviour.Behaviour):
    """Push object behavior using MPC controller for trajectory following"""
    
    def __init__(self, name="PushObject", robot_namespace="turtlebot0", distance_threshold=0.14):
        super().__init__(name)
        self.robot_namespace = robot_namespace  # Use provided namespace
        self.case = "simple_maze"  # Default case
        self.number=extract_namespace_number(robot_namespace)
        self.distance_threshold = distance_threshold  # Distance threshold for success condition
        
        # Setup blackboard access for namespaced current_parcel_index only
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.READ
        )
        
        # ROS2 topics for pushing coordination instead of blackboard
        self.pushing_finished_pub = None
        self.pushing_estimated_time_pub = None
        self.pushing_estimated_time = 45.0  # Default value
        
        # ROS node (will be set in setup)
        self.node = None
        
        # State variables (will be reset in initialise)
        self.current_state = None
        self.ref_trajectory = None
        self.trajectory_index = 0
        
        # MPC Controller (will be reset in initialise)
        self.mpc = None
        self.P_HOR = None
        
        # ROS publishers and subscribers (will be reset in initialise)
        self.cmd_vel_pub = None
        self.prediction_pub = None
        self.reference_pub = None
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        self.relay_pose_sub = None
        
        # Pose storage for success checking (will be reset in initialise)
        self.parcel_pose = None
        self.relay_pose = None
        self.current_parcel_index = 0
        
        # Path messages for visualization (will be reset in initialise)
        self.ref_path = None
        self.pred_path = None
        
        # Control loop timer (will be reset in initialise)
        self.control_timer = None
        
        # Control sequence management (will be reset in initialise)
        self.control_sequence = None
        self.control_step = 0
        self.dt = 0.1  # 0.1s timer period for MPC control
        
        # State tracking (will be reset in initialise)
        self.pushing_active = False
        self.start_time = None
        self.last_time = None
        
        # Control error tracking (will be reset in initialise)
        self.last_control_errors = {}
        
        # Thread-safe state lock for callback protection
        self.state_lock = Lock()
    
    def setup(self, **kwargs):
        """设置ROS节点和通信组件（优化：使用共享回调组管理器）"""
        if 'node' in kwargs:
            self.node = kwargs['node']
            
            # 🔧 关键修复：使用共享回调组管理器，避免创建新的回调组
            if hasattr(self.node, 'shared_callback_manager'):
                self.pose_callback_group = self.node.shared_callback_manager.get_group('sensor')
                self.control_callback_group = self.node.shared_callback_manager.get_group('control')
                print(f"[{self.name}] ✅ 使用共享回调组管理器: 传感器组={id(self.pose_callback_group)}, 控制组={id(self.control_callback_group)}")
            else:
                # 错误情况：如果没有共享管理器，则记录错误并使用默认组
                print(f"[{self.name}] ❌ 错误：没有找到shared_callback_manager，无法使用共享回调组")
                self.pose_callback_group = None
                self.control_callback_group = None
                return False
            
            # 获取命名空间参数
            try:
                self.robot_namespace = self.node.get_parameter('robot_namespace').value
            except:
                self.robot_namespace = "turtlebot0"
            
            # 创建发布者（无回调组）
            self.cmd_vel_pub = self.node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10
            )
            self.pushing_finished_pub = self.node.create_publisher(
                Bool, f'/{self.robot_namespace}/pushing_finished', 10
            )
            self.pushing_estimated_time_pub = self.node.create_publisher(
                Float64, f'/{self.robot_namespace}/pushing_estimated_time', 10
            )
            
            # 创建可视化发布者
            self.prediction_pub = self.node.create_publisher(
                Path, f'/{self.robot_namespace}/predicted_path', 10
            )
            self.reference_pub = self.node.create_publisher(
                Path, f'/{self.robot_namespace}/reference_path', 10
            )
            
            # 初始化路径消息对象
            self.ref_path = Path()
            self.ref_path.header.frame_id = 'world'
            self.pred_path = Path()
            self.pred_path.header.frame_id = 'world'
            
            print(f"[{self.name}] ✅ 发布者已创建，使用共享回调组管理器 for {self.robot_namespace}")
            
            # 初始化控制定时器为None (将在update中创建)
            self.control_timer = None
        return True
    
    def _extract_namespace_number(self):
        """Extract numerical index from robot namespace"""
        match = re.search(r'turtlebot(\d+)', self.robot_namespace)
        return int(match.group(1)) if match else 0
    
    def _load_trajectory(self):
        """Load reference trajectory from JSON file"""
        json_file_path = f'/root/workspace/data/{self.case}/tb{self.number}_Trajectory.json'
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            self.original_trajectory = data['Trajectory']  # Store original
            self.ref_trajectory = data['Trajectory'].copy()  # Working copy
            self.interpolation_added = False  # Flag to track if interpolation was added
            
            # Set initial pushing estimated time via ROS topic
            initial_estimated_time = len(self.ref_trajectory) * self.dt
            self.pushing_estimated_time = initial_estimated_time
            self.publish_pushing_estimated_time()
    
    def _add_interpolation_to_trajectory(self):
        """Add interpolation points for smoother approach between robot current position and first trajectory point"""
        if self.interpolation_added or self.current_state is None or self.original_trajectory is None:
            return
        
        # Get robot current position and first trajectory point
        robot_pos = [self.current_state[0], self.current_state[1], self.current_state[2]]
        first_traj_point = self.original_trajectory[0]
        
        # Calculate distance between robot and first trajectory point
        distance = np.sqrt((robot_pos[0] - first_traj_point[0])**2 + 
                          (robot_pos[1] - first_traj_point[1])**2)
        
        # Only add interpolation if distance is significant (> 1cm)
        if distance > 0.01:
            # Bezier interpolation for smoother approach, preserving start/end orientation
            p0 = np.array(robot_pos[:3], dtype=float)
            p1 = np.array(first_traj_point[:3], dtype=float)
            d = np.linalg.norm(p1[:2] - p0[:2])
            handle_len = 0.3 * d
            ctrl1 = p0[:2] + handle_len * np.array([np.cos(p0[2]), np.sin(p0[2])])
            ctrl2 = p1[:2] - handle_len * np.array([np.cos(p1[2]), np.sin(p1[2])])
            bezier_points = np.array([p0[:2], ctrl1, ctrl2, p1[:2]])
            n_points = max(int(d / 0.01), 1)
            ts = np.linspace(0, 1, n_points + 1)
            def bezier(t, points):
                return (
                    (1 - t) ** 3 * points[0] +
                    3 * (1 - t) ** 2 * t * points[1] +
                    3 * (1 - t) * t ** 2 * points[2] +
                    t ** 3 * points[3]
                )
            interp_points = []
            dt = self.dt if hasattr(self, 'dt') else 0.1
            for i, t in enumerate(ts):
                pos = bezier(t, bezier_points)
                theta = (1 - t) * p0[2] + t * p1[2]
                if i < n_points:
                    pos_next = bezier(ts[i+1], bezier_points)
                    distance_to_next = np.linalg.norm(pos_next - pos)
                    estimated_v = min(distance_to_next / dt, 0.1)
                else:
                    estimated_v = first_traj_point[3]
                interp_omega = 0.0
                interp_points.append([pos[0], pos[1], theta, estimated_v, interp_omega])
            self.ref_trajectory = interp_points + self.original_trajectory
            self.interpolation_added = True
            new_estimated_time = len(self.ref_trajectory) * self.dt
            self.pushing_estimated_time = new_estimated_time
            self.publish_pushing_estimated_time()
        else:
            self.interpolation_added = True

    
    def robot_pose_callback(self, msg):
        """Update robot state from odometry message - 优化为非阻塞快速处理"""
        try:
            # 记录回调时间戳和调试信息
            current_time = time.time()
            self._last_robot_callback_time = current_time
            
            # 🔧 关键调试：跟踪回调频率，特别是第三个机器人
            if not hasattr(self, '_robot_callback_count'):
                self._robot_callback_count = 0
                self._robot_callback_start_time = current_time
                print(f"[{self.name}][{self.robot_namespace}] 🎯 开始接收机器人回调数据")
            
            self._robot_callback_count += 1
            
            # 安全访问 _robot_callback_start_time，确保已初始化
            if hasattr(self, '_robot_callback_start_time'):
                time_since_start = current_time - self._robot_callback_start_time
                # 每100次回调或5秒钟打印一次状态
                if (self._robot_callback_count % 100 == 1) or (time_since_start > 0 and int(time_since_start) % 5 == 0 and self._robot_callback_count % 10 == 1):
                    callback_rate = self._robot_callback_count / time_since_start if time_since_start > 0 else 0
                    print(f"[{self.name}][{self.robot_namespace}] 🔄 机器人回调 #{self._robot_callback_count} (频率: {callback_rate:.1f} Hz)")
                    
                    # 特别关注第三个机器人（turtlebot2）
                    if 'turtlebot2' in self.robot_namespace:
                        print(f"[{self.name}][{self.robot_namespace}] 🚨 第三个机器人回调活跃，系统状态正常")
                        print(f"[{self.name}][{self.robot_namespace}] 🎯 第三机器人状态: 节点={self.node.get_name()}, 回调组={id(getattr(self, 'pose_callback_group', None))}")
            
            # 快速数据提取和更新（最小化锁持有时间）
            position_x = msg.pose.pose.position.x
            position_y = msg.pose.pose.position.y
            
            # 快速四元数转欧拉角
            quat = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            euler = tf.euler_from_quaternion(quat)
            yaw = euler[2]
            
            # 快速速度提取
            linear_x = msg.twist.twist.linear.x
            angular_z = msg.twist.twist.angular.z
            
            # 原子操作更新状态（最短锁时间）
            with self.state_lock:
                if self.current_state is None:
                    self.current_state = np.zeros(5)  # [x, y, theta, v, omega]
                
                self.current_state[0] = position_x
                self.current_state[1] = position_y
                self.current_state[2] = yaw
                self.current_state[3] = linear_x
                self.current_state[4] = angular_z
                
                # 添加到缓存
                try:
                    self._add_to_robot_cache(self.current_state)
                except Exception as cache_error:
                    # 缓存失败不应该影响主要功能
                    pass

            
        except Exception as e:
            print(f"[{self.name}] ERROR in robot_pose_callback: {e}")
            # 不打印完整traceback以避免I/O阻塞

    def parcel_pose_callback(self, msg):
        """Update parcel pose from PoseStamped message - 优化为原子操作"""
        try:
            # 原子操作更新姿态数据（最小化锁持有时间）
            with self.state_lock:
                self.parcel_pose = msg.pose
                # 添加到缓存
                try:
                    self._add_to_parcel_cache(msg.pose)
                except Exception as cache_error:
                    # 缓存失败不应该影响主要功能
                    pass
            
            # 无锁频率统计（避免阻塞）
            current_time = time.time()
            if not hasattr(self, '_last_parcel_callback_time'):
                self._last_parcel_callback_time = current_time
                self._parcel_callback_count = 0
                self._parcel_callback_start_time = current_time  # 记录开始时间用于频率计算
                print(f"[{self.name}] 📦 首次接收包裹姿态数据 (节点: {self.node.get_name() if self.node else 'None'})")
            else:
                self._parcel_callback_count += 1
                
                # 大幅降低调试输出频率以减少I/O阻塞
                if self._parcel_callback_count % 100 == 0:
                    if hasattr(self, '_parcel_callback_start_time'):
                        time_since_start = current_time - self._parcel_callback_start_time
                        frequency = self._parcel_callback_count / time_since_start if time_since_start > 0 else 0
                        print(f"[{self.name}] 📦 包裹话题频率: {frequency:.1f} Hz (总回调: {self._parcel_callback_count})")
                        # 重置计数器避免数值过大，但保持开始时间
                        self._parcel_callback_start_time = current_time
                        self._parcel_callback_count = 0
            
            # CRITICAL: Always update the timestamp for freshness checking
            self._last_parcel_callback_time = current_time
                    
        except Exception as e:
            print(f"[{self.name}] ERROR in parcel_pose_callback: {e}")
            # 避免打印完整traceback以减少I/O阻塞

    def relay_pose_callback(self, msg):
        """Update relay point pose from PoseStamped message - read once (static pose)"""
        # 中继点位置是静态的，只需要读取一次
        if self.relay_pose is None:
            with self.state_lock:
                self.relay_pose = msg.pose
            print(f"[{self.name}] 🏁 中继点位置已读取: x={self.relay_pose.position.x:.3f}, y={self.relay_pose.position.y:.3f} (静态，无需重复更新)")
            
            # 读取一次后立即销毁订阅以节省资源
            if hasattr(self, 'relay_pose_sub') and self.relay_pose_sub is not None:
                try:
                    self.node.destroy_subscription(self.relay_pose_sub)
                    self.relay_pose_sub = None
                    print(f"[{self.name}] 🏁 中继点订阅已销毁（静态数据，无需持续订阅）")
                except Exception as e:
                    print(f"[{self.name}] 警告: 销毁中继点订阅时出错: {e}")
        # 如果已经有中继点数据，忽略后续消息（不应该发生，因为订阅已销毁）
    
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
            print(f"[{self.name}] SUCCESS: Parcel is within relay range (distance: {distance:.3f}m <= {self.distance_threshold:.3f}m)")
        
        return is_in_range
    
    def _check_robot_close_to_target(self):
        """Check if robot is close to its target state (distance < 0.03m)"""
        if self.current_state is None or self.ref_trajectory is None:
            return False
        
        # Calculate the target position from the reference trajectory

        # Use the final point of the trajectory
        target_x = self.ref_trajectory[-1][0]
        target_y = self.ref_trajectory[-1][1]
    
        # Calculate Euclidean distance between robot current position and target position
        robot_x = self.current_state[0]
        robot_y = self.current_state[1]
        distance = math.sqrt((robot_x - target_x)**2 + (robot_y - target_y)**2)
        
        is_close = distance < 0.02  # 5cm threshold
        
        if is_close:
            print(f"[{self.name}] SUCCESS: Robot is close to target state(x:{target_x},y:{target_y}) (distance: {distance:.3f}m)")
        
        return is_close
    
    def _check_robot_in_relay_range(self):
        """Check if robot is within distance threshold of relay point"""
        if self.current_state is None or self.relay_pose is None:
            return False
        
        # Convert current robot state to geometry_msgs format for distance calculation
        robot_x = self.current_state[0]
        robot_y = self.current_state[1]
        
        # Calculate distance between robot and relay point
        relay_x = self.relay_pose.position.x
        relay_y = self.relay_pose.position.y
        
        distance = math.sqrt((robot_x - relay_x)**2 + (robot_y - relay_y)**2)
        is_in_range = distance <= 0.06 # 6cm threshold
        if is_in_range:
            print(f"[{self.name}] SUCCESS: Robot is within relay range (distance: {distance:.3f}m <= {self.distance_threshold:.3f}m)")
        
        return is_in_range
    
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
            
            # Update local variable and publish via ROS topic instead of blackboard
            self.pushing_estimated_time = estimated_time
            if self.pushing_estimated_time_pub:
                msg = Float64()
                msg.data = estimated_time
                self.pushing_estimated_time_pub.publish(msg)
            
            # Debug output removed - only show update time periodically
            if not hasattr(self, '_print_call_count'):
                self._print_call_count = 0
            
            self._print_call_count += 1
            if self._print_call_count >= 20:  # Print every 20 calls
                # Silent update - no debug output
                self._print_call_count = 0
    
    def _publish_reference_trajectory(self):
        """Publish reference trajectory for visualization"""
        if not self.ref_trajectory or self.ref_path is None:
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
        
        if self.reference_pub is not None:
            self.reference_pub.publish(self.ref_path)
    
    def _publish_predicted_trajectory(self):
        """Publish predicted trajectory from MPC with enhanced validation"""
        try:
            pred_traj = self.mpc.get_predicted_trajectory()
            if pred_traj is None or self.pred_path is None:
                # Silent return - no debug output
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
                if len(self.pred_path.poses) > 0 and self.prediction_pub is not None:
                    self.prediction_pub.publish(self.pred_path)
                # Silent - no debug output for empty poses
            else:
                # Silent - no debug output for invalid trajectory data
                pass
            
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
        
        # 🔧 NOTE: Trajectory index advancement is handled by MPC success/failure logic
        # Do not advance trajectory index here - it's managed by:
        # - MPC success: advance by N_c steps  
        # - MPC failure: advance by 1 step in PI control
        
        # Update pushing estimated time 
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
            return False
        
        # CRITICAL: Validate pose data before applying control
        if not self._has_valid_pose_data():
            print(f"[{self.name}] CONTROL SKIPPED: Missing pose data in apply_stored_control - not applying stored command")
            return False
        
        cmd_msg = Twist()
        cmd_msg.linear.x = float(self.control_sequence[0, self.control_step])
        cmd_msg.angular.z = float(self.control_sequence[1, self.control_step])
        
        # Check if publisher exists
        if self.cmd_vel_pub is None:
            print(f"[{self.name}] ERROR: cmd_vel_pub is None, cannot publish command")
            return False
            
        self.cmd_vel_pub.publish(cmd_msg)
        
        # Debug output for control verification
        print(f"[{self.name}] 控制命令已发布: v={cmd_msg.linear.x:.3f}, ω={cmd_msg.angular.z:.3f} [存储序列 {self.control_step+1}/{self.mpc.N_c}]")
        return True
    
    def control_loop(self):
        """Main MPC control loop - optimized with reduced solving frequency"""
        # CRITICAL: Check if we have valid pose data before attempting control
        # Do not send control commands if no pose data has been received
        if not self._has_valid_pose_data():
            # Throttle "missing pose data" warnings to reduce CPU usage
            if not hasattr(self, '_pose_warning_count'):
                self._pose_warning_count = 0
            self._pose_warning_count += 1
            
            if self._pose_warning_count % 100 == 1:  # Only warn every 100 iterations (10 seconds at 10Hz)
                self._log_pose_data_status("CONTROL SKIPPED: Missing pose data")
                print(f"[{self.name}] No control commands sent - waiting for valid pose data (warning #{self._pose_warning_count})")
            
            # Publish zero velocity to ensure robot stops
            if self.cmd_vel_pub:
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
            
            return  # Exit control loop without sending movement commands
        
        # Check if we have a valid stored control sequence we can use
        if self.control_sequence is not None:
            # Apply the stored control sequence
            if self.apply_stored_control():
                self.advance_control_step()
            return

        # No stored sequence available - calculate new MPC control output
        if not self.ref_trajectory:
            # Throttle trajectory warnings
            if not hasattr(self, '_traj_warning_count'):
                self._traj_warning_count = 0
            self._traj_warning_count += 1
            
            if self._traj_warning_count % 50 == 1:  # Only warn every 50 iterations (5 seconds at 10Hz)
                print(f"[{self.name}] Warning: No reference trajectory available (warning #{self._traj_warning_count})")
            return

        with self.state_lock:
            try:
                # Calculate progress to determine if we should continue MPC control
                curr_pos = np.array([self.current_state[0], self.current_state[1]])
                final_pos = np.array([self.ref_trajectory[-1][0], self.ref_trajectory[-1][1]])
                dist_to_final = np.linalg.norm(curr_pos - final_pos)
                
                # Check for success condition first
                pushing_complete = self._check_parcel_in_relay_range()
                print(f"[{self.name}] pushing_complete: {pushing_complete}, dist_to_final: {dist_to_final:.3f}m")
                
                # Only run MPC if not complete and no stored sequence exists
                if not pushing_complete:
                        # Reduced debug output for performance
                        if not hasattr(self, '_mpc_call_counter'):
                            self._mpc_call_counter = 0
                        self._mpc_call_counter += 1
                        
    
                        
                        # Step 1: Calculate current position error to determine if we need closest point search
                        if self.trajectory_index < len(self.ref_trajectory):
                            ref_pos = np.array([self.ref_trajectory[self.trajectory_index][0], 
                                              self.ref_trajectory[self.trajectory_index][1]])
                            curr_pos = np.array([self.current_state[0], self.current_state[1]])
                            current_error = np.linalg.norm(curr_pos - ref_pos)
                        else:
                            # At end of trajectory, calculate error from final point
                            ref_pos = np.array([self.ref_trajectory[-1][0], self.ref_trajectory[-1][1]])
                            curr_pos = np.array([self.current_state[0], self.current_state[1]])
                            current_error = np.linalg.norm(curr_pos - ref_pos)
                        
                        # Reduced position error logging frequency
                        if self._mpc_call_counter % 20 == 0:
                            print(f"[{self.name}] 📏 当前位置误差: {current_error:.4f}m")
                        
                        # Step 2: If error > 0.02m, find closest trajectory point and update trajectory index
                        if current_error > 0.02:
                            # Optimized closest point search with limited range
                            best_idx, min_dist = self._find_closest_reference_point(current_error)
                            
                            # Update trajectory index to closest point (forward-only)
                            old_idx = self.trajectory_index
                            self._safe_update_trajectory_index(max(self.trajectory_index, best_idx), "error_correction")
                            
                            # Reduced logging for performance - only log significant changes
                            if self.trajectory_index > old_idx + 5:  # Only log if significant jump
                                print(f"[{self.name}] ✅ 轨迹索引已更新: {old_idx} -> {self.trajectory_index} (最近距离: {min_dist:.3f}m)")
                        # Removed verbose logging for small errors to improve performance
                        
                        # Step 3: Prepare reference trajectory for MPC (from current index)
                        ref_array = np.zeros((5, self.mpc.N+1))
                        for i in range(self.mpc.N + 1):
                            if self.trajectory_index + i < len(self.ref_trajectory):
                                traj_idx = self.trajectory_index + i
                                ref_point = self.ref_trajectory[traj_idx]
                            else:
                                # Use last trajectory point if beyond end
                                ref_point = self.ref_trajectory[-1]
                            
                            ref_array[0, i] = ref_point[0]  # x
                            ref_array[1, i] = ref_point[1]  # y
                            ref_array[2, i] = ref_point[2]  # theta
                            ref_array[3, i] = ref_point[3]  # v
                            ref_array[4, i] = ref_point[4]  # omega
                    
                        current_state = self.current_state.copy()
                        
                        # Step 4: Try MPC solution
                        try:
                           
                            self.mpc.set_reference_trajectory(ref_array)
                            u_sequence = self.mpc.update(current_state)
                            
                            # Calculate and output control errors (with reduced frequency)
                            self._calculate_and_output_control_errors(current_state, ref_array)
                            
                            # Step 5: Check if MPC solution is valid
                            if u_sequence is None or np.isnan(u_sequence).any():
                                print(f"[{self.name}] ❌ MPC返回无效解，切换到PI控制")
                                self._handle_mpc_failure(current_state, ref_array)
                                return
                            
                            # Step 6: MPC success - store control sequence and apply first command
                            print(f"[{self.name}] ✅ MPC求解成功，存储 {self.mpc.N_c} 步控制序列")
                            self.control_sequence = u_sequence
                            self.control_step = 0
                            
                            # Apply first control command
                            cmd_msg = Twist()
                            cmd_msg.linear.x = float(u_sequence[0, 0])
                            cmd_msg.angular.z = float(u_sequence[1, 0])
                            self.cmd_vel_pub.publish(cmd_msg)
                            
                            print(f'[{self.name}] 🚀 MPC控制命令已发布: v={cmd_msg.linear.x:.3f}, ω={cmd_msg.angular.z:.3f} [新MPC解 1/{self.mpc.N_c}]')
                            
                            # Step 7: Advance trajectory index by N_c for next MPC cycle
                            old_traj_idx = self.trajectory_index
                            new_traj_idx = min(self.trajectory_index + self.mpc.N_c, len(self.ref_trajectory) - 1)
                            self._safe_update_trajectory_index(new_traj_idx, "mpc_success_advance")
                            print(f"[{self.name}] � 轨迹索引前进 N_c={self.mpc.N_c} 步: {old_traj_idx} -> {self.trajectory_index}")
                            
                            # Start from next step for stored sequence
                            self.control_step = 1
                            
                            # Update estimated time and publish visualization
                            self._update_pushing_estimated_time()
                            self._publish_predicted_trajectory()
                            
                        except Exception as e:
                            print(f'[{self.name}] ❌ MPC解算异常: {str(e)}，切换到PI控制')
                            self._handle_mpc_failure(current_state, ref_array)
                        
                        # Publish reference trajectory for visualization
                        self._publish_reference_trajectory()
                
                else:
                    # If pushing is complete or at end of trajectory, stop the robot
                    cmd_msg = Twist()
                    cmd_msg.linear.x = 0.0
                    cmd_msg.angular.z = 0.0
                    self.cmd_vel_pub.publish(cmd_msg)
                    print(f"[{self.name}] 推送完成或轨迹结束 - 停止机器人 [停止命令已发布]")
                
            except Exception as e:
                print(f"[{self.name}] Error in control loop: {e}")
                traceback.print_exc()
    
    def _calculate_and_output_control_errors(self, current_state, ref_array):
        """Calculate and output control errors between current state and reference state - optimized for speed"""
        try:
            # Current position error (robot vs reference at current time step)
            current_pos_error = np.sqrt((current_state[0] - ref_array[0, 0])**2 + 
                                       (current_state[1] - ref_array[1, 0])**2)
            current_angle_error = abs(current_state[2] - ref_array[2, 0])
            # Normalize angle error to [0, pi] for proper convergence assessment
            current_angle_error = min(current_angle_error, 2*np.pi - current_angle_error)
            
            # Store errors for potential logging or further analysis
            self.last_control_errors = {
                'current_pos_error': current_pos_error,
                'current_angle_error': current_angle_error
            }
            
            # Reduced output frequency for performance - only output significant errors
            if not hasattr(self, '_error_output_counter'):
                self._error_output_counter = 0
            self._error_output_counter += 1
            
            # Only output every 10th error calculation or if error is large
            if self._error_output_counter % 10 == 0 or current_pos_error > 0.1:
                print(f"[{self.name}] Control Errors: pos={current_pos_error:.4f}m, θ={current_angle_error:.4f}rad({np.degrees(current_angle_error):.1f}°)")
            
        except Exception as e:
            print(f"[{self.name}] Error calculating control errors: {e}")
            # Set default error values if calculation fails
            self.last_control_errors = {
                'current_pos_error': float('inf'),
                'current_angle_error': float('inf')
            }
    
    def setup_parcel_subscription(self):
        """Set up parcel subscription when blackboard is ready - 使用ReentrantCallbackGroup优化性能"""
        if self.node is None:
            print(f"[{self.name}] WARNING: Cannot setup parcel subscription - no ROS node")
            return False
            
        try:
            # Try to get current parcel index from blackboard with proper error handling
            try:
                current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
                print(f"[{self.name}] 调试: 从黑板检索包裹索引: {current_parcel_index}")
            except Exception as bb_error:
                # Blackboard key doesn't exist yet - use default value
                print(f"[{self.name}] 信息: 黑板键尚未就绪，使用默认包裹索引0: {bb_error}")
                current_parcel_index = 0
            
            # Always update self.current_parcel_index and create subscription
            old_index = getattr(self, 'current_parcel_index', 'none')
            self.current_parcel_index = current_parcel_index
            
            # Clean up existing subscription if it exists
            if self.parcel_pose_sub is not None:
                self.node.destroy_subscription(self.parcel_pose_sub)
                self.parcel_pose_sub = None
                print(f"[{self.name}] 调试: 已销毁现有包裹订阅")
                
            # 🔧 使用共享回调组管理器 - 不再创建临时回调组
            if not hasattr(self, 'pose_callback_group') or self.pose_callback_group is None:
                print(f"[{self.name}] ❌ 错误: 共享回调组未正确设置，无法创建包裹订阅")
                return False
            
            # Create new parcel subscription (使用可靠QoS和共享ReentrantCallbackGroup)
            parcel_topic = f'/parcel{current_parcel_index}/pose'
            reliable_qos = self._create_reliable_qos_profile()
            self.parcel_pose_sub = self.node.create_subscription(
                PoseStamped,
                parcel_topic,
                self.parcel_pose_callback,
                reliable_qos,
                callback_group=self.pose_callback_group
            )
            print(f"[{self.name}] ✅ 成功订阅包裹话题: {parcel_topic} (索引变化: {old_index} -> {current_parcel_index}) [使用ReentrantCallbackGroup支持高频并发, 节点: {self.node.get_name()}]")
            return True
            
        except Exception as e:
            print(f"[{self.name}] ERROR: Failed to setup parcel subscription: {e}")
            traceback.print_exc()
            return False

    def setup_relay_subscription(self):
        """Set up relay subscription - one-shot for static pose, 使用ReentrantCallbackGroup"""
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
            
            # 🔧 使用共享回调组管理器 - 不再创建临时回调组
            if not hasattr(self, 'pose_callback_group') or self.pose_callback_group is None:
                print(f"[{self.name}] ❌ 错误: 共享回调组未正确设置，无法创建中继点订阅")
                return False
            
            # Subscribe to relay point pose (一次性读取静态数据，使用可靠QoS)
            relay_number = self._extract_namespace_number() + 1  # Relaypoint{i+1}
            relay_topic = f'/Relaypoint{relay_number}/pose'
            reliable_qos = self._create_reliable_qos_profile()
            self.relay_pose_sub = self.node.create_subscription(
                PoseStamped, relay_topic,
                self.relay_pose_callback, reliable_qos,
                callback_group=self.pose_callback_group)
            print(f"[{self.name}] ✅ 成功订阅中继话题: {relay_topic} (中继点: {relay_number}) [一次性读取，使用ReentrantCallbackGroup, 节点: {self.node.get_name()}]")
            return True
            
        except Exception as e:
            print(f"[{self.name}] ERROR: Failed to setup relay subscription: {e}")
            return False

    def setup_robot_subscription(self):
        """Set up robot pose subscription - 使用ReentrantCallbackGroup优化性能"""
        if self.node is None:
            print(f"[{self.name}] WARNING: Cannot setup robot subscription - no ROS node")
            return False
            
        try:
            # Clean up existing subscription if it exists
            if self.robot_pose_sub is not None:
                self.node.destroy_subscription(self.robot_pose_sub)
                self.robot_pose_sub = None
            
            # 🔧 使用共享回调组管理器 - 不再创建临时回调组
            if not hasattr(self, 'pose_callback_group') or self.pose_callback_group is None:
                print(f"[{self.name}] ❌ 错误: 共享回调组未正确设置，无法创建机器人订阅")
                return False
            
            # Subscribe to robot odometry (使用可靠QoS和共享ReentrantCallbackGroup) 
            robot_odom_topic = f'/turtlebot{self._extract_namespace_number()}/odom_map'
            reliable_qos = self._create_reliable_qos_profile()
            self.robot_pose_sub = self.node.create_subscription(
                Odometry, robot_odom_topic, self.robot_pose_callback, reliable_qos,
                callback_group=self.pose_callback_group)
            
            # 🔧 关键调试：记录订阅创建信息
            print(f"[{self.name}] ✅ 成功订阅机器人话题: {robot_odom_topic}")
            print(f"[{self.name}] 🔧 节点名称: {self.node.get_name()}")
            print(f"[{self.name}] 🔧 回调组ID: {id(self.pose_callback_group)}")
            print(f"[{self.name}] 🔧 订阅对象ID: {id(self.robot_pose_sub)}")
            print(f"[{self.name}] 🔧 机器人命名空间: {self.robot_namespace}")
            
            return True
            
        except Exception as e:
            print(f"[{self.name}] ERROR: Failed to setup robot subscription: {e}")
            import traceback
            traceback.print_exc()
            return False

    def initialise(self):
        """初始化行为状态并启动专用控制线程"""
        print(f"[{self.name}] 开始初始化推送行为...")
        
        # 初始化数据缓存机制
        try:
            self._init_data_cache()
            print(f"[{self.name}] ✅ 数据缓存机制已初始化")
        except Exception as e:
            print(f"[{self.name}] ⚠️ 缓存初始化失败: {e}，继续执行")
        
        # 重置状态变量 - 使用None而不是zeros来区分"未初始化"和"收到零值"
        self.current_state = None
        self.pushing_active = True
        self.start_time = time.time()
        self.parcel_pose = None
        self.relay_pose = None
        
        # 重置轨迹索引到起始位置
        self.trajectory_index = 0
        
        # 添加初始化完成标志，防止控制循环在数据到达前运行
        self.initialization_complete = False
        
        # 重置回调计数器用于调试
        self._robot_callback_count = 0
        self._parcel_callback_count = 0
        
        # 线程安全标志
        self.terminating = False
        
        # 重置PI控制器
        self._reset_pi_controller()
        
        # 设置初始状态发布
        self.publish_pushing_finished(False)
        
        # 创建新的MPC控制器
        self.mpc = MobileRobotMPC()
        self.P_HOR = self.mpc.N
        
        # 加载轨迹
        self._load_trajectory()
        
        # 设置ROS订阅（使用优化的回调组和QoS）
        print(f"[{self.name}] 设置ROS订阅...")
        robot_sub_ok = self.setup_robot_subscription()
        parcel_sub_ok = self.setup_parcel_subscription()
        relay_sub_ok = self.setup_relay_subscription()
        
        print(f"[{self.name}] 订阅状态: robot={robot_sub_ok}, parcel={parcel_sub_ok}, relay={relay_sub_ok}")
        
        # 关键：等待更长时间确保订阅建立和回调开始接收数据
        print(f"[{self.name}] 等待订阅建立和数据接收...")
        time.sleep(0.8)  # 增加等待时间到800ms以确保gazebo到ros2转换稳定
        self.check_topic_connectivity()
        
        # Non-blocking data availability check (no while loops in timer callbacks)
        print(f"[{self.name}] 检查关键数据可用性...")
        
        # Check if we have sufficient data to start control (non-blocking)
        robot_has_data = (self.current_state is not None and 
                        hasattr(self, '_robot_callback_count') and 
                        self._robot_callback_count > 0)
        parcel_has_data = self.parcel_pose is not None
        
        if robot_has_data and parcel_has_data:
            print(f"[{self.name}] ✅ 关键数据已就绪")
        else:
            robot_status = "✓" if robot_has_data else "✗"
            parcel_status = "✓" if parcel_has_data else "✗"
            print(f"[{self.name}] ⚠️ 等待数据: robot:{robot_status} parcel:{parcel_status} - 控制将在数据到达后自动开始")
        
        # 最终数据检查
        robot_has_data = (self.current_state is not None and 
                        hasattr(self, '_robot_callback_count') and 
                        self._robot_callback_count > 0)
        parcel_has_data = self.parcel_pose is not None
        relay_has_data = self.relay_pose is not None
        
        print(f"[{self.name}] 初始化数据检查: robot_data={robot_has_data}, parcel_data={parcel_has_data}, relay_data={relay_has_data}")
        if not robot_has_data or not parcel_has_data:
            print(f"[{self.name}] ⚠️ 警告: 初始化后缺少关键姿态数据，将使用缓存机制和容错处理")
        
        # 设置初始化完成标志
        self.initialization_complete = True
        
        # 启动控制定时器（替代专用线程）- 使用共享回调组
        if hasattr(self, 'control_callback_group') and self.control_callback_group is not None:
            self.control_timer = self.node.create_timer(
                self.dt,  # 0.1s 定时器周期
                self.control_loop,
                callback_group=self.control_callback_group  # 使用共享回调组
            )
            print(f"[{self.name}] ✅ 控制定时器已启动 (周期: {self.dt}s，使用共享回调组)")
        else:
            # 回退方案：使用默认回调组
            self.control_timer = self.node.create_timer(
                self.dt,  # 0.1s 定时器周期
                self.control_loop
            )
            print(f"[{self.name}] ✅ 控制定时器已启动 (周期: {self.dt}s，使用默认回调组)")
        print(f"[{self.name}] 推送行为初始化完成")
        
    def check_topic_connectivity(self):
        """验证话题数据流连通性（使用回调组避免阻塞）"""
        topics = [
            f'/turtlebot{self._extract_namespace_number()}/odom_map',
            f'/parcel{self.current_parcel_index}/pose',
            f'/Relaypoint{self._extract_namespace_number()+1}/pose'
        ]
        
        print(f"[{self.name}] 📊 话题连通性检查 (节点: {self.node.get_name() if self.node else 'None'})")
        
        for topic in topics:
            try:
                publishers = self.node.count_publishers(topic)
                subscribers = self.node.count_subscribers(topic)
                if publishers == 0:
                    print(f"⚠️ 话题 {topic} 无发布者！")
                else:
                    # 特殊处理中继点话题
                    if "Relaypoint" in topic:
                        if self.relay_pose is not None:
                            print(f"✅ 中继点数据已获取 {topic} (发布者: {publishers}, 订阅者: {subscribers}) [静态数据]")
                        else:
                            print(f"✅ 已连接 {topic} (发布者: {publishers}, 订阅者: {subscribers}) [等待静态数据]")
                    else:
                        print(f"✅ 已连接 {topic} (发布者: {publishers}, 订阅者: {subscribers}) [回调组模式]")
            except Exception as e:
                print(f"❌ 话题检查失败: {topic} - {str(e)}")
                
        # Additional debug: Check if callbacks are actually being triggered
        print(f"[{self.name}] 📊 回调状态检查:")
        print(f"   机器人状态: {self.current_state is not None and not np.allclose(self.current_state[:3], [0.0, 0.0, 0.0])} (回调次数: {getattr(self, '_robot_callback_count', 0)})")
        print(f"   包裹姿态: {self.parcel_pose is not None} (回调次数: {getattr(self, '_parcel_callback_count', 0)})")
        print(f"   中继点姿态: {self.relay_pose is not None}")
        
        # Check subscription objects
        print(f"[{self.name}] 📊 订阅对象状态:")
        print(f"   robot_pose_sub: {self.robot_pose_sub is not None}")
        print(f"   parcel_pose_sub: {self.parcel_pose_sub is not None}")
        print(f"   relay_pose_sub: {self.relay_pose_sub is not None}")
        print(f"   callback_group: {hasattr(self, 'callback_group') and self.callback_group is not None}")
        
        # Force a topic list check to see what topics actually exist
        try:
            topic_names_and_types = self.node.get_topic_names_and_types()
            available_topics = [name for name, _ in topic_names_and_types]
            print(f"[{self.name}] 📊 系统可用话题数量: {len(available_topics)}")
            
            # Check if our expected topics exist
            for topic in topics:
                if topic in available_topics:
                    print(f"   ✅ 话题存在: {topic}")
                else:
                    print(f"   ❌ 话题不存在: {topic}")
                    
        except Exception as e:
            print(f"[{self.name}] 警告: 无法检查系统话题列表: {e}")
    
    def publish_pushing_estimated_time(self):
        """Publish the pushing estimated time via ROS topic"""
        if self.pushing_estimated_time_pub:
            msg = Float64()
            msg.data = self.pushing_estimated_time
            self.pushing_estimated_time_pub.publish(msg)
    
    def publish_pushing_finished(self, finished_status):
        """Publish the pushing finished status via ROS topic"""
        if self.pushing_finished_pub:
            msg = Bool()
            msg.data = finished_status
            self.pushing_finished_pub.publish(msg)

    def update(self):
        """Update pushing behavior status - behavior tree logic only"""
        if self.start_time is None:
            self.start_time = time.time()
        
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor
        # which handles all callbacks including our timer automatically
        
        # Periodically publish pushing_finished = False while still running
        # to ensure other robots know this robot is still pushing
        if hasattr(self, '_last_false_publish_time'):
            if time.time() - self._last_false_publish_time > 1.0:  # Publish every 1 second
                self.publish_pushing_finished(False)
                self._last_false_publish_time = time.time()
        else:
            # First time - publish False and set timer
            self.publish_pushing_finished(False)
            self._last_false_publish_time = time.time()
        
        # TRIPLE SUCCESS CONDITIONS: Check if any condition is met
        # Condition 1: Parcel has reached relay point
        parcel_in_relay_range = self._check_parcel_in_relay_range()
        
        # Condition 2: Robot is close to its target state (distance < 0.03m)
        robot_close_to_target = self._check_robot_close_to_target()
        
        # Condition 3: Robot is in relay range (NEW SUCCESS CONDITION)
        robot_in_relay_range = self._check_robot_in_relay_range()
        
        if parcel_in_relay_range:
            print(f"[{self.name}] SUCCESS: Parcel has reached the relay point")
            self.pushing_active = False
            # Publish pushing_finished = True via ROS topic
            self.publish_pushing_finished(True)
            print(f"[{self.name}] DEBUG: Published pushing_finished = True via ROS topic (parcel reached relay)")
            return py_trees.common.Status.SUCCESS
        elif robot_close_to_target:
            print(f"[{self.name}] SUCCESS: Robot is close to target state (< 0.03m)")
            self.pushing_active = False
            # Publish pushing_finished = True via ROS topic
            self.publish_pushing_finished(True)
            print(f"[{self.name}] DEBUG: Published pushing_finished = True via ROS topic (robot reached target)")
            return py_trees.common.Status.SUCCESS
        elif robot_in_relay_range:
            print(f"[{self.name}] SUCCESS: Robot is in relay range (distance <= {self.distance_threshold}m)")
            self.pushing_active = False
            # Publish pushing_finished = True via ROS topic
            self.publish_pushing_finished(True)
            print(f"[{self.name}] DEBUG: Published pushing_finished = True via ROS topic (robot in relay range)")
            return py_trees.common.Status.SUCCESS
        
        # Always return RUNNING if parcel hasn't reached relay point yet
        # No FAILURE conditions - keep trying until success
        elapsed = time.time() - self.start_time
        
        # Update feedback with distance information
        with self.state_lock:
            if self.parcel_pose and self.relay_pose:
                distance = self._calculate_distance(self.parcel_pose, self.relay_pose)
                trajectory_status = "trajectory complete" if (self.ref_trajectory and self.trajectory_index >= len(self.ref_trajectory) - 1) else f"trajectory index: {self.trajectory_index}"
                self.feedback_message = f"[{self.robot_namespace}] Pushing object... {elapsed:.1f}s elapsed, {trajectory_status}, parcel-relay distance: {distance:.3f}m"
            else:
                # More detailed debug for missing pose data, but still try to calculate distance if possible
                robot_pose_status = "✓" if (self.current_state is not None and np.any(self.current_state[:3] != 0)) else "✗"
                parcel_pose_status = "✓" if self.parcel_pose is not None else "✗"
                relay_pose_status = "✓" if self.relay_pose is not None else "✗"
                
                # Try to calculate distance even if one pose is missing for better feedback
                distance_info = ""
                if self.parcel_pose and self.relay_pose:
                    distance = self._calculate_distance(self.parcel_pose, self.relay_pose)
                    distance_info = f", distance: {distance:.3f}m"
                elif self.parcel_pose or self.relay_pose:
                    distance_info = ", distance: N/A (missing pose)"
                else:
                    distance_info = ", distance: N/A (no poses)"
                
                self.feedback_message = f"[{self.robot_namespace}] Pushing object... {elapsed:.1f}s elapsed, waiting for pose data [robot:{robot_pose_status}, parcel:{parcel_pose_status}, relay:{relay_pose_status}]{distance_info}"
            
        # Even if trajectory is complete, we continue running until the parcel reaches the relay point
        # The control_timer_callback will maintain a minimal pushing force in the direction of the relay
        if self.ref_trajectory and self.trajectory_index >= len(self.ref_trajectory) - 1:
            # Check if we have pose data to give more detailed feedback
            if self.parcel_pose and self.relay_pose:
                distance = self._calculate_distance(self.parcel_pose, self.relay_pose)
                relay_number = self._extract_namespace_number()+1  # Relaypoint number
                print(f"[{self.name}] RUNNING: Trajectory complete, continuing to push parcel{self.current_parcel_index} toward relaypoint{relay_number}. Distance: {distance:.3f}m")
            else:
                relay_number = self._extract_namespace_number()+1  # Relaypoint number
                print(f"[{self.name}] RUNNING: Trajectory complete, continuing to push parcel{self.current_parcel_index} toward relaypoint{relay_number}")
        
        # Timeout check (adjust based on expected trajectory duration)
        if elapsed >= 120.0: 
            from .tree_builder import report_node_failure
            error_msg = f"PushObject timeout after {elapsed:.1f}s - trajectory execution failed"
            report_node_failure(self.name, error_msg, self.robot_namespace)
            print(f"[{self.name}] FAILURE: Push operation timed out after {elapsed:.1f} seconds")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates - with thread-safe cleanup"""
        print(f"[{self.name}] 开始终止推送行为，状态: {new_status}")
        
        # Step 1: 设置终止标志（线程安全）
        self.terminating = True
        self.pushing_active = False
        
        # Step 2: Publish final pushing_finished state based on termination status
        if new_status == py_trees.common.Status.SUCCESS:
            # If terminating with success, ensure pushing_finished = True is published
            self.publish_pushing_finished(True)
        else:
            # If terminating with failure or being interrupted, publish pushing_finished = False
            self.publish_pushing_finished(False)
        
        # Step 3: Stop the robot immediately
        try:
            if self.cmd_vel_pub and hasattr(self, 'node') and self.node:
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
                print(f"[{self.name}] 机器人已停止 [发布停止命令]")
        except Exception as e:
            print(f"[{self.name}] 警告: 停止机器人时出错: {e}")
        
        # Step 4: 等待控制线程安全退出
        # Step 4: 等待控制线程安全退出
        if hasattr(self, 'control_timer') and self.control_timer is not None:
            try:
                # 停止定时器
                self.node.destroy_timer(self.control_timer)
                self.control_timer = None
                print(f"[{self.name}] 控制定时器已销毁")
            except Exception as e:
                print(f"[{self.name}] 警告: 销毁定时器时出错: {e}")
                self.control_timer = None
        
        # Step 5: 线程安全的订阅清理
        with self.state_lock:
            subscription_errors = []
            
            # 清理机器人姿态订阅
            if hasattr(self, 'robot_pose_sub') and self.robot_pose_sub is not None:
                try:
                    self.node.destroy_subscription(self.robot_pose_sub)
                    self.robot_pose_sub = None
                except Exception as e:
                    subscription_errors.append(f"robot_pose_sub: {e}")
            
            # 清理包裹姿态订阅
            if hasattr(self, 'parcel_pose_sub') and self.parcel_pose_sub is not None:
                try:
                    self.node.destroy_subscription(self.parcel_pose_sub)
                    self.parcel_pose_sub = None
                except Exception as e:
                    subscription_errors.append(f"parcel_pose_sub: {e}")
            
            # 清理中继点姿态订阅（可能已经被自动销毁）
            if hasattr(self, 'relay_pose_sub') and self.relay_pose_sub is not None:
                try:
                    self.node.destroy_subscription(self.relay_pose_sub)
                    self.relay_pose_sub = None
                    print(f"[{self.name}] 中继点订阅已清理")
                except Exception as e:
                    subscription_errors.append(f"relay_pose_sub: {e}")
            else:
                # 中继点订阅可能已经在读取静态数据后被自动销毁
                print(f"[{self.name}] 中继点订阅已经销毁（正常情况）")
            
            if subscription_errors:
                print(f"[{self.name}] 订阅清理警告: {subscription_errors}")
        
        print(f"[{self.name}] 推送行为终止完成，状态: {new_status}")
    
    def _find_closest_reference_point(self, current_error=None, search_range=None):
        """
        Find the closest reference point in the trajectory to the current robot position
        Optimized for speed with intelligent search range selection
        
        Args:
            current_error: Current position error (optional, for optimization)
            search_range: Range to search in (optional, for optimization)
        """
        if not self.ref_trajectory or self.current_state is None:
            return self.trajectory_index, float('inf')
        
        curr_pos = np.array([self.current_state[0], self.current_state[1]])
        min_dist = float('inf')
        best_idx = self.trajectory_index
        
        # Optimized search range selection for speed
        if search_range is not None:
            # Use provided search range
            start_idx = max(0, self.trajectory_index)
            end_idx = min(len(self.ref_trajectory), start_idx + search_range)
            search_scope = "limited"
        elif current_error is not None and current_error < 0.03:
            # If error is very small, only search next 10 points for maximum efficiency
            start_idx = max(0, self.trajectory_index)
            end_idx = min(len(self.ref_trajectory), start_idx + 10)
            search_scope = "micro"
        elif current_error is not None and current_error < 0.1:
            # If error is small, only search next 15 points for efficiency
            start_idx = max(0, self.trajectory_index - 5)
            end_idx = min(len(self.ref_trajectory), self.trajectory_index + 15)
            search_scope = "local"
        else:
            # Search broader range but still limited for performance
            start_idx = max(0, self.trajectory_index - 10)
            end_idx = min(len(self.ref_trajectory), start_idx + 50)
            search_scope = "regional"
        
        # Vectorized distance calculation for speed (if range is small)
        search_range_size = end_idx - start_idx
        if search_range_size <= 30:
            # Use vectorized calculation for small ranges
            ref_positions = np.array([[self.ref_trajectory[i][0], self.ref_trajectory[i][1]] 
                                    for i in range(start_idx, end_idx)])
            if len(ref_positions) > 0:
                distances = np.linalg.norm(ref_positions - curr_pos, axis=1)
                min_idx = np.argmin(distances)
                best_idx = start_idx + min_idx
                min_dist = distances[min_idx]
        else:
            # Use loop for larger ranges (fallback)
            for idx in range(start_idx, end_idx):
                ref_pos = np.array([self.ref_trajectory[idx][0], self.ref_trajectory[idx][1]])
                dist = np.linalg.norm(curr_pos - ref_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
        
        # Reduced debug output for performance
        if hasattr(self, '_debug_search_counter'):
            self._debug_search_counter += 1
        else:
            self._debug_search_counter = 1
            
        # Only print debug info every 20 searches to reduce overhead
        if self._debug_search_counter % 20 == 0:
            print(f"[{self.name}] Search #{self._debug_search_counter} ({search_scope}): range [{start_idx}:{end_idx}], found idx {best_idx}, dist: {min_dist:.3f}m")
        
        return best_idx, min_dist
    
    def _simple_pi_control(self, current_state, ref_array):
        """Simple PI controller fallback when MPC solver fails"""
        try:
            # CRITICAL: Validate pose data before PI control
            if not self._has_valid_pose_data():
                print(f"[{self.name}] PI CONTROL SKIPPED: Missing pose data - returning zero control")
                return np.zeros((2, 1))  # Return zero control command
            
            # Get current target from reference trajectory (first point in reference)
            target_x = ref_array[0, 0]
            target_y = ref_array[1, 0]
            target_theta = ref_array[2, 0]
            target_v = ref_array[3, 0]
            
            # Current robot state
            robot_x = current_state[0]
            robot_y = current_state[1]
            robot_theta = current_state[2]
            
            # Position error
            error_x = target_x - robot_x
            error_y = target_y - robot_y
            position_error = np.sqrt(error_x**2 + error_y**2)
            
            # Angular error (normalize to [-pi, pi])
            angular_error = target_theta - robot_theta
            while angular_error > np.pi:
                angular_error -= 2*np.pi
            while angular_error < -np.pi:
                angular_error += 2*np.pi
            
            # PI control gains (conservative for stability)
            kp_linear = 0.8  # Proportional gain for linear velocity
            ki_linear = 0.1  # Integral gain for linear velocity
            kp_angular = 1.2  # Proportional gain for angular velocity
            ki_angular = 0.1  # Integral gain for angular velocity
            
            # Initialize integral terms if not exists
            if not hasattr(self, '_pi_integral_linear'):
                self._pi_integral_linear = 0.0
                self._pi_integral_angular = 0.0
                self._pi_last_time = time.time()
            
            # Calculate dt for integral term
            current_time = time.time()
            dt = current_time - self._pi_last_time
            dt = max(dt, 0.01)  # Prevent division by zero
            self._pi_last_time = current_time
            
            # Update integral terms with windup protection
            self._pi_integral_linear += position_error * dt
            self._pi_integral_angular += angular_error * dt
            
            # Anti-windup: limit integral terms
            max_integral = 1.0
            self._pi_integral_linear = np.clip(self._pi_integral_linear, -max_integral, max_integral)
            self._pi_integral_angular = np.clip(self._pi_integral_angular, -max_integral, max_integral)
            
            # Calculate control outputs
            # Transform error to robot frame for better control
            error_robot_frame_x = error_x * np.cos(robot_theta) + error_y * np.sin(robot_theta)
            error_robot_frame_y = -error_x * np.sin(robot_theta) + error_y * np.cos(robot_theta)
            
            # Linear velocity control (in robot frame)
            v_command = kp_linear * error_robot_frame_x + ki_linear * self._pi_integral_linear
            
            # Angular velocity control 
            omega_command = kp_angular * angular_error + ki_angular * self._pi_integral_angular
            
            # Apply velocity limits for safety
            max_linear_vel = 0.15  # Conservative max linear velocity
            max_angular_vel = 0.8  # Conservative max angular velocity
            
            v_command = np.clip(v_command, -max_linear_vel, max_linear_vel)
            omega_command = np.clip(omega_command, -max_angular_vel, max_angular_vel)
            
            # Apply additional safety: slow down when position error is large
            if position_error > 0.1:  # If more than 10cm away
                scale_factor = 0.5  # Reduce speed
                v_command *= scale_factor
                omega_command *= scale_factor
            
            print(f"[{self.name}] PI控制: pos_err={position_error:.3f}m, ang_err={np.degrees(angular_error):.1f}°, v={v_command:.3f}, ω={omega_command:.3f}")
            
            return np.array([[v_command], [omega_command]])
            
        except Exception as e:
            print(f"[{self.name}] Error in PI control: {e}")
            # Emergency fallback: stop the robot
            return np.array([[0.0], [0.0]])
    
    def _reset_pi_controller(self):
        """Reset PI controller integral terms to prevent windup"""
        self._pi_integral_linear = 0.0
        self._pi_integral_angular = 0.0
        self._pi_last_time = time.time()
        print(f"[{self.name}] PI控制器积分项已重置")

    def _advance_trajectory_index(self, best_idx=None):
        """Advance trajectory index with progress tracking - FORWARD ONLY"""
        old_trajectory_index = self.trajectory_index
        
        if self.trajectory_index < len(self.ref_trajectory) - 1:
            if best_idx is not None:
                # Use the closest index found, but ENSURE FORWARD PROGRESS ONLY
                # Never allow trajectory index to decrease
                new_index = max(self.trajectory_index + 1, best_idx)
                safe_new_index = max(self.trajectory_index, new_index)  # Double-check: never go backwards
                self._safe_update_trajectory_index(safe_new_index, "_advance_trajectory_index")
            else:
                # Normal forward progress
                self._safe_update_trajectory_index(self.trajectory_index + 1, "_advance_trajectory_index_normal")
        else:
            # At end of trajectory, keep using last state as target
            print(f"[{self.name}] At end of trajectory, maintaining last state as target (index: {self.trajectory_index})")
        
        # Update pushing estimated time after trajectory index change
        self._update_pushing_estimated_time()

    def _safe_update_trajectory_index(self, new_index, context="unknown"):
        """Safely update trajectory index ensuring forward-only progress"""
        old_index = self.trajectory_index
        
        # CRITICAL: Never allow trajectory index to decrease
        if new_index < self.trajectory_index:
            print(f"[{self.name}] WARNING: Attempted to decrease trajectory index from {old_index} to {new_index} in context '{context}' - BLOCKED")
            return False  # Index not updated
        
        # Allow forward progress or staying at same index
        self.trajectory_index = new_index
        
        # 🔧 DEBUG: Log all trajectory index changes to identify abnormal updates
        if new_index != old_index:
            current_time = time.time()
            if not hasattr(self, '_last_trajectory_update_time'):
                self._last_trajectory_update_time = current_time
                self._trajectory_update_count = 0
            
            time_since_last = current_time - self._last_trajectory_update_time
            self._trajectory_update_count += 1
            
            # Log trajectory index changes with timing information
            if time_since_last < 0.2:  # If updates are happening faster than 5Hz
                print(f"[{self.name}] ⚠️ FAST trajectory index update: {old_index} -> {new_index} in '{context}' (Δt={time_since_last:.3f}s)")
            else:
                print(f"[{self.name}] ✅ Normal trajectory index update: {old_index} -> {new_index} in '{context}' (Δt={time_since_last:.3f}s)")
            
            self._last_trajectory_update_time = current_time
            
            return True  # Index updated
        else:
            return False  # Index unchanged

    def _check_and_advance_trajectory_based_on_progress(self):
        """Advance trajectory index only when robot makes actual spatial progress"""
        # 🔧 NOTE: This method is deprecated in favor of explicit trajectory management
        # in MPC success/failure logic. Trajectory advancement is now handled by:
        # - MPC success: advance by N_c steps in control_loop()
        # - MPC failure: advance by 1 step in _handle_mpc_failure()
        print(f"[{self.name}] ⚠️ DEPRECATED: _check_and_advance_trajectory_based_on_progress() called")
        print(f"[{self.name}] 📝 轨迹索引现在由MPC成功/失败逻辑明确管理")
        return
    
    def _has_valid_pose_data(self):
        """检查是否有有效的姿态数据用于控制操作 - 修复属性初始化问题"""
        current_time = time.time()
        
        # 确保计数器存在
        if not hasattr(self, '_validity_check_count'):
            self._validity_check_count = 0
        
        # 检查机器人状态数据 - 基础有效性，增强调试
        robot_state_exists = self.current_state is not None
        robot_callback_count_ok = hasattr(self, '_robot_callback_count') and self._robot_callback_count > 0
        robot_not_zero = True
        
        if robot_state_exists:
            robot_position_zero = np.allclose(self.current_state[:3], [0.0, 0.0, 0.0])
            robot_not_zero = not robot_position_zero
            
            # 详细调试机器人状态
            if self._validity_check_count % 10 == 0:
                print(f"   机器人状态详情: pos=({self.current_state[0]:.3f}, {self.current_state[1]:.3f}, {self.current_state[2]:.3f})")
                print(f"   位置是否为零: {robot_position_zero}")
        
        robot_data_basic = robot_state_exists and robot_callback_count_ok and robot_not_zero
        
        # 检查包裹姿态数据 - 基础有效性
        parcel_data_basic = self.parcel_pose is not None
        
        # 检查数据时效性（如果数据超过3秒没有更新，认为无效）
        robot_data_fresh = True
        parcel_data_fresh = True
        
        # 增加时效性检查的容忍度，从2秒改为3秒，因为Gazebo仿真可能有延迟
        freshness_timeout = 3.0
        
        # 只有当时间戳存在时才进行时效性检查
        if hasattr(self, '_last_robot_callback_time') and self._last_robot_callback_time is not None:
            robot_data_age = current_time - self._last_robot_callback_time
            robot_data_fresh = robot_data_age < freshness_timeout
            if not robot_data_fresh and self._validity_check_count % 10 == 0:
                print(f"   机器人数据过期: {robot_data_age:.2f}s > {freshness_timeout}s")
        
        if hasattr(self, '_last_parcel_callback_time') and self._last_parcel_callback_time is not None:
            parcel_data_age = current_time - self._last_parcel_callback_time
            parcel_data_fresh = parcel_data_age < freshness_timeout
            if not parcel_data_fresh and self._validity_check_count % 10 == 0:
                print(f"   包裹数据过期: {parcel_data_age:.2f}s > {freshness_timeout}s")
        
        # 合并基础有效性和时效性检查
        robot_data_valid = robot_data_basic and robot_data_fresh
        parcel_data_valid = parcel_data_basic and parcel_data_fresh
        
        # 递增计数器（已在上面初始化）
        self._validity_check_count += 1
        
        # 更频繁的调试输出来诊断问题
        if self._validity_check_count % 10 == 0:  # 每10次检查输出一次详细信息
            print(f"[{self.name}] � 数据有效性诊断 #{self._validity_check_count}:")
            print(f"   机器人数据: 基础={robot_data_basic}, 时效={robot_data_fresh}, 最终={robot_data_valid}")
            print(f"   包裹数据: 基础={parcel_data_basic}, 时效={parcel_data_fresh}, 最终={parcel_data_valid}")
            print(f"   机器人状态非零: {self.current_state is not None and not np.allclose(self.current_state[:3], [0.0, 0.0, 0.0]) if self.current_state is not None else False}")
            print(f"   回调次数: robot={getattr(self, '_robot_callback_count', 0)}, parcel={getattr(self, '_parcel_callback_count', 0)}")
            
            # 添加时间戳调试
            if hasattr(self, '_last_robot_callback_time'):
                robot_age = current_time - self._last_robot_callback_time
                print(f"   机器人数据年龄: {robot_age:.2f}s")
            else:
                print(f"   机器人时间戳: 未设置")
                
            if hasattr(self, '_last_parcel_callback_time'):
                parcel_age = current_time - self._last_parcel_callback_time
                print(f"   包裹数据年龄: {parcel_age:.2f}s")
            else:
                print(f"   包裹时间戳: 未设置")
        
        final_result = robot_data_valid and parcel_data_valid
        
        # 如果数据应该有效但结果为False，输出详细的失败原因
        if not final_result and (robot_data_basic and parcel_data_basic):
            print(f"[{self.name}] ❌ 数据有效性检查失败详情:")
            print(f"   机器人数据问题: 基础OK={robot_data_basic}, 时效OK={robot_data_fresh}")
            print(f"   包裹数据问题: 基础OK={parcel_data_basic}, 时效OK={parcel_data_fresh}")
        
        return final_result
    
    def _log_pose_data_status(self, message):
        """Log current pose data status for debugging"""
        robot_status = "✓" if (self.current_state is not None and 
                              hasattr(self, '_robot_callback_count') and 
                              self._robot_callback_count > 0) else "✗"
        parcel_status = "✓" if self.parcel_pose is not None else "✗"
        relay_status = "✓" if self.relay_pose is not None else "✗"
        
        print(f"[{self.name}] {message}: robot={robot_status}, parcel={parcel_status}, relay={relay_status}")
    
    def _handle_mpc_failure(self, current_state, ref_array):
        """Handle MPC failure by using PI control fallback and advancing trajectory by 1 step"""
        try:
            print(f"[{self.name}] 🔄 MPC失败，切换到PI控制...")
            
            # Use PI control for single step
            pi_command = self._simple_pi_control(current_state, ref_array)
            
            if pi_command is not None and not np.isnan(pi_command).any():
                # Apply PI control command
                cmd_msg = Twist()
                cmd_msg.linear.x = float(pi_command[0, 0])
                cmd_msg.angular.z = float(pi_command[1, 0])
                self.cmd_vel_pub.publish(cmd_msg)
                
                print(f"[{self.name}] 🎯 PI控制命令已发布: v={cmd_msg.linear.x:.3f}, ω={cmd_msg.angular.z:.3f}")
                
                # Advance trajectory index by only 1 step on MPC failure
                old_idx = self.trajectory_index
                new_idx = min(self.trajectory_index + 1, len(self.ref_trajectory) - 1)
                self._safe_update_trajectory_index(new_idx, "mpc_failure_pi_advance")
                print(f"[{self.name}] 📈 MPC失败后轨迹索引前进1步: {old_idx} -> {self.trajectory_index}")
                
                # Clear control sequence so MPC will be called again next time
                self.control_sequence = None
                self.control_step = 0
                
            else:
                print(f"[{self.name}] ❌ PI控制也失败，停止机器人")
                # Emergency stop
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
                
        except Exception as e:
            print(f"[{self.name}] Error in MPC failure handler: {e}")
            # Emergency stop
            if self.cmd_vel_pub:
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)

    def _create_reliable_qos_profile(self):
        """创建可靠的QoS配置文件以减少数据丢失"""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # 确保消息可靠传输
            history=HistoryPolicy.KEEP_LAST,         # 保持最新的N个消息
            depth=5,                                 # 保持最新5个消息
            durability=DurabilityPolicy.VOLATILE     # 不持久化（性能优化）
        )
        return qos_profile
    
    def _create_best_effort_qos_profile(self):
        """创建最佳努力QoS配置文件用于高频数据"""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # 高频数据使用最佳努力
            history=HistoryPolicy.KEEP_LAST,            # 保持最新消息
            depth=1,                                    # 只保持最新1个消息
            durability=DurabilityPolicy.VOLATILE        # 不持久化
        )
        return qos_profile

    def _init_data_cache(self):
        """初始化数据缓存机制以提高数据可靠性"""
        self.robot_pose_cache = []
        self.parcel_pose_cache = []
        self.cache_size = 5  # 保持最新5个数据点
        self.cache_lock = threading.Lock()
        
    def _add_to_robot_cache(self, pose_data):
        """添加机器人姿态数据到缓存"""
        with self.cache_lock:
            self.robot_pose_cache.append({
                'data': pose_data.copy(),
                'timestamp': time.time()
            })
            if len(self.robot_pose_cache) > self.cache_size:
                self.robot_pose_cache.pop(0)
    
    def _add_to_parcel_cache(self, pose_data):
        """添加包裹姿态数据到缓存"""
        with self.cache_lock:
            self.parcel_pose_cache.append({
                'data': pose_data,
                'timestamp': time.time()
            })
            if len(self.parcel_pose_cache) > self.cache_size:
                self.parcel_pose_cache.pop(0)
    
    def _get_latest_valid_robot_pose(self):
        """获取最新有效的机器人姿态数据"""
        current_time = time.time()
        with self.cache_lock:
            for cached_data in reversed(self.robot_pose_cache):
                if current_time - cached_data['timestamp'] < 1.0:  # 1秒内的数据
                    return cached_data['data']
        return self.current_state
    
    def _get_latest_valid_parcel_pose(self):
        """获取最新有效的包裹姿态数据"""
        current_time = time.time()
        with self.cache_lock:
            for cached_data in reversed(self.parcel_pose_cache):
                if current_time - cached_data['timestamp'] < 1.0:  # 1秒内的数据
                    return cached_data['data']
        return self.parcel_pose




