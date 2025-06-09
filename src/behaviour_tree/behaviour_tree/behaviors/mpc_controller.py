#!/usr/bin/env python3
"""
General MPC Controller for Behavior Tree Actions
Provides unified trajectory following capabilities for PushObject and PickObject behaviors.
"""

import numpy as np
import casadi as ca
import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
import json


class GeneralMPCController:
    """
    General MPC controller that can be used by different behavior tree actions
    to follow reference trajectories with cmd_vel publishing.
    """
    
    def __init__(self, namespace="turtlebot0", case="simple_maze", controller_type="push"):
        self.namespace = namespace
        self.case = case
        self.controller_type = controller_type  # "push" or "pick"
        
        # MPC parameters - optimized for trajectory following
        self.N = 10           # Prediction horizon
        self.N_c = 3          # Control horizon
        self.dt = 0.1         # Time step
        
        # Tuned weights for precise trajectory tracking
        if controller_type == "pick":
            # More aggressive tracking for picking operations
            self.Q = np.diag([100.0, 100.0, 10.0])  # State weights (x, y, theta)
            self.R = np.diag([0.01, 0.01])          # Control input weights
            self.F = np.diag([200.0, 200.0, 20.0])  # Terminal cost weights
            self.max_vel = 0.2      # m/s
            self.max_omega = 1.0    # rad/s
        else:  # push controller
            # More stable tracking for pushing operations
            self.Q = np.diag([30.0, 30.0, 8.0])    # State weights
            self.R = np.diag([0.03, 0.01])         # Control input weights 
            self.F = np.diag([60.0, 60.0, 20.0])   # Terminal cost weights
            self.max_vel = 0.15     # m/s
            self.max_omega = np.pi/2 # rad/s
        
        self.min_vel = -0.2     # Allow reverse
        self.min_omega = -self.max_omega
        
        # System dimensions
        self.nx = 3   # States (x, y, theta)
        self.nu = 2   # Controls (v, omega)
        
        # State variables
        self.current_state = np.zeros(3)
        self.reference_trajectory = None
        self.trajectory_data = []
        self.target_pose = np.zeros(3)
        
        # Control variables
        self.control_active = False
        self.target_reached = False
        self.state_lock = threading.Lock()
        self.last_solution = None
        
        # Lookahead parameters for trajectory selection
        self.lookahead_distance = 0.3  # meters
        self.cross_track_gain = 0.5
        
        # Setup MPC
        self.setup_mpc()
        
    def setup_mpc(self):
        """Initialize the MPC optimization problem"""
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N+1)  # State trajectory
        self.U = self.opti.variable(self.nu, self.N)    # Control trajectory
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)          # Initial state
        self.ref = self.opti.parameter(self.nx, self.N+1)  # Reference trajectory
        
        # Cost function with progressive weighting
        cost = 0
        for k in range(self.N):
            # Progressive weighting (increases toward end of horizon)
            weight_factor = 1.0 + 2.0 * k / self.N
            
            # State tracking cost
            state_error = self.X[:, k] - self.ref[:, k]
            # Normalize angle difference
            angle_error = ca.fmod(state_error[2] + ca.pi, 2*ca.pi) - ca.pi
            state_error_normalized = ca.vertcat(state_error[:2], angle_error)
            cost += weight_factor * ca.mtimes([state_error_normalized.T, self.Q, state_error_normalized])
            
            # Control cost with smoothing
            cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
            
            # Control smoothing (penalize changes in control)
            if k > 0:
                control_diff = self.U[:, k] - self.U[:, k-1]
                cost += 0.1 * ca.mtimes([control_diff.T, control_diff])
        
        # Terminal cost with extra emphasis on final convergence
        terminal_error = self.X[:, -1] - self.ref[:, -1]
        # Position terminal error
        pos_term_error = terminal_error[:2]
        cost += 3.0 * ca.mtimes([pos_term_error.T, self.F[:2,:2], pos_term_error])
        
        # Terminal orientation error with proper angle normalization
        theta_term_error = ca.fmod(terminal_error[2] + ca.pi, 2*ca.pi) - ca.pi
        cost += 3.0 * self.F[2,2] * theta_term_error**2
        
        self.opti.minimize(cost)
        
        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # System dynamics constraints
        for k in range(self.N):
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)
        
        # Control constraints
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))
        
        # Solver settings optimized for real-time performance
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.limited_memory_max_history': 5,
            'ipopt.linear_solver': 'mumps'
        }
        self.opti.solver('ipopt', opts)
        
    def robot_model(self, x, u):
        """Robot dynamics model"""
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,
            x[1] + u[0] * ca.sin(x[2]) * self.dt,
            x[2] + u[1] * self.dt
        )
    
    def load_trajectory(self):
        """Load trajectory data from JSON file"""
        json_file_path = f'/root/workspace/data/{self.case}/{self.namespace}_Trajectory.json'
        
        try:
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
            
            if self.controller_type == "pick":
                # For picking, reverse the trajectory and add interpolation
                self.goal = data[0]
                self.target_pose[0] = self.goal[0] - 0.5 * np.cos(self.goal[2])
                self.target_pose[1] = self.goal[1] - 0.5 * np.sin(self.goal[2])
                self.target_pose[2] = self.goal[2]
                
                # Create interpolation points
                num_interp_points = 5
                interp_points = []
                for i in range(num_interp_points):
                    ratio = (i + 1) / (num_interp_points + 1)
                    interp_x = self.target_pose[0] + ratio * (self.goal[0] - self.target_pose[0])
                    interp_y = self.target_pose[1] + ratio * (self.goal[1] - self.target_pose[1])
                    interp_theta = self.target_pose[2] + ratio * (self.goal[2] - self.target_pose[2])
                    interp_points.append([interp_x, interp_y, interp_theta])
                
                # Reverse trajectory and add interpolation points
                self.trajectory_data = data[::-1] + interp_points
                
            else:  # push controller
                self.trajectory_data = data
                if self.trajectory_data:
                    self.target_pose = np.array(self.trajectory_data[-1])
                    
            print(f"[MPCController] Loaded {len(self.trajectory_data)} trajectory points for {self.controller_type}")
            return True
            
        except Exception as e:
            print(f"[MPCController] Error loading trajectory: {e}")
            return False
    
    def select_reference_point(self):
        """Select reference point using lookahead distance"""
        if not self.trajectory_data:
            return self.target_pose
        
        # Find the closest point on trajectory
        min_distance = float('inf')
        closest_index = 0
        
        for i, point in enumerate(self.trajectory_data):
            distance = np.sqrt((self.current_state[0] - point[0])**2 + 
                             (self.current_state[1] - point[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # Look ahead from closest point
        lookahead_index = closest_index
        cumulative_distance = 0.0
        
        for i in range(closest_index, len(self.trajectory_data) - 1):
            segment_distance = np.sqrt(
                (self.trajectory_data[i+1][0] - self.trajectory_data[i][0])**2 +
                (self.trajectory_data[i+1][1] - self.trajectory_data[i][1])**2
            )
            
            if cumulative_distance + segment_distance >= self.lookahead_distance:
                lookahead_index = i + 1
                break
            cumulative_distance += segment_distance
            lookahead_index = i + 1
        
        # Ensure we don't go beyond trajectory
        lookahead_index = min(lookahead_index, len(self.trajectory_data) - 1)
        
        return np.array(self.trajectory_data[lookahead_index])
    
    def generate_reference_trajectory(self):
        """Generate reference trajectory for MPC horizon"""
        if not self.trajectory_data:
            # Use target pose if no trajectory
            ref_traj = np.tile(self.target_pose.reshape(-1, 1), (1, self.N + 1))
            return ref_traj
        
        # Get current reference point
        current_ref = self.select_reference_point()
        
        # Create reference trajectory by extending from current reference
        ref_traj = np.zeros((self.nx, self.N + 1))
        
        # Fill with current reference point (can be improved with trajectory interpolation)
        for k in range(self.N + 1):
            ref_traj[:, k] = current_ref
            
        return ref_traj
    
    def compute_control(self):
        """Compute optimal control using MPC"""
        with self.state_lock:
            current_state_copy = self.current_state.copy()
        
        # Generate reference trajectory
        ref_traj = self.generate_reference_trajectory()
        
        # Set optimization parameters
        self.opti.set_value(self.x0, current_state_copy)
        self.opti.set_value(self.ref, ref_traj)
        
        # Warm start if available
        if self.last_solution is not None:
            try:
                # Shift previous solution for warm start
                X_warm = np.zeros((self.nx, self.N + 1))
                U_warm = np.zeros((self.nu, self.N))
                
                # Shift states
                X_warm[:, :-1] = self.last_solution['X'][:, 1:]
                X_warm[:, -1] = self.last_solution['X'][:, -1]
                
                # Shift controls
                U_warm[:, :-1] = self.last_solution['U'][:, 1:]
                U_warm[:, -1] = self.last_solution['U'][:, -1]
                
                self.opti.set_initial(self.X, X_warm)
                self.opti.set_initial(self.U, U_warm)
            except:
                pass
        
        try:
            # Solve optimization
            sol = self.opti.solve()
            
            # Extract optimal control
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)
            
            # Store solution for warm starting
            self.last_solution = {'U': u_opt, 'X': x_opt}
            
            # Return first control action
            return u_opt[:, 0]
            
        except Exception as e:
            print(f"[MPCController] MPC solve failed: {e}")
            
            # Fallback proportional controller
            return self.proportional_controller()
    
    def proportional_controller(self):
        """Fallback proportional controller"""
        # Get target point
        if self.trajectory_data:
            target = self.select_reference_point()
        else:
            target = self.target_pose
            
        # Position error
        dx = target[0] - self.current_state[0]
        dy = target[1] - self.current_state[1]
        
        # Distance and angle to target
        distance = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)
        
        # Angle error (normalized to [-pi, pi])
        angle_error = target_angle - self.current_state[2]
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # Proportional control
        if distance > 0.05:  # Move towards target
            v = min(0.3 * distance, self.max_vel)
            omega = 2.0 * angle_error
            omega = np.clip(omega, self.min_omega, self.max_omega)
        else:  # At target, just adjust orientation
            v = 0.0
            omega = 1.0 * angle_error
            omega = np.clip(omega, self.min_omega, self.max_omega)
        
        return np.array([v, omega])
    
    def update_robot_pose(self, pose_msg):
        """Update robot pose from odometry"""
        with self.state_lock:
            self.current_state[0] = pose_msg.pose.pose.position.x
            self.current_state[1] = pose_msg.pose.pose.position.y
            
            # Extract yaw from quaternion
            orientation = pose_msg.pose.pose.orientation
            siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
            cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
            self.current_state[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    def check_target_reached(self, tolerance_pos=0.08, tolerance_angle=0.2):
        """Check if target is reached"""
        if self.trajectory_data:
            target = np.array(self.trajectory_data[-1])  # Final point
        else:
            target = self.target_pose
            
        # Position error
        pos_error = np.sqrt((self.current_state[0] - target[0])**2 + 
                           (self.current_state[1] - target[1])**2)
        
        # Angle error
        angle_error = abs(self.current_state[2] - target[2])
        angle_error = min(angle_error, 2*np.pi - angle_error)  # Normalize
        
        return pos_error < tolerance_pos and angle_error < tolerance_angle
    
    def start_control(self):
        """Start the control process"""
        self.control_active = True
        self.target_reached = False
        
        # Load trajectory
        if not self.load_trajectory():
            print(f"[MPCController] Failed to load trajectory for {self.controller_type}")
            return False
            
        print(f"[MPCController] Started {self.controller_type} control for {self.namespace}")
        return True
    
    def stop_control(self):
        """Stop the control process"""
        self.control_active = False
        print(f"[MPCController] Stopped {self.controller_type} control")
    
    def get_control_command(self):
        """Get control command (called from behavior tree)"""
        if not self.control_active:
            return np.array([0.0, 0.0])
        
        # Check if target reached
        if self.check_target_reached():
            self.target_reached = True
            self.control_active = False
            return np.array([0.0, 0.0])
        
        # Compute and return optimal control
        return self.compute_control()
    
    def is_control_complete(self):
        """Check if control is complete"""
        return self.target_reached or not self.control_active


class MPCControllerNode(Node):
    """
    ROS2 Node wrapper for the MPC controller that publishes cmd_vel
    Can be used by behavior tree actions.
    """
    
    def __init__(self, namespace="turtlebot0", case="simple_maze", controller_type="push"):
        super().__init__(f'mpc_controller_{controller_type}_{namespace}')
        
        self.namespace = namespace
        self.controller_type = controller_type
        
        # Create MPC controller
        self.mpc_controller = GeneralMPCController(namespace, case, controller_type)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'/{namespace}/{controller_type}/cmd_vel', 10)
        
        # Subscribers  
        self.robot_pose_sub = self.create_subscription(
            Odometry, f'/{namespace}/{controller_type}/robot_pose', 
            self.robot_pose_callback, 10)
        
        # Control timer
        self.control_timer = None
        self.control_frequency = 10.0  # Hz
        
    def robot_pose_callback(self, msg):
        """Update robot pose in MPC controller"""
        self.mpc_controller.update_robot_pose(msg)
    
    def start_control(self):
        """Start MPC control and publishing"""
        if self.mpc_controller.start_control():
            # Start control timer
            self.control_timer = self.create_timer(
                1.0 / self.control_frequency, self.control_loop)
            self.get_logger().info(f'Started MPC control for {self.controller_type}')
            return True
        return False
    
    def stop_control(self):
        """Stop MPC control"""
        if self.control_timer:
            self.control_timer.cancel()
            self.control_timer = None
        
        self.mpc_controller.stop_control()
        
        # Publish zero velocity
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)
        
        self.get_logger().info(f'Stopped MPC control for {self.controller_type}')
    
    def control_loop(self):
        """Main control loop"""
        # Get control command from MPC
        control = self.mpc_controller.get_control_command()
        
        # Create and publish cmd_vel message
        cmd_vel = Twist()
        cmd_vel.linear.x = float(control[0])
        cmd_vel.angular.z = float(control[1])
        
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Check if control is complete
        if self.mpc_controller.is_control_complete():
            self.get_logger().info(f'{self.controller_type} control completed!')
            self.stop_control()
    
    def is_control_active(self):
        """Check if control is currently active"""
        return self.control_timer is not None
    
    def is_control_complete(self):
        """Check if control task is complete"""
        return self.mpc_controller.is_control_complete()
