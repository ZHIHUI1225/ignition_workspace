#!/usr/bin/env python3
"""
Mobile Robot MPC Controller
Extracted from manipulation_behaviors.py for standalone testing
"""

import numpy as np
import casadi as ca
import os
import yaml


class MobileRobotMPC:
    def __init__(self):
        # Load configuration
        self.config = self.load_config()
        
        # Get dt from config to match trajectory generation frequency
        self.dt = self.config.get('planning', {}).get('discrete_dt', 0.1)
        
        # Simple MPC parameters
        self.N = 5           # Shorter prediction horizon for speed
        
        # Log the dt being used
        print(f"MPC Controller initialized with dt = {self.dt}s (frequency = {1.0/self.dt:.1f}Hz)")
        
        # Optimized weights for better tracking without increasing angular sensitivity
        self.Q = np.diag([200.0, 200.0, 3.0, 0.2, 0.2])  # Higher position weight for better tracking
        self.R = np.diag([0.05, 0.3])                     # Lower control costs for more responsive control
        
        # Load velocity limits from config for easy tuning
        mpc_config = self.config.get('mpc_controller', {})
        self.max_vel = mpc_config.get('max_linear_velocity', 0.08)     # m/s - default 8 cm/s
        self.min_vel = mpc_config.get('min_linear_velocity', -0.03)    # m/s - default 3 cm/s  
        self.max_omega = mpc_config.get('max_angular_velocity', 0.6)   # rad/s - default 34 deg/s
        self.min_omega = mpc_config.get('min_angular_velocity', -0.6)  # rad/s
        
        # Also load prediction horizon from config
        self.N = mpc_config.get('prediction_horizon', 5)
        
        print(f"MPC Velocity Limits: linear=[{self.min_vel:.3f}, {self.max_vel:.3f}] m/s, angular=[{self.min_omega:.1f}, {self.max_omega:.1f}] rad/s")
        print(f"MPC Prediction Horizon: {self.N} steps (fixed)")
        
        # MPC state for warm starting
        self.last_solution_U = None
        self.last_solution_X = None
        
        # System dimensions
        self.nx = 5   # Number of states (x, y, theta, v, omega)
        self.nu = 2   # Number of controls (v, omega)
        
        # Reference trajectory
        self.ref_traj = None
        
        # PI controller parameters (fallback) - load from config or use defaults
        pi_config = mpc_config.get('pi_controller', {})
        self.kp_pos = pi_config.get('kp_position', 0.5)      # Position proportional gain 
        self.ki_pos = pi_config.get('ki_position', 0.02)     # Position integral gain 
        self.kp_angle = pi_config.get('kp_angle', 0.6)       # Angle proportional gain - reduced for smoother motion
        self.ki_angle = pi_config.get('ki_angle', 0.01)      # Angle integral gain - reduced for smoother motion
        
        # PI controller state
        self.pos_error_integral = 0.0
        self.angle_error_integral = 0.0
        self.max_integral = 0.5  # Reduced anti-windup limit
        
        # PI control thresholds - increased angle threshold to be less sensitive
        self.angle_threshold = 0.8     # Max angle error before pure rotation (increased from 0.5)
        self.distance_threshold = 0.05  # Distance threshold for position vs orientation control
        
        # Initialize MPC with fixed horizon
        self.setup_mpc()

    def setup_mpc(self):
        """Setup MPC optimization problem with fixed horizon"""        
        # Optimization problem
        self.opti = ca.Opti()
        
        # Decision variables (use fixed horizon)
        self.X = self.opti.variable(self.nx, self.N+1)  # State trajectory
        self.U = self.opti.variable(self.nu, self.N)    # Control trajectory
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)          # Initial state
        self.ref = self.opti.parameter(self.nx, self.N+1)  # Reference trajectory

        # Simple cost function with angle wrapping
        cost = 0
        
        for k in range(self.N):
            # Position tracking cost (x, y)
            pos_error = self.X[:2, k] - self.ref[:2, k]
            cost += ca.mtimes([pos_error.T, self.Q[:2, :2], pos_error])
            
            # Angle tracking cost with wrapping - use sin/cos representation
            theta_ref = self.ref[2, k]
            theta_current = self.X[2, k]
            angle_error = ca.atan2(ca.sin(theta_current - theta_ref), ca.cos(theta_current - theta_ref))
            cost += self.Q[2, 2] * angle_error**2
            
            # Velocity tracking cost
            vel_error = self.X[3:, k] - self.ref[3:, k]
            cost += ca.mtimes([vel_error.T, self.Q[3:, 3:], vel_error])
            
            # Control cost - reduced to allow more aggressive control
            control_cost = ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
            cost += control_cost
            
            # Small progress incentive - reward forward motion towards target (reduced to prevent conflicts)
            dx_to_target = self.ref[0, k] - self.X[0, k]
            dy_to_target = self.ref[1, k] - self.X[1, k]
            forward_progress = self.U[0, k] * (ca.cos(self.X[2, k]) * dx_to_target + ca.sin(self.X[2, k]) * dy_to_target)
            cost -= 0.01 * forward_progress  # Much smaller reward to prevent circular motion
            
            # System dynamics constraint
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost with angle wrapping - optimized for better final tracking
        pos_error_f = self.X[:2, -1] - self.ref[:2, -1]
        cost += 20.0 * ca.mtimes([pos_error_f.T, self.Q[:2, :2], pos_error_f])  # Higher terminal position cost for better tracking
        
        theta_ref_f = self.ref[2, -1]
        theta_current_f = self.X[2, -1]
        angle_error_f = ca.atan2(ca.sin(theta_current_f - theta_ref_f), ca.cos(theta_current_f - theta_ref_f))
        cost += 2.0 * self.Q[2, 2] * angle_error_f**2  # Slightly higher terminal angle cost for better orientation tracking
        
        vel_error_f = self.X[3:, -1] - self.ref[3:, -1]
        cost += ca.mtimes([vel_error_f.T, self.Q[3:, 3:], vel_error_f])

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Input constraints
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))
        
        # Angular velocity rate constraint to prevent abrupt rotation changes
        for k in range(self.N-1):
            omega_rate = (self.U[1, k+1] - self.U[1, k]) / self.dt
            self.opti.subject_to(self.opti.bounded(-3.0, omega_rate, 3.0))  # Limit angular acceleration

        # Solver settings - optimized for speed and accuracy balance
        self.opti.minimize(cost)
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 30,               # Reduced iterations for faster solving
            'ipopt.max_cpu_time': 0.15,         # Tighter time constraint for real-time performance
            'ipopt.tol': 5e-3,                  # Slightly relaxed tolerance for speed
            'ipopt.acceptable_tol': 1e-2,       # Acceptable tolerance for early termination
            'ipopt.acceptable_iter': 5,         # Early termination after 5 acceptable iterations
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.mu_strategy': 'adaptive',    # Adaptive barrier parameter for faster convergence
            'ipopt.warm_start_init_point': 'yes' # Enable warm starting for faster subsequent solves
        }
        
        self.opti.solver('ipopt', opts)

    def robot_model(self, x, u):
        """System dynamics for 5-state model: x_next = f(x, u)"""
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,
            x[1] + u[0] * ca.sin(x[2]) * self.dt,
            x[2] + u[1] * self.dt,
            u[0],
            u[1]
        )
        

    def set_reference_trajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def update(self, current_state):
        """Update MPC with current state and return control commands"""
        # Validate state
        current_state = np.array(current_state[:5], dtype=np.float64)
        
        if np.any(np.isnan(current_state)) or np.any(np.isinf(current_state)):
            print("Invalid current state, using PI control")
            return self.pi_control(current_state)
        
        if self.ref_traj is None:
            print("No reference trajectory set")
            return np.array([0.0, 0.0])
        
        # Try MPC first
        try:
            # Normalize angles in reference trajectory for better MPC performance
            ref_traj_normalized = self.ref_traj.copy()
            for i in range(ref_traj_normalized.shape[1]):
                ref_traj_normalized[2, i] = self._normalize_angle(ref_traj_normalized[2, i])
            
            # Debug: Check state and reference
            current_pos = current_state[:3]
            target_pos = ref_traj_normalized[:3, 0]
            pos_error = np.linalg.norm(target_pos[:2] - current_pos[:2])
            angle_error = self._normalize_angle(target_pos[2] - current_pos[2])
            
            print(f"MPC Debug: pos_err={pos_error:.3f}m, angle_err={np.degrees(angle_error):.1f}°")
            
            # Set problem parameters
            self.opti.set_value(self.x0, current_state)
            self.opti.set_value(self.ref, ref_traj_normalized)
            
            # Warm start from previous solution if available
            if self.last_solution_U is not None and self.last_solution_X is not None:
                try:
                    # Shift previous solution and extend with last control input
                    u_init = np.hstack([self.last_solution_U[:, 1:], self.last_solution_U[:, -1:]])
                    x_init = np.hstack([self.last_solution_X[:, 1:], self.last_solution_X[:, -1:]])
                    
                    self.opti.set_initial(self.U, u_init)
                    self.opti.set_initial(self.X, x_init)
                except:
                    pass  # If warm start fails, solver will use default initialization
            
            # Solve
            sol = self.opti.solve()
            u_optimal = sol.value(self.U)
            x_optimal = sol.value(self.X)
            
            # Store solution for next warm start
            self.last_solution_U = u_optimal
            self.last_solution_X = x_optimal
            
            # Reset PI integral terms on successful MPC solve
            self.pos_error_integral = 0.0
            self.angle_error_integral = 0.0
            
            print(f"MPC Success: v={u_optimal[0,0]:.3f}, ω={u_optimal[1,0]:.3f}")
            return u_optimal[:, 0]
            
        except Exception as e:
            print(f"MPC failed: {type(e).__name__}. Switching to PI control.")
            return self.pi_control(current_state)
    
    def pi_control(self, current_state):
        """Improved PI controller fallback when MPC fails"""
        if self.ref_traj is None:
            return np.array([0.0, 0.0])
        
        # Get current target (first point in reference trajectory)
        target_x = self.ref_traj[0, 0]
        target_y = self.ref_traj[1, 0]
        target_theta = self.ref_traj[2, 0]
        
        # Position error
        dx = target_x - current_state[0]
        dy = target_y - current_state[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate desired movement direction
        if distance > self.distance_threshold:
            desired_angle = np.arctan2(dy, dx)
            # Use movement direction for control
            angle_error = self._normalize_angle(desired_angle - current_state[2])
            
            # Update position integral (only when moving)
            self.pos_error_integral += distance * self.dt
            self.pos_error_integral = np.clip(self.pos_error_integral, -self.max_integral, self.max_integral)
            
            # Movement-based control with smoother angular response
            if abs(angle_error) > self.angle_threshold:
                # Large angle error: rotate towards target with minimal forward motion
                v_cmd = 0.005  # Very small forward velocity to prevent getting stuck
                omega_cmd = np.sign(angle_error) * min(abs(self.kp_angle * angle_error), self.max_omega * 0.5)  # Limit to 50% of max
                # Don't use integral for large angle errors to prevent windup
                self.angle_error_integral = 0.0
            else:
                # Small angle error: move forward with gentle course correction
                v_cmd = self.kp_pos * distance + self.ki_pos * self.pos_error_integral
                v_cmd = max(v_cmd, 0.01)  # Ensure minimum forward velocity - reduced for smoother motion
                
                # Update angle integral for fine course correction
                self.angle_error_integral += angle_error * self.dt
                self.angle_error_integral = np.clip(self.angle_error_integral, -self.max_integral, self.max_integral)
                omega_cmd = 0.2 * self.kp_angle * angle_error + self.ki_angle * self.angle_error_integral  # Further reduced for smoothness
        else:
            # Close to target: pure orientation control
            angle_error = self._normalize_angle(target_theta - current_state[2])
            
            # Reset position integral when close
            self.pos_error_integral = 0.0
            
            # Pure orientation control with reduced sensitivity
            v_cmd = 0.0
            if abs(angle_error) > 0.3:  # Only rotate if significant angle error (increased for smoother motion)
                self.angle_error_integral += angle_error * self.dt
                self.angle_error_integral = np.clip(self.angle_error_integral, -self.max_integral, self.max_integral)
                omega_cmd = 0.4 * self.kp_angle * angle_error + self.ki_angle * self.angle_error_integral  # Further reduced gain
            else:
                omega_cmd = 0.0  # Stop rotation when close enough
                self.angle_error_integral = 0.0
        
        # Apply constraints with better limits
        v_cmd = np.clip(v_cmd, self.min_vel, self.max_vel)
        omega_cmd = np.clip(omega_cmd, self.min_omega, self.max_omega)
        
        # Debug output for understanding control behavior
        if distance > 0.01:
            print(f"PI Control: dist={distance:.3f}m, angle_err={np.degrees(angle_error):.1f}°, v={v_cmd:.3f}, ω={omega_cmd:.3f}")
        
        return np.array([v_cmd, omega_cmd])
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def get_predicted_trajectory(self):
        """Get predicted trajectory from last solution (simplified)"""
        return None  # Simplified - not used in PI fallback mode

    def load_config(self):
        """Load configuration from YAML file"""
        config_path = '/root/workspace/config/config.yaml'
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration")
            return {
                'planning': {
                    'discrete_dt': 0.1  # Default fallback
                }
            }
