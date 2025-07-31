#!/usr/bin/env python3
"""
Mobile Robot MPC Controller
Extracted from manipulation_behaviors.py for standalone testing
"""

import numpy as np
import casadi as ca
import os


class MobileRobotMPC:
    def __init__(self):
        # MPC parameters - optimized for position convergence
        self.N = 8           # Longer horizon for better convergence
        self.N_c = 3         # Longer control horizon for smoother control
        self.dt = 0.5        # Time step (increased from 0.1s to 0.5s)
        
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
            # State tracking cost
            state_error = self.X[:, k] - self.ref[:, k]
            cost += ca.mtimes([state_error.T, self.Q, state_error])
            
            # Control cost
            control_cost = ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
            cost += control_cost
            
            # System dynamics constraint
            x_next = self.robot_model_safe(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost - simplified
        pos_error_x_f = self.X[0, -1] - self.ref[0, -1]
        pos_error_y_f = self.X[1, -1] - self.ref[1, -1]
        theta_diff_f = self.X[2, -1] - self.ref[2, -1]
        
        # Use atan2(sin(θ), cos(θ)) for terminal orientation error as well
        theta_error_f = ca.atan2(ca.sin(theta_diff_f), ca.cos(theta_diff_f))
        
        # Very strong terminal position cost to force convergence
        cost += 5.0 * self.F[0,0] * pos_error_x_f**2 
        cost += 5.0 * self.F[1,1] * pos_error_y_f**2 
        cost += self.F[2,2] * theta_error_f**2
        
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
            'ipopt.tol': 5e-2,
            'ipopt.acceptable_tol': 1e-1,
            'ipopt.acceptable_iter': 2,
            'ipopt.warm_start_init_point': 'no',
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.linear_solver': 'mumps',
            'ipopt.mu_strategy': 'monotone',
            'ipopt.nlp_scaling_method': 'none',
            'ipopt.constr_viol_tol': 5e-2,
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
            x[0] + u_safe[0] * cos_theta * self.dt,
            x[1] + u_safe[0] * sin_theta * self.dt,
            x[2] + u_safe[1] * self.dt,
            u_safe[0],
            u_safe[1]
        )

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
        # Simplified, robust state validation
        current_state = np.array(current_state[:5], dtype=np.float64)  # Ensure 5-state
        
        # Basic validation - reject clearly invalid states
        if np.any(np.isnan(current_state)) or np.any(np.isinf(current_state)):
            print("Invalid current state detected")
            return np.array([0.0, 0.0])
        
        # Basic bounds checking
        current_state[2] = np.clip(current_state[2], -2*np.pi, 2*np.pi)  # theta
        current_state[3] = np.clip(current_state[3], -1.0, 1.0)          # v
        current_state[4] = np.clip(current_state[4], -np.pi, np.pi)      # omega
        
        # Validate reference trajectory
        if self.ref_traj is None:
            print("No reference trajectory set")
            return np.array([0.0, 0.0])
        
        ref_traj = np.array(self.ref_traj)
        if np.any(np.isnan(ref_traj)) or np.any(np.isinf(ref_traj)):
            print("Invalid reference trajectory")
            return np.array([0.0, 0.0])
        
        # Set problem parameters
        try:
            self.opti.set_value(self.x0, current_state)
            self.opti.set_value(self.ref, ref_traj)
        except Exception as e:
            print(f"Error setting MPC parameters: {e}")
            return np.array([0.0, 0.0])
        
        # Simple cold start initialization for stability
        u_init = np.zeros((self.nu, self.N))
        x_init = np.tile(current_state.reshape(-1, 1), (1, self.N+1))
        
        try:
            self.opti.set_initial(self.U, u_init)
            self.opti.set_initial(self.X, x_init)
        except:
            pass
        
        # Solve with robust error handling
        try:
            sol = self.opti.solve()
            u_optimal = sol.value(self.U)
            self.last_solution = u_optimal
            return u_optimal[:, 0]  # Return first control input
            
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return np.array([0.0, 0.0])
    
    def robot_model_np(self, x, u):
        """System dynamics in numpy for warm starting"""
        return np.array([
            x[0] + u[0] * np.cos(x[2]) * self.dt,
            x[1] + u[0] * np.sin(x[2]) * self.dt,
            x[2] + u[1] * self.dt,
            u[0],
            u[1]
        ])

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range using atan2 for smoother handling of discontinuities"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def get_predicted_trajectory(self):
        """Get predicted trajectory from last solution"""
        try:
            if self.last_solution is not None:
                return self.opti.debug.value(self.X)
            else:
                return None
        except:
            return None
