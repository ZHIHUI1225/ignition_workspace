#!/usr/bin/env python3
import time
import math
import re
import os
import json
import copy
import numpy as np
import py_trees
import rclpy
import casadi as ca
import traceback
import threading
import tf_transformations
from scipy.interpolate import CubicSpline
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Float64

# Physical constants from Planning_pickup_simplified.py
aw_max = 0.2 * np.pi  # the maximum angular acceleration
w_max = 0.6 * np.pi   # the maximum angular velocity
r_limit = 0.75        # m
r_w = 0.033          # the radius of the wheel
v_max = w_max * r_w  # m/s
a_max = aw_max * r_w
l_r = 0.14           # the wheel base

def calculate_angular_velocity_limit(r):
    """
    Calculate the maximum angular velocity (ω_c) for a specific arc radius.
    
    This is derived from the differential drive constraint:
    [ω_r]   = [1/r_w,  l_r/(2*r_w)] * [ω_c*r]
    [ω_l]     [1/r_w, -l_r/(2*r_w)]   [ω_c]
    
    Args:
        r: Radius of the arc (meters)
    
    Returns:
        Maximum allowable angular velocity (rad/s)
    """
    if abs(r) <= l_r/2:
        # For very tight turns, one wheel would need to move backward
        # We set a conservative limit
        return 0.1 * w_max
    
    # Calculate limits for both wheels
    limit_right = w_max * r_w / (abs(r) + l_r/2)
    limit_left = w_max * r_w / (abs(r) - l_r/2)
    
    # Return the more restrictive limit
    return min(limit_right, limit_left)

def calculate_angular_acceleration_limit(r):
    """
    Calculate the maximum angular acceleration (a_c) for a specific arc radius.
    
    This is derived from the differential drive constraint:
    [a_r]   = [1/r_w,  l_r/(2*r_w)] * [a_c*r]
    [a_l]     [1/r_w, -l_r/(2*r_w)]   [a_c]
    
    Args:
        r: Radius of the arc (meters)
    
    Returns:
        Maximum allowable angular acceleration (rad/s²)
    """
    if abs(r) <= l_r/2:
        # For very tight turns, one wheel would need to move backward
        # We set a conservative limit
        return 0.1 * aw_max
    
    # Calculate limits for both wheels
    limit_right = aw_max * r_w / (abs(r) + l_r/2)
    limit_left = aw_max * r_w / (abs(r) - l_r/2)
    
    # Return the more restrictive limit
    return min(limit_right, limit_left)

def load_reeb_graph_from_file(file_path):
    """Load reeb graph from JSON file - placeholder implementation"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data  # This would return a proper graph object in the full implementation
    except Exception as e:
        print(f"Error loading reeb graph from {file_path}: {e}")
        return None
    
class ReplanPath(py_trees.behaviour.Behaviour):
    """Path replanning behavior with trajectory optimization"""
    
    def __init__(self, name, duration=1.5, robot_namespace="turtlebot0", case="simple_maze"):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
        self.robot_namespace = robot_namespace
        self.case = case
        
        # Extract robot ID from namespace for file paths
        self.robot_id = self.extract_namespace_number(robot_namespace)
        
        # Setup blackboard access to read pushing_estimated_time
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/pushing_estimated_time", 
            access=py_trees.common.Access.READ
        )
        
        # Replanning state
        self.replanning_complete = False
        self.replanned_successfully = False
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'turtlebot0' -> 0)"""
        match = re.search(r'turtlebot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def initialise(self):
        self.start_time = time.time()
        self.replanning_complete = False
        self.replanned_successfully = False
        self.feedback_message = f"Replanning path for {self.duration}s"
        print(f"[{self.name}] Starting trajectory replanning for robot {self.robot_id}...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        
        # Perform replanning on first update
        if not self.replanning_complete:
            try:
                # Get target time from blackboard
                raw_target_time = getattr(self.blackboard, f'{self.robot_namespace}/pushing_estimated_time', 45.0)
                
                # If the target time is too small (remaining trajectory time), use a more reasonable value
                # This happens when replanning is called near the end of the trajectory
                if raw_target_time < 5.0:  # Less than 5 seconds remaining
                    # Use the default initialization value or a reasonable minimum time
                    target_time = 45.0  # Use the default full trajectory time
                    print(f"[{self.name}] Raw target time ({raw_target_time:.2f}s) too small, using default: {target_time:.2f}s")
                else:
                    target_time = raw_target_time
                    
                print(f"[{self.name}] Target time for replanning: {target_time:.2f}s")
                
                # Perform trajectory replanning
                result = self.replan_trajectory_to_target(
                    case=self.case,
                    target_time=target_time,
                    robot_id=self.robot_id
                )
                
                self.replanning_complete = True
                self.replanned_successfully = result
                
                if result:
                    print(f"[{self.name}] Trajectory replanning successful!")
                    return py_trees.common.Status.SUCCESS
                else:
                    print(f"[{self.name}] Trajectory replanning failed!")
                    return py_trees.common.Status.FAILURE
                    
            except Exception as e:
                print(f"[{self.name}] Error during replanning: {e}")
                self.replanning_complete = True
                self.replanned_successfully = False
                return py_trees.common.Status.FAILURE
    
        # If replanning already completed, return result
        if self.replanning_complete:
            if self.replanned_successfully:
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        
        # Check if timeout exceeded
        if elapsed >= 120.0:  # 2 minutes timeout
            print(f"[{self.name}] Trajectory replanning timeout!")
            return py_trees.common.Status.FAILURE
                
        return py_trees.common.Status.RUNNING
    
    def replan_trajectory_to_target(self, case, target_time, robot_id):
        """
        Load trajectory parameters and replan to achieve target time using optimization
        Similar to replan_trajectory_parameters_to_target from Planning_pickup_simplified.py
        """
        print(f"[{self.name}] === Replanning Trajectory Parameters for {case} ===")
        print(f"[{self.name}] Robot ID: {robot_id}")
        print(f"[{self.name}] Target time: {target_time:.3f}s")
        
        data_dir = f'/root/workspace/data/{case}/'
        robot_file = f'{data_dir}robot_{robot_id}_trajectory_parameters_{case}.json'
        
        if os.path.exists(robot_file):
            try:
                with open(robot_file, 'r') as f:
                    data = json.load(f)
                print(f"[{self.name}] Loaded trajectory parameters from {robot_file}")
                return self.replan_trajectory_parameters_to_target_full(data, case, target_time, robot_id)

            except Exception as e:
                print(f"[{self.name}] Error loading {robot_file}: {e}")
                return None


    def replan_trajectory_parameters_to_target_full(self, trajectory_data, case, target_time, robot_id):
        """
        Complete implementation of trajectory replanning using CasADi optimization
        Based on the full implementation from Planning_pickup_simplified.py
        """
        
        # Calculate current time
        print(f"\n[{self.name}] 2. Analyzing current trajectory...")
        current_robot_time = trajectory_data.get('total_time', 0)
        print(f"[{self.name}]    Current time: {current_robot_time:.3f}s")
        print(f"[{self.name}]    Target time: {target_time:.3f}s")
        
        # Extract time segments
        time_segments = trajectory_data.get('time_segments', [])
        num_segments = len(time_segments)
        
        if num_segments == 0:
            print(f"[{self.name}] No time segments found!")
            return False
        
        print(f"[{self.name}]    Number of segments: {num_segments}")
        
        # Calculate current segment times
        current_segment_times = []
        for segment in time_segments:
            segment_time = 0.0
            if 'arc' in segment and isinstance(segment['arc'], list):
                segment_time += sum(segment['arc'])
            if 'line' in segment and isinstance(segment['line'], list):
                segment_time += sum(segment['line'])
            current_segment_times.append(segment_time)
        
        # Replan using CasADi optimization (full implementation)
        print(f"\n[{self.name}] 3. Optimizing trajectory times...")
        
        try:
            opti = ca.Opti()
            
            # Extract robot trajectory parameters for constraint calculations
            waypoints = trajectory_data.get('waypoints', [])
            phi = np.array(trajectory_data.get('phi', []))
            r0 = np.array(trajectory_data.get('r0', []))
            l = np.array(trajectory_data.get('l', []))
            phi_new = np.array(trajectory_data.get('phi_new', []))
            Flagb = trajectory_data.get('Flagb', [])
            
            # Create decision variables for detailed time segments
            delta_t_arcs = []
            delta_t_lines = []
            segment_arc_indices = []
            segment_line_indices = []
            
            # Process each segment to create detailed variables
            Deltal = 0.02  # Small segment length (from Planning_deltaT.py)
            
            for i, segment in enumerate(time_segments):
                # Get original arc and line times
                orig_arc_times = segment.get('arc', [])
                orig_line_times = segment.get('line', [])
                
                # Calculate segment parameters
                if i < len(r0) and i < len(l) and i < len(phi):
                    delta_phi = phi[i+1] - phi_new[i] if i+1 < len(phi) else 0
                    arc_length = abs(r0[i] * delta_phi) if abs(r0[i]) > 0 else 0
                    line_length = l[i]
                    
                    # Calculate the number of subsegments
                    N_arc = max(1, int(arc_length / Deltal)) if arc_length > 0.03 else 0
                    N_line = max(1, int(line_length / Deltal)) if line_length > 0.03 else 0
                    
                    # Create arc variables
                    if N_arc > 0 and len(orig_arc_times) > 0:
                        arc_vars = [opti.variable() for _ in range(N_arc)]
                        delta_t_arcs.append(arc_vars)
                        segment_arc_indices.append(i)
                    else:
                        delta_t_arcs.append([])
                    
                    # Create line variables  
                    if N_line > 0 and len(orig_line_times) > 0:
                        line_vars = [opti.variable() for _ in range(N_line)]
                        delta_t_lines.append(line_vars)
                        segment_line_indices.append(i)
                    else:
                        delta_t_lines.append([])
            
            # Detailed constraints similar to Planning_deltaT.py
            all_accelerations = []  # Store acceleration terms for objective penalty
            
            for i in range(len(time_segments)):
                if i < len(r0) and i < len(l) and i < len(phi):
                    # Arc constraints
                    if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                        delta_phi = phi[i+1] - phi_new[i] if i+1 < len(phi) else 0
                        arc_length = abs(r0[i] * delta_phi)
                        arc_segment_length = arc_length / len(delta_t_arcs[i])
                        
                        # Initial state constraint for arc segments starting from zero velocity
                        if i == 0 or (i < len(Flagb) and Flagb[i] != 0):
                            min_t_initial = np.sqrt(2*arc_segment_length / calculate_angular_acceleration_limit(r0[i]) / abs(r0[i])) if abs(r0[i]) > 0 else 0.2
                            opti.subject_to(delta_t_arcs[i][0] >= min_t_initial)
                            opti.subject_to(delta_t_arcs[i][0] <= 10.0)
                        
                        for j, dt_arc in enumerate(delta_t_arcs[i]):
                            # Time bounds
                            opti.subject_to(dt_arc >= 0.20)
                            opti.subject_to(dt_arc <= 10.0)
                            
                            # Angular velocity constraint
                            if abs(r0[i]) > 0:
                                omega_c = arc_segment_length / abs(r0[i]) / dt_arc
                                w_max_arc = calculate_angular_velocity_limit(abs(r0[i]))
                                opti.subject_to(omega_c >= 0)
                                opti.subject_to(omega_c <= w_max_arc)
                            
                            # Angular acceleration constraint between consecutive arc subsegments
                            if j > 0:
                                v1 = arc_segment_length / delta_t_arcs[i][j-1]
                                v2 = arc_segment_length / delta_t_arcs[i][j]
                                t_avg = (delta_t_arcs[i][j-1] + delta_t_arcs[i][j])/2
                                a_tangential = (v2 - v1) / t_avg
                                all_accelerations.append(a_tangential)
                                
                                alpha = a_tangential / abs(r0[i])
                                aw_max_arc = calculate_angular_acceleration_limit(r0[i])
                                opti.subject_to(alpha >= -aw_max_arc)
                                opti.subject_to(alpha <= aw_max_arc)
                    
                    # Line constraints
                    if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                        line_length = l[i] if i < len(l) else 0
                        line_segment_length = line_length / len(delta_t_lines[i])
                        
                        for j, dt_line in enumerate(delta_t_lines[i]):
                            # Time bounds
                            opti.subject_to(dt_line >= 0.1)
                            opti.subject_to(dt_line <= 5.0)
                            
                            # Linear velocity constraint
                            velocity_expr = line_segment_length / dt_line
                            opti.subject_to(velocity_expr >= 0)
                            opti.subject_to(velocity_expr <= v_max)
                            
                            # Linear acceleration constraint between consecutive line subsegments
                            if j > 0:
                                a_lin = line_segment_length * (1/dt_line - 1/delta_t_lines[i][j-1]) / ((dt_line + delta_t_lines[i][j-1])/2)
                                all_accelerations.append(a_lin)
                                
                                constraint_expr = (dt_line**2 - delta_t_lines[i][j-1]**2) / delta_t_lines[i][j-1] / dt_line
                                opti.subject_to(constraint_expr >= -a_max/2/line_segment_length)
                                opti.subject_to(constraint_expr <= a_max/2/line_segment_length)
            
                # Calculate total time for detailed optimization
                total_time = 0
                for i in range(len(time_segments)):
                    if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                        total_time += ca.sum1(ca.vertcat(*delta_t_arcs[i]))
                    if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                        total_time += ca.sum1(ca.vertcat(*delta_t_lines[i]))
            
            # Objective: minimize deviation from target time + acceleration penalty
            # Use adaptive weighting based on feasibility of target time
            current_total_time = sum(current_segment_times)
            
            if target_time < current_total_time * 0.5:
                # When target time is unrealistic, prioritize finding the minimum feasible time
                # while still considering the target as a soft constraint
                min_feasible_time = current_total_time * 0.3
                feasible_target = max(target_time, min_feasible_time)
                
                # Soft constraint towards target but allow deviation
                objective = 0.1 * (total_time - target_time)**2 + 10.0 * (total_time - feasible_target)**2
                print(f"[{self.name}] Using adaptive objective with feasible_target: {feasible_target:.3f}s")
            else:
                # Normal case - target time is reasonable
                objective = (total_time - target_time)**2
            
            # Add acceleration penalty for smoother trajectories
            if 'all_accelerations' in locals() and all_accelerations:
                acceleration_penalty_weight = 1000.0
                accel_terms_vector = ca.vertcat(*all_accelerations)
                objective += acceleration_penalty_weight * ca.sumsqr(accel_terms_vector)
            
            opti.minimize(objective)
            
            # Total time constraints (allow more flexibility for robust optimization)
            # If target time is much smaller than current time, use current time as basis
            current_total_time = sum(current_segment_times)
            
            # Determine feasible time bounds
            if target_time < current_total_time * 0.5:
                # Target time is much smaller - use more flexible bounds
                min_feasible_time = max(target_time, current_total_time * 0.3)
                max_feasible_time = current_total_time * 1.2
                print(f"[{self.name}] Target time {target_time:.3f}s is much smaller than current {current_total_time:.3f}s")
                print(f"[{self.name}] Using flexible bounds: [{min_feasible_time:.3f}s, {max_feasible_time:.3f}s]")
            else:
                # Normal case - target time is reasonable
                min_feasible_time = target_time * 0.8
                max_feasible_time = target_time * 1.2
                
            opti.subject_to(total_time >= min_feasible_time)
            opti.subject_to(total_time <= max_feasible_time)
            
          
            # Detailed subsegment optimization - use loaded data
            for i, segment in enumerate(time_segments):
                # Set initial values for arc variables using loaded data
                if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                    orig_arc_times = segment.get('arc', [])
                    if orig_arc_times and len(orig_arc_times) == len(delta_t_arcs[i]):
                        for j, arc_var in enumerate(delta_t_arcs[i]):
                            initial_value = max(0.05, orig_arc_times[j])
                            opti.set_initial(arc_var, initial_value)
                    else:
                        for arc_var in delta_t_arcs[i]:
                            opti.set_initial(arc_var, 0.5)
                    
                    # Set initial values for line variables using loaded data
                    if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                        orig_line_times = segment.get('line', [])
                        if orig_line_times and len(orig_line_times) == len(delta_t_lines[i]):
                            for j, line_var in enumerate(delta_t_lines[i]):
                                initial_value = max(0.05, orig_line_times[j])
                                opti.set_initial(line_var, initial_value)
                        else:
                            for line_var in delta_t_lines[i]:
                                opti.set_initial(line_var, 0.5)
            
            # Solver settings
            opts = {
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.sb': 'yes',
                'ipopt.max_iter': 3000,
                'ipopt.acceptable_tol': 1e-4
            }
            opti.solver('ipopt', opts)
            
            # Solve
            sol = opti.solve()
            
            # Extract solution
            optimized_total_time = float(sol.value(total_time))
            
                # Detailed subsegment optimization - reconstruct segment times
            T_opt_list = []
            for i in range(len(time_segments)):
                segment_time = 0.0
                if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                    for arc_var in delta_t_arcs[i]:
                        segment_time += float(sol.value(arc_var))
                if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                    for line_var in delta_t_lines[i]:
                        segment_time += float(sol.value(line_var))
                T_opt_list.append(max(0.1, segment_time))
            
            # Create detailed trajectory with subsegment times
            replanned_trajectory = self.create_detailed_replanned_trajectory(
                trajectory_data, delta_t_arcs, delta_t_lines, sol, optimized_total_time
            )
            
            optimization_results = {
                'original_total_time': current_robot_time,
                'target_time': target_time,
                'optimized_total_time': optimized_total_time,
                'segment_times': T_opt_list,
                'deviation': abs(optimized_total_time - target_time),
                'improvement': current_robot_time - optimized_total_time
            }
            
            print(f"[{self.name}] ✓ Optimization successful!")
            print(f"[{self.name}]   Original time: {current_robot_time:.3f}s")
            print(f"[{self.name}]   Target time: {target_time:.3f}s")
            print(f"[{self.name}]   Optimized time: {optimized_total_time:.3f}s")
            print(f"[{self.name}]   Deviation from target: {abs(optimized_total_time - target_time):.3f}s")
            print(f"[{self.name}]   Improvement from original: {current_robot_time - optimized_total_time:.3f}s")
            
            # Save results
            saved_file = self.save_replanned_trajectory_parameters(replanned_trajectory, case, robot_id)
            
            if saved_file:
                # Generate discrete trajectory from replanned parameters
                discrete_success = self.generate_discrete_trajectory_from_replanned(case, robot_id, dt=0.1)
                
                if discrete_success:
                    print(f"\n[{self.name}] === Complete Replanning Summary ===")
                    print(f"[{self.name}] Robot {robot_id} results:")
                    print(f"[{self.name}]   Original → Replanned: {optimization_results['original_total_time']:.3f}s → {optimization_results['optimized_total_time']:.3f}s")
                    print(f"[{self.name}]   Target: {optimization_results['target_time']:.3f}s")
                    print(f"[{self.name}]   Deviation: {optimization_results['deviation']:.3f}s")
                    print(f"[{self.name}]   Improvement: {optimization_results['improvement']:.3f}s")
                    return True
                else:
                    print(f"[{self.name}] Failed to generate discrete trajectory from replanned parameters")
                    return False
            else:
                print(f"[{self.name}] Failed to save replanned trajectory parameters")
                return False
                
        except Exception as e:
            print(f"[{self.name}] ✗ Optimization failed: {e}")
            print(f"[{self.name}] ⚠️  Warning: Could not optimize trajectory parameters for Robot {robot_id}")
            print(f"[{self.name}] ⚠️  Reason: {str(e)}")
            
            # Try fallback: simple proportional scaling if target time is much smaller
            current_total_time = sum(current_segment_times)
            if target_time < current_total_time * 0.5:
                print(f"[{self.name}] 🔄 Attempting fallback: simple proportional scaling")
                return self.replan_trajectory_simple_approach(trajectory_data, case, target_time, robot_id)
            
            return False

    def create_replanned_trajectory_from_optimization_simple(self, original_data, optimized_times, total_time):
        """
        Create new trajectory data with optimized times maintaining original structure (simple version)
        """
        # Deep copy original data
        replanned_data = copy.deepcopy(original_data)
        
        # Update time segments
        time_segments = replanned_data.get('time_segments', [])
        
        for i, segment in enumerate(time_segments):
            if i < len(optimized_times):
                new_segment_time = optimized_times[i]
                
                # Get original arc and line times
                orig_arc_times = segment.get('arc', [])
                orig_line_times = segment.get('line', [])
                
                # Calculate original total for this segment
                orig_segment_total = 0.0
                if isinstance(orig_arc_times, list):
                    orig_segment_total += sum(orig_arc_times)
                if isinstance(orig_line_times, list):
                    orig_segment_total += sum(orig_line_times)
                
                # Distribute new segment time proportionally based on original structure
                if orig_segment_total > 0:
                    arc_proportion = sum(orig_arc_times) / orig_segment_total if orig_arc_times else 0
                    line_proportion = sum(orig_line_times) / orig_segment_total if orig_line_times else 0
                    
                    # Distribute times proportionally while maintaining structure
                    if isinstance(orig_arc_times, list) and len(orig_arc_times) > 0:
                        total_arc_time = new_segment_time * arc_proportion
                        segment['arc'] = [total_arc_time / len(orig_arc_times)] * len(orig_arc_times)
                    
                    # Distribute line times proportionally
                    if isinstance(orig_line_times, list) and len(orig_line_times) > 0:
                        total_line_time = new_segment_time * line_proportion
                        segment['line'] = [total_line_time / len(orig_line_times)] * len(orig_line_times)
                else:
                    # If no original times, distribute evenly
                    segment['arc'] = [new_segment_time * 0.3]  # 30% for arc
                    segment['line'] = [new_segment_time * 0.7]  # 70% for line
        
        # Update total time
        replanned_data['total_time'] = total_time
        
        # Update metadata
        if 'metadata' in replanned_data:
            replanned_data['metadata']['replanned'] = True
            replanned_data['metadata']['replan_timestamp'] = str(np.datetime64('now'))
            replanned_data['metadata']['original_total_time'] = original_data.get('total_time', 0)
            replanned_data['metadata']['replanned_total_time'] = total_time
        
        return replanned_data

    
    def solve_trajectory_optimization(self, original_trajectory, target_time):
        """
        Solve optimization problem to replan trajectory for target time
        Uses CasADi to solve nonlinear optimization problem
        """
        try:
            print(f"[{self.name}] Setting up optimization problem...")
            
            # Extract trajectory information
            n_points = len(original_trajectory)
            dt_original = 0.1  # Original time step
            current_duration = n_points * dt_original
            
            # Extract positions and velocities from original trajectory
            x_orig = [pt[0] for pt in original_trajectory]
            y_orig = [pt[1] for pt in original_trajectory] 
            theta_orig = [pt[2] for pt in original_trajectory]
            v_orig = [pt[3] for pt in original_trajectory]
            w_orig = [pt[4] for pt in original_trajectory]
            
            # Segment the trajectory for optimization
            n_segments = min(10, n_points // 5)  # Limit number of segments for computational efficiency
            segment_size = n_points // n_segments
            
            print(f"[{self.name}] Dividing trajectory into {n_segments} segments of size ~{segment_size}")
            
            # Setup CasADi optimization problem
            opti = ca.Opti()
            
            # Decision variables: time allocation for each segment
            segment_times = opti.variable(n_segments)
            
            # Constraints: all segment times must be positive and reasonable
            min_segment_time = 0.1  # Minimum time per segment
            max_segment_time = target_time * 0.7  # Maximum time per segment (prevent one segment taking all time)
            
            for i in range(n_segments):
                opti.subject_to(segment_times[i] >= min_segment_time)
                opti.subject_to(segment_times[i] <= max_segment_time)
            
            
            # Smoothness constraints: adjacent segments shouldn't differ too much
            max_time_ratio = 3.0  # Adjacent segments can differ by at most 3x
            for i in range(n_segments - 1):
                opti.subject_to(segment_times[i] <= max_time_ratio * segment_times[i+1])
                opti.subject_to(segment_times[i+1] <= max_time_ratio * segment_times[i])
            
            # Objective: minimize deviation from original velocity profile while maintaining smoothness
            cost = 0
            
            # Cost for deviating from original timing profile (weighted)
            original_segment_times = [current_duration / n_segments] * n_segments
            for i in range(n_segments):
                time_deviation = (segment_times[i] - original_segment_times[i]) ** 2
                cost += 0.5 * time_deviation  # Reduced weight to allow more flexibility
            
            # Smoothness cost: penalize large differences between adjacent segment times
            for i in range(n_segments - 1):
                smoothness_cost = (segment_times[i] - segment_times[i+1]) ** 2
                cost += 0.2 * smoothness_cost  # Increased smoothness weight
            
            # Velocity constraint cost: ensure velocities remain feasible
            max_velocity = 0.8  # m/s - increased maximum velocity
            max_angular_velocity = 1.5  # rad/s - increased maximum angular velocity
            
            for i in range(n_segments):
                # Calculate segment characteristics
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, n_points - 1)
                
                if start_idx < len(x_orig) and end_idx < len(x_orig):
                    # Calculate required velocity for this segment
                    segment_distance = math.sqrt(
                        (x_orig[end_idx] - x_orig[start_idx])**2 + 
                        (y_orig[end_idx] - y_orig[start_idx])**2
                    )
                    
                    if segment_distance > 0:
                        # Required velocity = distance / time
                        required_velocity = segment_distance / segment_times[i]
                        
                        # Soft constraint for velocity limits
                        velocity_violation = ca.fmax(0, required_velocity - max_velocity)
                        cost += 100 * velocity_violation**2  # High penalty for velocity violations
                        
                        # Add preference for reasonable velocities (not too slow either)
                        min_velocity = 0.05  # m/s minimum
                        slow_penalty = ca.fmax(0, min_velocity - required_velocity)
                        cost += 10 * slow_penalty**2
            
            # Set objective
            opti.minimize(cost)
            
            # Initial guess: proportional to original segment distances
            initial_times = []
            total_distance = 0
            segment_distances = []
            
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, n_points - 1)
                
                if start_idx < len(x_orig) and end_idx < len(x_orig):
                    segment_distance = math.sqrt(
                        (x_orig[end_idx] - x_orig[start_idx])**2 + 
                        (y_orig[end_idx] - y_orig[start_idx])**2
                    )
                    segment_distances.append(segment_distance)
                    total_distance += segment_distance
                else:
                    segment_distances.append(1.0)  # Default distance
                    total_distance += 1.0
            
            # Allocate time proportional to distance, but bounded
            for i in range(n_segments):
                if total_distance > 0:
                    proportional_time = (segment_distances[i] / total_distance) * target_time
                    # Bound the initial guess to reasonable values
                    bounded_time = max(min_segment_time, min(proportional_time, max_segment_time))
                    initial_times.append(bounded_time)
                else:
                    initial_times.append(target_time / n_segments)
            
            # Normalize to ensure total equals target_time
            time_sum = sum(initial_times)
            if time_sum > 0:
                initial_times = [t * target_time / time_sum for t in initial_times]
            else:
                initial_times = [target_time / n_segments] * n_segments
            
            opti.set_initial(segment_times, initial_times)
            print(f"[{self.name}] Initial time guess: {[f'{t:.3f}' for t in initial_times]}")
            
            # Configure solver
            opts = {
                'ipopt.max_iter': 500,
                'ipopt.tol': 1e-6,
                'ipopt.print_level': 0,
                'print_time': False,
                'ipopt.acceptable_tol': 1e-4,
                'ipopt.acceptable_obj_change_tol': 1e-6
            }
            opti.solver('ipopt', opts)
            
            print(f"[{self.name}] Solving optimization problem...")
            
            # Solve the optimization problem
            sol = opti.solve()
            
            # Extract optimal segment times
            optimal_times = sol.value(segment_times);
            
            print(f"[{self.name}] Optimization solved successfully!")
            print(f"[{self.name}] Optimal segment times: {[f'{t:.3f}' for t in optimal_times]}")
            
            # Generate new trajectory with optimal timing
            optimized_trajectory = self.generate_trajectory_from_optimal_times(
                original_trajectory, optimal_times, target_time
            )
            
            return optimized_trajectory
            
        except Exception as e:
            print(f"[{self.name}] Optimization failed: {e}")
            # Fallback to simple interpolation if optimization fails
            return self.create_simple_time_interpolated_trajectory(original_trajectory, target_time)
    
    def generate_trajectory_from_optimal_times(self, original_trajectory, optimal_times, target_time):
        """
        Generate new trajectory using optimal time allocation
        """
        try:
            n_points = len(original_trajectory)
            n_segments = len(optimal_times)
            segment_size = n_points // n_segments
            
            # New time step to achieve target duration
            dt_new = 0.1  # Keep same discretization
            n_new_points = int(target_time / dt_new)
            
            # Extract original trajectory data
            x_orig = np.array([pt[0] for pt in original_trajectory])
            y_orig = np.array([pt[1] for pt in original_trajectory])
            theta_orig = np.array([pt[2] for pt in original_trajectory])
            
            # Create cumulative time arrays for original and new trajectories
            original_times = np.arange(0, len(original_trajectory) * 0.1, 0.1)
            
            # Build new time array based on optimal segment times
            new_times = []
            current_time = 0.0
            
            for i, seg_time in enumerate(optimal_times):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, n_points)
                segment_points = end_idx - start_idx
                
                if segment_points > 0:
                    segment_dt = seg_time / segment_points
                    for j in range(segment_points):
                        new_times.append(current_time + j * segment_dt)
                    current_time += seg_time
            
            # Ensure we have the right number of time points
            while len(new_times) < n_points:
                new_times.append(new_times[-1] + dt_new)
            new_times = new_times[:n_points]
            
            # Create uniform output time grid
            output_times = np.arange(0, target_time, dt_new)
            
            # Interpolate trajectory to new timing
            from scipy.interpolate import interp1d
            
            # Create interpolation functions
            if len(new_times) >= 2 and len(original_times) >= 2:
                # Map from original times to new times
                time_mapping = interp1d(original_times[:len(new_times)], new_times, 
                                      kind='linear', fill_value='extrapolate')
                
                # Interpolate positions using the new timing
                interp_x = interp1d(new_times, x_orig[:len(new_times)], 
                                  kind='cubic', fill_value='extrapolate')
                interp_y = interp1d(new_times, y_orig[:len(new_times)], 
                                  kind='cubic', fill_value='extrapolate')
                interp_theta = interp1d(new_times, theta_orig[:len(new_times)], 
                                      kind='linear', fill_value='extrapolate')
                
                # Generate new trajectory
                new_trajectory = []
                for t in output_times:
                    x = float(interp_x(t))
                    y = float(interp_y(t))
                    theta = float(interp_theta(t))
                    
                    # Calculate velocities
                    if len(new_trajectory) > 0:
                        prev_x, prev_y, prev_theta = new_trajectory[-1][:3]
                        dt = dt_new
                        
                        # Linear velocity
                        v = math.sqrt((x - prev_x)**2 + (y - prev_y)**2) / dt
                        
                        # Angular velocity
                        dtheta = theta - prev_theta
                        while dtheta > math.pi:
                            dtheta -= 2 * math.pi
                        while dtheta < -math.pi:
                            dtheta += 2 * math.pi
                        w = dtheta / dt
                    else:
                        v = 0.0
                        w = 0.0
                    
                    new_trajectory.append([x, y, theta, v, w])
                
                print(f"[{self.name}] Generated optimized trajectory with {len(new_trajectory)} points")
                return new_trajectory
            else:
                print(f"[{self.name}] Insufficient points for interpolation")
                return self.create_simple_time_interpolated_trajectory(original_trajectory, target_time)
                
        except Exception as e:
            print(f"[{self.name}] Error generating trajectory from optimal times: {e}")
            return self.create_simple_time_interpolated_trajectory(original_trajectory, target_time)
    
    def create_simple_time_interpolated_trajectory(self, original_trajectory, target_time):
        """
        Fallback method: simple time interpolation of trajectory
        """
        try:
            dt_new = 0.1
            n_new_points = int(target_time / dt_new)
            n_orig_points = len(original_trajectory)
            
            if n_orig_points < 2:
                return original_trajectory
            
            # Create uniform interpolation
            new_trajectory = []
            for i in range(n_new_points):
                # Map new index to original trajectory
                orig_index = (i / (n_new_points - 1)) * (n_orig_points - 1)
                
                # Linear interpolation between adjacent points
                idx_low = int(orig_index)
                idx_high = min(idx_low + 1, n_orig_points - 1)
                alpha = orig_index - idx_low
                
                if idx_low < len(original_trajectory) and idx_high < len(original_trajectory):
                    pt_low = original_trajectory[idx_low]
                    pt_high = original_trajectory[idx_high]
                    
                    # Interpolate all components
                    x = pt_low[0] + alpha * (pt_high[0] - pt_low[0])
                    y = pt_low[1] + alpha * (pt_high[1] - pt_low[1])
                    theta = pt_low[2] + alpha * (pt_high[2] - pt_low[2])
                    v = pt_low[3] + alpha * (pt_high[3] - pt_low[3])
                    w = pt_low[4] + alpha * (pt_high[4] - pt_low[4])
                    
                    new_trajectory.append([x, y, theta, v, w])
            
            print(f"[{self.name}] Created fallback interpolated trajectory with {len(new_trajectory)} points")
            return new_trajectory
            
        except Exception as e:
            print(f"[{self.name}] Error in fallback trajectory creation: {e}")
            return original_trajectory

    def load_trajectory_parameters_individual(self, case, robot_id):
        """Load trajectory parameters from individual robot trajectory parameter file"""
        data_dir = f'/root/workspace/data/{case}/'
        robot_file = f'{data_dir}robot_{robot_id}_trajectory_parameters_{case}.json'
        
        if os.path.exists(robot_file):
            try:
                with open(robot_file, 'r') as f:
                    data = json.load(f)
                print(f"[{self.name}] Loaded trajectory parameters from {robot_file}")
                return data
            except Exception as e:
                print(f"[{self.name}] Error loading {robot_file}: {e}")
                return None
        else:
            print(f"[{self.name}] Robot trajectory file not found: {robot_file}")
            return None

    def solve_parameter_optimization(self, trajectory_data, target_time, current_segment_times):
        """
        Solve optimization problem for trajectory parameters using CasADi
        Based on the approach in Planning_pickup_simplified.py with detailed subsegment optimization
        """
        try:
            print(f"[{self.name}] Setting up parameter optimization...")
            
            # Extract trajectory components from loaded data
            time_segments = trajectory_data.get('time_segments', [])
            n_segments = len(time_segments)
            
            if n_segments == 0:
                print(f"[{self.name}] No segments found for optimization")
                return None
            
            # Physical constants (from Planning_pickup_simplified.py)
            aw_max = 0.2 * np.pi  # maximum angular acceleration
            w_max = 0.6 * np.pi   # maximum angular velocity  
            r_w = 0.033          # wheel radius
            v_max = w_max * r_w  # maximum linear velocity
            a_max = aw_max * r_w # maximum linear acceleration
            l_r = 0.14           # wheel base
            
            # Setup CasADi optimization
            opti = ca.Opti()
            
            # Create decision variables for each subsegment
            delta_t_arcs = []   # List of arc time variables for each segment
            delta_t_lines = []  # List of line time variables for each segment
            
            # Extract subsegment structure from time_segments
            for i, segment in enumerate(time_segments):
                arc_times = segment.get('arc', [])
                line_times = segment.get('line', [])
                
                # Create variables for arc subsegments
                if len(arc_times) > 0:
                    arc_vars = [opti.variable() for _ in range(len(arc_times))]
                    delta_t_arcs.append(arc_vars)
                else:
                    delta_t_arcs.append([])
                
                # Create variables for line subsegments  
                if len(line_times) > 0:
                    line_vars = [opti.variable() for _ in range(len(line_times))]
                    delta_t_lines.append(line_vars)
                else:
                    delta_t_lines.append([])
            
            # Flatten all variables for the optimization vector
            all_vars = []
            for i in range(n_segments):
                all_vars.extend(delta_t_arcs[i])
                all_vars.extend(delta_t_lines[i])
            
            if not all_vars:
                print(f"[{self.name}] No optimization variables created")
                return None
            
            opt_vars = ca.vertcat(*all_vars)
            
            # Constraints
            g = []
            lbg = []
            ubg = []
            all_accelerations = []  # For acceleration penalty in objective
            
            # Constraints for each segment
            for i in range(n_segments):
                arc_vars = delta_t_arcs[i]
                line_vars = delta_t_lines[i]
                
                # Arc subsegment constraints
                for j, arc_var in enumerate(arc_vars):
                    # Time bounds
                    opti.subject_to(arc_var >= 0.1)  # Minimum time
                    opti.subject_to(arc_var <= target_time * 0.5)  # Maximum time
                    
                    # Angular acceleration constraint between consecutive arc subsegments
                    if j > 0:
                        # Simplified acceleration constraint
                        prev_arc_var = arc_vars[j-1]
                        accel_term = (1/arc_var - 1/prev_arc_var) / ((arc_var + prev_arc_var)/2)
                        all_accelerations.append(accel_term)
                        
                        # Bound acceleration
                        g.append(accel_term)
                        lbg.append(-2.0)  # Conservative acceleration limit
                        ubg.append(2.0)
                
                # Line subsegment constraints  
                for j, line_var in enumerate(line_vars):
                    # Time bounds
                    opti.subject_to(line_var >= 0.1)  # Minimum time
                    opti.subject_to(line_var <= target_time * 0.5)  # Maximum time
                    
                    # Linear acceleration constraint between consecutive line subsegments
                    if j > 0:
                        # Simplified acceleration constraint
                        prev_line_var = line_vars[j-1]
                        accel_term = (1/line_var - 1/prev_line_var) / ((line_var + prev_line_var)/2)
                        all_accelerations.append(accel_term)
                        
                        # Bound acceleration
                        g.append(accel_term)
                        lbg.append(-2.0)  # Conservative acceleration limit  
                        ubg.append(2.0)
                
                # Acceleration continuity between arc and line in same segment
                if len(arc_vars) > 0 and len(line_vars) > 0:
                    # Transition from last arc subsegment to first line subsegment
                    last_arc = arc_vars[-1]
                    first_line = line_vars[0]
                    
                    # Simplified velocity continuity constraint
                    transition_accel = (1/first_line - 1/last_arc) / ((last_arc + first_line)/2)
                    all_accelerations.append(transition_accel)
                    
                    g.append(transition_accel)
                    lbg.append(-2.0)
                    ubg.append(2.0)
            
            # Total time constraint - sum of all subsegments must equal target time
            total_time_expr = ca.sum1(opt_vars)
            opti.subject_to(total_time_expr == target_time)
            
            # Segment time constraints - each segment should be reasonable
            for i in range(n_segments):
                segment_time = ca.sum1(ca.vertcat(*delta_t_arcs[i])) + ca.sum1(ca.vertcat(*delta_t_lines[i])) if (delta_t_arcs[i] or delta_t_lines[i]) else 0
                
                if segment_time != 0:  # Only add constraint if segment has variables
                    opti.subject_to(segment_time >= 0.5)  # Minimum segment time
                    opti.subject_to(segment_time <= target_time * 0.8)  # Maximum segment time
            
            # Objective function (based on Planning_pickup_simplified.py)
            cost = 0
            
            # Primary objective: Match target time exactly (high weight)
            time_matching_weight = 10000.0
            cost += time_matching_weight * (total_time_expr - target_time)**2
            
            # Secondary objective: Minimize deviation from original segment time ratios
            original_total = sum(current_segment_times)
            if original_total > 0:
                for i in range(n_segments):
                    # Calculate new segment time
                    new_segment_time = (ca.sum1(ca.vertcat(*delta_t_arcs[i])) + 
                                      ca.sum1(ca.vertcat(*delta_t_lines[i]))) if (delta_t_arcs[i] or delta_t_lines[i]) else 0
                    
                    if new_segment_time != 0 and i < len(current_segment_times):
                        # Original ratio vs new ratio
                        original_ratio = current_segment_times[i] / original_total
                        target_ratio = new_segment_time / target_time
                        ratio_deviation = (target_ratio - original_ratio) ** 2
                        cost += 20.0 * ratio_deviation
            
            # Tertiary objective: Acceleration smoothness penalty
            acceleration_penalty_weight = 500.0
            if all_accelerations:
                accel_vector = ca.vertcat(*all_accelerations)
                cost += acceleration_penalty_weight * ca.sumsqr(accel_vector)
            
            # Smoothness penalty between adjacent segments
            for i in range(n_segments - 1):
                if (delta_t_arcs[i] or delta_t_lines[i]) and (delta_t_arcs[i+1] or delta_t_lines[i+1]):
                    # Get last time of current segment and first time of next segment
                    current_last = delta_t_lines[i][-1] if delta_t_lines[i] else (delta_t_arcs[i][-1] if delta_t_arcs[i] else None)
                    next_first = delta_t_arcs[i+1][0] if delta_t_arcs[i+1] else (delta_t_lines[i+1][0] if delta_t_lines[i+1] else None)
                    
                    if current_last is not None and next_first is not None:
                        smoothness_cost = (current_last - next_first) ** 2
                        cost += 0.1 * smoothness_cost
            
            opti.minimize(cost)
            
            # Initial guess: scale original times proportionally
            scale_factor = target_time / sum(current_segment_times) if sum(current_segment_times) > 0 else 1.0
            initial_values = []
            
            for i, segment in enumerate(time_segments):
                arc_times = segment.get('arc', [])
                line_times = segment.get('line', [])
                
                # Scale arc times
                for orig_time in arc_times:
                    scaled_time = max(0.1, min(orig_time * scale_factor, target_time * 0.5))
                    initial_values.append(scaled_time)
                
                # Scale line times
                for orig_time in line_times:
                    scaled_time = max(0.1, min(orig_time * scale_factor, target_time * 0.5))
                    initial_values.append(scaled_time)
            
            # Normalize initial guess to meet target time
            if initial_values:
                guess_sum = sum(initial_values)
                if guess_sum > 0:
                    initial_values = [v * target_time / guess_sum for v in initial_values]
                
                opti.set_initial(opt_vars, initial_values)
            
            # Configure solver (same as Planning_pickup_simplified.py)
            opts = {
                'ipopt.max_iter': 5000,
                'ipopt.tol': 1e-6,
                'ipopt.print_level': 0,
                'print_time': False,
                'ipopt.acceptable_tol': 1e-4,
                'ipopt.acceptable_obj_change_tol': 1e-4
            }
            opti.solver('ipopt', opts)
            
            print(f"[{self.name}] Solving parameter optimization...")
            
            # Add constraints if any were created
            if g:
                opti.subject_to(ca.vertcat(*g) >= ca.vertcat(*lbg))
                opti.subject_to(ca.vertcat(*g) <= ca.vertcat(*ubg))
            
            sol = opti.solve()
            
            # Extract results and reconstruct segment times
            optimal_values = sol.value(opt_vars).full().flatten()
            
            # Reconstruct optimized segment times
            optimized_segment_times = []
            value_idx = 0
            
            for i in range(n_segments):
                arc_count = len(time_segments[i].get('arc', []))
                line_count = len(time_segments[i].get('line', []))
                
                segment_total = 0.0
                
                # Sum arc times for this segment
                for j in range(arc_count):
                    segment_total += optimal_values[value_idx]
                    value_idx += 1
                
                # Sum line times for this segment  
                for j in range(line_count):
                    segment_total += optimal_values[value_idx]
                    value_idx += 1
                
                optimized_segment_times.append(segment_total)
            optimized_total_time = sum(optimized_segment_times)
            
            print(f"[{self.name}] Parameter optimization successful!")
            print(f"[{self.name}] Optimal segment times: {[f'{t:.3f}' for t in optimized_segment_times]}")
            
            return {
                'optimized_segment_times': optimized_segment_times,
                'optimized_total_time': optimized_total_time,
                'original_total_time': sum(current_segment_times),
                'target_time': target_time,
                'deviation': abs(optimized_total_time - target_time),
                'improvement': abs(sum(current_segment_times) - target_time) - abs(optimized_total_time - target_time),
                'optimal_subsegment_values': optimal_values.tolist()
            }
            
        except Exception as e:
            print(f"[{self.name}] Parameter optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_detailed_replanned_trajectory(self, original_data, delta_t_arcs, delta_t_lines, sol, total_time):
        """
        Create new trajectory data with optimized detailed subsegment times
        
        Args:
            original_data: Original robot trajectory data
            delta_t_arcs: List of arc time variables for each segment
            delta_t_lines: List of line time variables for each segment  
            sol: CasADi solution object
            total_time: Total optimized time
            
        Returns:
            Updated trajectory data with new detailed timing
        """
        # Deep copy original data
        replanned_data = copy.deepcopy(original_data)
        
        # Update time segments with detailed subsegment times
        time_segments = replanned_data.get('time_segments', [])
        
        for i, segment in enumerate(time_segments):
            # Update arc times
            if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                new_arc_times = []
                for arc_var in delta_t_arcs[i]:
                    new_arc_times.append(float(sol.value(arc_var)))
                segment['arc'] = new_arc_times
            
            # Update line times
            if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                new_line_times = []
                for line_var in delta_t_lines[i]:
                    new_line_times.append(float(sol.value(line_var)))
                segment['line'] = new_line_times
        
        # Update total time
        replanned_data['total_time'] = total_time
        
        # Update metadata
        if 'metadata' in replanned_data:
            replanned_data['metadata']['replanned'] = True
            replanned_data['metadata']['replan_timestamp'] = str(np.datetime64('now'))
            replanned_data['metadata']['original_total_time'] = original_data.get('total_time', 0)
            replanned_data['metadata']['replanned_total_time'] = total_time
            replanned_data['metadata']['detailed_optimization'] = True
        
        return replanned_data

    def generate_single_robot_discrete_trajectory(self, robot_id, waypoints, phi, r0, l, phi_new, 
                                                time_segments, Flagb, reeb_graph, dt, save_dir, case):
        """
        Generate discrete trajectory for a single robot using parameters.
        Following the exact pattern from Planning_pickup_simplified.py for consistency.
        
        Args:
            robot_id: Robot ID
            waypoints: List of waypoint indices in the reeb graph
            phi: List of angles at each waypoint  
            r0: List of arc radii for each segment
            l: List of line lengths for each segment
            phi_new: List of adjusted angles accounting for flag values
            time_segments: List of dictionaries with 'arc' and 'line' time values for each segment
            Flagb: List of flag values for each waypoint
            reeb_graph: The reeb graph containing waypoint coordinates
            dt: Time step for uniform sampling
            save_dir: Directory to save files
            case: Case name
            
        Returns:
            Dictionary containing trajectory data and file paths
        """
        try:
            print(f"[{self.name}] Processing Robot {robot_id}")
            
            # Generate discrete trajectory using trajectory parameters
            N = len(waypoints) - 1  # Number of segments
            
            # Generate trajectory points using discretization from parameters
            trajectory_points = self.discretize_trajectory_from_parameters(
                waypoints, phi, r0, l, phi_new, time_segments, Flagb, dt
            )
            
            if not trajectory_points:
                print(f"[{self.name}] Failed to generate discrete trajectory points")
                return None
            
            # Save discrete trajectory to JSON file following tb0_Trajectory.json format
            discrete_trajectory_file = os.path.join(save_dir, f'tb{robot_id}_Trajectory_replanned.json')
            trajectory_data = {
                "Trajectory": trajectory_points
            }
            
            with open(discrete_trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, separators=(',', ': '))  # Compact format like originals
            
            print(f"[{self.name}] Discrete trajectory saved: {discrete_trajectory_file}")
            
            # Calculate trajectory statistics
            total_time = len(trajectory_points) * dt
            
            # Calculate basic trajectory statistics for reporting
            distances = []
            velocities = []
            for i in range(1, len(trajectory_points)):
                prev_point = trajectory_points[i-1]
                curr_point = trajectory_points[i]
                dx = curr_point[0] - prev_point[0]
                dy = curr_point[1] - prev_point[1]
                distance = math.sqrt(dx*dx + dy*dy)
                distances.append(distance)
                velocities.append(curr_point[3])  # Linear velocity
            
            stats = {
                "total_distance": sum(distances) if distances else 0.0,
                "average_velocity": sum(velocities) / len(velocities) if velocities else 0.0,
                "max_velocity": max(velocities) if velocities else 0.0,
                "min_velocity": min(velocities) if velocities else 0.0
            }
            
            print(f"[{self.name}] Robot {robot_id} trajectory stats:")
            print(f"[{self.name}]   Points: {len(trajectory_points)}")
            print(f"[{self.name}]   Duration: {total_time:.3f}s")
            print(f"[{self.name}]   Distance: {stats['total_distance']:.3f}m")
            print(f"[{self.name}]   Avg velocity: {stats['average_velocity']:.3f}m/s")
            
            return {
                "robot_id": robot_id,
                "discrete_trajectory_file": discrete_trajectory_file,
                "trajectory_points": trajectory_points,
                "statistics": stats,
                "total_time": total_time,
                "num_points": len(trajectory_points)
            }
            
        except Exception as e:
            print(f"[{self.name}] Error in generate_single_robot_discrete_trajectory: {e}")
            traceback.print_exc()
            return None

    def discretize_trajectory_from_parameters(self, waypoints, phi, r0, l, phi_new, time_segments, Flagb, dt):
        """
        Convert trajectory parameters to discrete trajectory points following the same approach as Planning_pickup_simplified.py
        
        Args:
            waypoints: List of waypoint indices
            phi: List of angles at waypoints
            r0: List of arc radii for each segment
            l: List of line lengths for each segment
            phi_new: List of adjusted angles
            time_segments: List of time segment dictionaries with 'arc' and 'line' times
            Flagb: List of flag values
            dt: Time step for discretization
            
        Returns:
            List of trajectory points [x, y, theta, v, w] or None if failed
        """
        try:
            print(f"[{self.name}] Discretizing trajectory from parameters...")
            
            # Initialize trajectory generation
            trajectory_points = []
            current_time = 0.0
            
            # Starting position (assuming waypoints reference reeb graph positions)
            # For now, use simple starting position - in full implementation would use reeb graph
            current_x = 0.0
            current_y = 0.0
            current_theta = phi[0] if phi else 0.0
            
            # Process each segment
            N = len(waypoints) - 1 if len(waypoints) > 1 else 0
            
            for i in range(N):
                if i >= len(time_segments):
                    break
                    
                segment = time_segments[i]
                arc_times = segment.get('arc', [])
                line_times = segment.get('line', [])
                
                # Arc segment
                if arc_times and i < len(r0) and i < len(phi_new):
                    arc_total_time = sum(arc_times)
                    if arc_total_time > 0 and abs(r0[i]) > 1e-6:
                        # Calculate arc parameters
                        radius = abs(r0[i])
                        delta_phi = phi_new[i] if i < len(phi_new) else 0
                        arc_length = radius * abs(delta_phi)
                        
                        # Discretize arc with simple approach
                        num_arc_points = max(1, int(arc_total_time / dt))
                        for j in range(num_arc_points):
                            t_arc = j * dt
                            progress = t_arc / arc_total_time if arc_total_time > 0 else 0
                            
                            # Simple arc discretization (circular motion)
                            angle_progress = delta_phi * progress
                            if r0[i] > 0:  # Left turn
                                arc_x = current_x + radius * (np.sin(current_theta + angle_progress) - np.sin(current_theta))
                                arc_y = current_y + radius * (np.cos(current_theta) - np.cos(current_theta + angle_progress))
                            else:  # Right turn
                                arc_x = current_x + radius * (np.sin(current_theta) - np.sin(current_theta + angle_progress))
                                arc_y = current_y + radius * (np.cos(current_theta + angle_progress) - np.cos(current_theta))
                            
                            arc_theta = current_theta + angle_progress
                            
                            # Calculate velocities for arc motion
                            if arc_total_time > 0:
                                angular_velocity = abs(delta_phi) / arc_total_time
                                linear_velocity = angular_velocity * radius
                            else:
                                angular_velocity = 0.0
                                linear_velocity = 0.0
                            
                            trajectory_points.append([
                                float(arc_x), 
                                float(arc_y), 
                                float(arc_theta), 
                                float(linear_velocity), 
                                float(angular_velocity)
                            ])
                            current_time += dt
                        
                        # Update position for next segment
                        if num_arc_points > 0:
                            current_x = arc_x
                            current_y = arc_y
                            current_theta = arc_theta
                
                # Line segment processing
                if line_times and i < len(l):
                    line_total_time = sum(line_times)
                    if line_total_time > 0 and l[i] > 1e-6:
                        line_length = l[i]
                        
                        # Discretize line segment
                        num_line_points = max(1, int(line_total_time / dt))
                        for j in range(num_line_points):
                            t_line = j * dt
                            progress = t_line / line_total_time if line_total_time > 0 else 0
                            
                            # Simple line discretization (straight line motion)
                            distance_progress = line_length * progress
                            line_x = current_x + distance_progress * np.cos(current_theta)
                            line_y = current_y + distance_progress * np.sin(current_theta)
                            line_theta = current_theta  # Constant angle for line
                            
                            # Calculate velocities for line motion
                            if line_total_time > 0:
                                linear_velocity = line_length / line_total_time
                                angular_velocity = 0.0
                            else:
                                linear_velocity = 0.0
                                angular_velocity = 0.0
                            
                            trajectory_points.append([
                                float(line_x), 
                                float(line_y), 
                                float(line_theta), 
                                float(linear_velocity), 
                                float(angular_velocity)
                            ])
                            current_time += dt
                        
                        # Update position for next segment
                        if num_line_points > 0:
                            current_x = line_x
                            current_y = line_y
                            # theta remains the same for line segments
            
            # Ensure we have at least some trajectory points
            if not trajectory_points:
                # Create a minimal trajectory as fallback
                trajectory_points = [
                    [0.0, 0.0, 0.0, 0.1, 0.0],
                    [0.1, 0.0, 0.0, 0.1, 0.0],
                    [0.2, 0.0, 0.0, 0.1, 0.0]
                ]
                print(f"[{self.name}] Generated minimal fallback trajectory with {len(trajectory_points)} points")
            else:
                print(f"[{self.name}] Generated {len(trajectory_points)} trajectory points from parameters")
            
            return trajectory_points
            
        except Exception as e:
            print(f"[{self.name}] Error in discretize_trajectory_from_parameters: {e}")
            traceback.print_exc()
            
            # Return a minimal fallback trajectory
            return [
                [0.0, 0.0, 0.0, 0.1, 0.0],
                [0.1, 0.0, 0.0, 0.1, 0.0],
                [0.2, 0.0, 0.0, 0.1, 0.0]
            ]

    def save_replanned_trajectory_parameters(self, replanned_trajectory, case, robot_id):
        """
        Save replanned trajectory parameters to file following the same pattern as Planning_pickup_simplified.py
        """
        try:
            output_dir = f'/root/workspace/data/{case}/'
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare data structure following the exact pattern from robot_0_trajectory_parameters_simple_maze.json
            save_data = {
                "robot_id": robot_id,
                "waypoints": replanned_trajectory.get('waypoints', []),
                "phi": replanned_trajectory.get('phi', []),
                "r0": replanned_trajectory.get('r0', []),
                "l": replanned_trajectory.get('l', []),
                "phi_new": replanned_trajectory.get('phi_new', []),
                "time_segments": replanned_trajectory.get('time_segments', []),
                "Flagb": replanned_trajectory.get('Flagb', []),
                "waypoint_positions": replanned_trajectory.get('waypoint_positions', []),
                "total_time": replanned_trajectory.get('total_time', 0.0)
            }
            
            # Add metadata following the exact pattern
            current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_data["metadata"] = {
                "robot_id": robot_id,
                "total_waypoints": len(replanned_trajectory.get('waypoints', [])),
                "total_segments": len(replanned_trajectory.get('time_segments', [])),
                "start_waypoint_idx": 0,
                "end_waypoint_idx": len(replanned_trajectory.get('waypoints', [])) - 1 if replanned_trajectory.get('waypoints', []) else 0,
                "save_timestamp": current_timestamp,
                "case": case,
                "N": len(replanned_trajectory.get('waypoints', [])),
                "replanned": True,
                "replan_timestamp": current_timestamp,
                "original_total_time": replanned_trajectory.get('metadata', {}).get('original_total_time', replanned_trajectory.get('total_time', 0.0)),
                "replanned_total_time": replanned_trajectory.get('total_time', 0.0),
                "detailed_optimization": replanned_trajectory.get('metadata', {}).get('detailed_optimization', True)
            }
            
            # Save replanned trajectory parameters
            output_file = f'{output_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}.json'
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"[{self.name}] ✓ Replanned trajectory parameters saved: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"[{self.name}] Error saving replanned trajectory parameters: {e}")
            traceback.print_exc()
            return None

    def generate_discrete_trajectory_from_replanned(self, case, robot_id, dt=0.1):
        """
        Generate discrete trajectory from replanned parameters
        Following the same pattern as generate_discrete_trajectories_from_replanned_data in Planning_pickup_simplified.py
        """
        try:
            print(f"[{self.name}] Generating discrete trajectory from replanned parameters...")
            
            # Load replanned trajectory parameters
            data_dir = f'/root/workspace/data/{case}/'
            replanned_file = f'{data_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}.json'
            
            if not os.path.exists(replanned_file):
                # Try original file if replanned doesn't exist
                replanned_file = f'{data_dir}robot_{robot_id}_trajectory_parameters_{case}.json'
                
            if not os.path.exists(replanned_file):
                print(f"[{self.name}] No replanned trajectory parameters found")
                return False
            
            with open(replanned_file, 'r') as f:
                trajectory_data = json.load(f)
            
            # Load reeb graph for waypoint coordinates
            graph_file = f'/root/workspace/data/Graph_new_{case}.json'
            if not os.path.exists(graph_file):
                print(f"[{self.name}] Reeb graph file not found: {graph_file}")
                return False
                
            with open(graph_file, 'r') as f:
                reeb_graph = json.load(f)
            
            # Extract trajectory components
            waypoints = trajectory_data.get('waypoints', [])
            phi = trajectory_data.get('phi', [])
            r0 = trajectory_data.get('r0', [])
            l = trajectory_data.get('l', [])
            phi_new = trajectory_data.get('phi_new', [])
            time_segments = trajectory_data.get('time_segments', [])
            Flagb = trajectory_data.get('Flagb', [])
            
            if not waypoints or not time_segments:
                print(f"[{self.name}] Invalid trajectory data structure")
                return False
            
            # Generate discrete trajectory using the same approach as Planning_pickup_simplified.py
            result = self.generate_single_robot_discrete_trajectory(
                robot_id=robot_id,
                waypoints=waypoints,
                phi=phi,
                r0=r0,
                l=l,
                phi_new=phi_new,
                time_segments=time_segments,
                Flagb=Flagb,
                reeb_graph=reeb_graph,
                dt=dt,
                save_dir=data_dir,
                case=case
            )
            
            if result:
                print(f"[{self.name}] Successfully generated discrete trajectory from replanned parameters")
                return True
            else:
                print(f"[{self.name}] Failed to generate discrete trajectory")
                return False
                
        except Exception as e:
            print(f"[{self.name}] Error generating discrete trajectory from replanned parameters: {e}")
            return False

    def replan_trajectory_simple_approach(self, trajectory_data, case, target_time, robot_id):
        """Simple fallback approach: proportionally scale trajectory times"""
        try:
            print(f"[{self.name}] === Simple Trajectory Approach (Fallback) ===")
            
            # Calculate current trajectory time
            time_segments = trajectory_data.get('time_segments', [])
            current_segment_times = []
            
            for segment in time_segments:
                segment_time = 0.0
                if 'arc' in segment and isinstance(segment['arc'], list):
                    segment_time += sum(segment['arc'])
                if 'line' in segment and isinstance(segment['line'], list):
                    segment_time += sum(segment['line'])
                current_segment_times.append(segment_time)
            
            current_total_time = sum(current_segment_times)
            
            # Determine a feasible target time
            min_feasible_time = current_total_time * 0.3  # Can't compress more than 70%
            feasible_target = max(target_time, min_feasible_time)
            
            print(f"[{self.name}] Current time: {current_total_time:.3f}s")
            print(f"[{self.name}] Requested target: {target_time:.3f}s")
            print(f"[{self.name}] Feasible target: {feasible_target:.3f}s")
            
            # Calculate scale factor
            scale_factor = feasible_target / current_total_time
            print(f"[{self.name}] Scale factor: {scale_factor:.3f}")
            
            # Create scaled trajectory data
            scaled_data = copy.deepcopy(trajectory_data)
            scaled_segment_times = []
            
            # Scale each segment proportionally
            for i, segment in enumerate(scaled_data.get('time_segments', [])):
                if i < len(current_segment_times):
                    scaled_time = current_segment_times[i] * scale_factor
                    scaled_segment_times.append(scaled_time)
                    
                    # Get original arc and line times
                    orig_arc_times = segment.get('arc', [])
                    orig_line_times = segment.get('line', [])
                    
                    # Scale arc times proportionally
                    if isinstance(orig_arc_times, list) and len(orig_arc_times) > 0:
                        arc_scale = scale_factor
                        segment['arc'] = [t * arc_scale for t in orig_arc_times]
                    
                    # Scale line times proportionally
                    if isinstance(orig_line_times, list) and len(orig_line_times) > 0:
                        line_scale = scale_factor
                        segment['line'] = [t * line_scale for t in orig_line_times]
            
            # Update total time
            scaled_total_time = sum(scaled_segment_times)
            scaled_data['total_time'] = scaled_total_time
            
            # Update metadata
            if 'metadata' in scaled_data:
                scaled_data['metadata']['replanned'] = True
                scaled_data['metadata']['replan_method'] = 'simple_proportional_scaling'
                scaled_data['metadata']['replan_timestamp'] = str(np.datetime64('now'))
                scaled_data['metadata']['original_total_time'] = current_total_time
                scaled_data['metadata']['replanned_total_time'] = scaled_total_time
                scaled_data['metadata']['scale_factor'] = scale_factor
                scaled_data['metadata']['requested_target_time'] = target_time
                scaled_data['metadata']['feasible_target_time'] = feasible_target
            
            print(f"[{self.name}] ✓ Simple approach successful!")
            print(f"[{self.name}]   Original → Scaled: {current_total_time:.3f}s → {scaled_total_time:.3f}s")
            
            # Save the scaled trajectory
            saved_file = self.save_replanned_trajectory_parameters(scaled_data, case, robot_id)
            
            if saved_file:
                # Generate discrete trajectory from scaled parameters
                discrete_success = self.generate_discrete_trajectory_from_replanned(case, robot_id, dt=0.1)
                
                if discrete_success:
                    print(f"[{self.name}] === Simple Approach Summary ===")
                    print(f"[{self.name}] Robot {robot_id} fallback results:")
                    print(f"[{self.name}]   Method: Simple proportional scaling")
                    print(f"[{self.name}]   Original → Replanned: {current_total_time:.3f}s → {scaled_total_time:.3f}s")
                    print(f"[{self.name}]   Requested target: {target_time:.3f}s")
                    print(f"[{self.name}]   Achieved time: {scaled_total_time:.3f}s")
                    print(f"[{self.name}]   Time reduction: {current_total_time - scaled_total_time:.3f}s")
                    return True
                else:
                    print(f"[{self.name}] Failed to generate discrete trajectory from simple approach")
                    return False
            else:
                print(f"[{self.name}] Failed to save simple approach trajectory")
                return False
                
        except Exception as e:
            print(f"[{self.name}] Simple approach failed: {e}")
            traceback.print_exc()
            return False
