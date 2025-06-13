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
                # Get target time from previous robot's pushing_estimated_time
                # Extract namespace number for determining previous robot
                namespace_match = re.search(r'turtlebot(\d+)', self.robot_namespace)
                namespace_number = int(namespace_match.group(1)) if namespace_match else 0
                
                if namespace_number == 0:
                    # turtlebot0 uses default 45s
                    raw_target_time = 45.0
                    print(f"[{self.name}] Using default target time for turtlebot0: {raw_target_time:.2f}s")
                else:
                    # turtlebotN gets pushing_estimated_time from turtlebot(N-1)
                    previous_robot = f'turtlebot{namespace_number - 1}'
                    raw_target_time = getattr(self.blackboard, f'{previous_robot}/pushing_estimated_time', 45.0)
                    print(f"[{self.name}] Getting target time from {previous_robot}: {raw_target_time:.2f}s")

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
        Load trajectory parameters and replan to achieve target time using optimization.
        Combines the functionality of the original replan_trajectory_to_target and 
        replan_trajectory_parameters_to_target_full functions.
        """
        print(f"[{self.name}] === Replanning Trajectory Parameters for {case} ===")
        print(f"[{self.name}] Robot ID: {robot_id}")
        print(f"[{self.name}] Target time: {target_time:.3f}s")
        
        # Load trajectory parameters
        data_dir = f'/root/workspace/data/{case}/'
        robot_file = f'{data_dir}robot_{robot_id}_trajectory_parameters_{case}.json'
        
        if not os.path.exists(robot_file):
            print(f"[{self.name}] Trajectory parameters file not found: {robot_file}")
            return False
            
        try:
            with open(robot_file, 'r') as f:
                trajectory_data = json.load(f)
            print(f"[{self.name}] Loaded trajectory parameters from {robot_file}")
        except Exception as e:
            print(f"[{self.name}] Error loading {robot_file}: {e}")
            return False

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
                # Generate discrete trajectory from replanned parameters using cubic spline inter
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
        Uses cubic spline interpolation for smooth trajectory generation with proper velocity calculations.
        
        Args:
            waypoints: List of waypoint indices
            phi: List of angles at waypoints
            r0: List of arc radii for each segment
            l: List of line lengths for each segment
            phi_new: List of adjusted angles
            time_segments: List of time segment dictionaries with 'arc' and 'line' times
            Flagb: List of flag values
            dt: Time step for discretization (default: 0.1s)
            
        Returns:
            List of trajectory points [x, y, theta, v, w] or None if failed
        """
        try:
            print(f"[{self.name}] Discretizing trajectory from parameters using cubic spline interpolation...")
            
            # Load reeb graph for waypoint coordinates
            graph_file = f'/root/workspace/data/Graph_new_{self.case}.json'
            if os.path.exists(graph_file):
                with open(graph_file, 'r') as f:
                    reeb_graph_json = json.load(f)
                
                # Convert JSON format to a simple access structure
                # JSON format: {"nodes": [[id, [x, y], parent, is_goal], ...]}
                # Convert coordinates from cm to m by dividing by 100
                reeb_graph = {}
                for node_data in reeb_graph_json['nodes']:
                    node_id = node_data[0]
                    coordinates = node_data[1]
                    # Convert from cm to m
                    coordinates_m = [coord / 100.0 for coord in coordinates]
                    reeb_graph[node_id] = {
                        'configuration': np.array(coordinates_m)
                    }
            
            N = len(waypoints) - 1  # Number of segments
            
            # Collect all discretized segments (original discrete points)
            all_times, all_xs, all_ys = [], [], []
            total_time = 0.0
            
            # Process each segment independently to get the original discrete points
            for i in range(N):
                # Discretize this segment directly without using external discretize_segment function
                try:
                    x_seg, y_seg, t_seg = self.discretize_single_segment(
                        i, waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph
                    )
                    
                    # Adjust time values to continue from previous segment
                    t_seg = [t + total_time for t in t_seg]
                    
                    # Add to the collections, but avoid duplicating the start point
                    if i > 0 and len(all_times) > 0:
                        all_times.extend(t_seg[1:])  # Skip the first point
                        all_xs.extend(x_seg[1:])
                        all_ys.extend(y_seg[1:])
                    else:
                        all_times.extend(t_seg)
                        all_xs.extend(x_seg)
                        all_ys.extend(y_seg)
                    
                    # Update total time for next segment
                    if len(t_seg) > 0:
                        total_time = t_seg[-1]
                        
                except Exception as e:
                    print(f"[{self.name}] Error discretizing segment {i}: {e}")
                    # Continue with next segment if one fails
            
            # Create uniform time grid for interpolation
            t_uniform = np.arange(0, total_time, dt)
            x_uniform = np.zeros_like(t_uniform)
            y_uniform = np.zeros_like(t_uniform)
            
            # Process each segment independently with its own spline
            segment_boundaries = [0]
            current_time = 0.0
            segment_times = []
            segment_xs = []
            segment_ys = []
            
            # Collect boundary indices for each segment
            for i in range(N):
                # Discretize this segment
                try:
                    x_seg, y_seg, t_seg = self.discretize_single_segment(
                        i, waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph
                    )
                    
                    # Store segment data with adjusted time values
                    t_adjusted = [t + current_time for t in t_seg]
                    segment_times.append(t_adjusted)
                    segment_xs.append(x_seg)
                    segment_ys.append(y_seg)
                    
                    # Update current time
                    if t_seg:
                        current_time += t_seg[-1]
                        segment_boundaries.append(len(all_times))
                        
                except Exception as e:
                    print(f"[{self.name}] Error discretizing segment {i} for spline: {e}")
                    # Add empty segment to maintain indexing
                    segment_times.append([])
                    segment_xs.append([])
                    segment_ys.append([])
            
            # Interpolate each segment separately and combine using cubic splines
            from scipy.interpolate import CubicSpline
            
            for i in range(N):
                # Skip empty segments
                if len(segment_times[i]) < 2:
                    continue
                    
                # Create spline for this segment
                cs_x_seg = CubicSpline(segment_times[i], segment_xs[i])
                cs_y_seg = CubicSpline(segment_times[i], segment_ys[i])
                
                # Determine which portion of t_uniform corresponds to this segment
                t_min = segment_times[i][0]
                t_max = segment_times[i][-1]
                mask = (t_uniform >= t_min) & (t_uniform <= t_max)
                
                # Apply spline for this segment's time range
                t_seg_uniform = t_uniform[mask]
                if len(t_seg_uniform) > 0:
                    x_uniform[mask] = cs_x_seg(t_seg_uniform)
                    y_uniform[mask] = cs_y_seg(t_seg_uniform)
            
            # Calculate orientations, velocities and angular velocities
            thetas = []
            velocities = []
            angular_velocities = []
            
            # Process each time point individually
            for i, t in enumerate(t_uniform):
                # Determine which segment this time point belongs to
                segment_idx = None
                for j in range(len(segment_times)):
                    if segment_times[j] and segment_times[j][0] <= t <= segment_times[j][-1]:
                        segment_idx = j
                        break
                
                if segment_idx is not None:
                    # Get splines for this segment
                    cs_x_seg = CubicSpline(segment_times[segment_idx], segment_xs[segment_idx])
                    cs_y_seg = CubicSpline(segment_times[segment_idx], segment_ys[segment_idx])
                    
                    # First derivatives for orientation and velocity
                    dx_dt = cs_x_seg(t, 1)
                    dy_dt = cs_y_seg(t, 1)
                    
                    # Calculate orientation
                    theta = np.arctan2(dy_dt, dx_dt)
                    thetas.append(float(theta))
                    
                    # Calculate linear velocity
                    velocity = np.sqrt(dx_dt**2 + dy_dt**2)
                    velocities.append(float(velocity))
                    
                    # Calculate angular velocity using second derivatives
                    d2x_dt2 = cs_x_seg(t, 2)
                    d2y_dt2 = cs_y_seg(t, 2)
                    
                    denominator = dx_dt**2 + dy_dt**2
                    if denominator > 1e-10:
                        angular_velocity = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / denominator
                    else:
                        angular_velocity = 0
                    
                    angular_velocities.append(float(angular_velocity))
                else:
                    # Default values for points outside segments
                    thetas.append(0.0)
                    velocities.append(0.0)
                    angular_velocities.append(0.0)
            
            # Create trajectory points as [x, y, theta, v, w]
            trajectory_points = []
            for i in range(len(t_uniform)):
                point = [float(x_uniform[i]), float(y_uniform[i]), float(thetas[i]), 
                        float(velocities[i]), float(angular_velocities[i])]
                trajectory_points.append(point)
            
            print(f"[{self.name}] Generated {len(trajectory_points)} trajectory points using cubic spline interpolation")
            print(f"[{self.name}] Total trajectory time: {total_time:.3f}s")
            
            return trajectory_points
            
        except Exception as e:
            print(f"[{self.name}] Error in discretize_trajectory_from_parameters: {e}")
            traceback.print_exc()
            
            # Return a minimal fallback trajectory
            print(f"[{self.name}] Falling back to minimal trajectory")
            return [
                [0.0, 0.0, 0.0, 0.1, 0.0],
                [0.1, 0.0, 0.0, 0.1, 0.0],
                [0.2, 0.0, 0.0, 0.1, 0.0]
            ]

    def save_replanned_trajectory_parameters(self, replanned_trajectory, case, robot_id):
        """
        Save replanned trajectory parameters to file preserving the original structure
        and only replacing the optimized time_segments and total_time
        """
        try:
            output_dir = f'/root/workspace/data/{case}/'
            os.makedirs(output_dir, exist_ok=True)
            
            # Load original trajectory parameters to preserve the exact structure
            original_file = f'{output_dir}robot_{robot_id}_trajectory_parameters_{case}.json'
            if os.path.exists(original_file):
                with open(original_file, 'r') as f:
                    save_data = json.load(f)
                print(f"[{self.name}] Loaded original structure from {original_file}")
            else:
                # Fallback to creating new structure if original doesn't exist
                save_data = {
                    "robot_id": robot_id,
                    "waypoints": replanned_trajectory.get('waypoints', []),
                    "phi": replanned_trajectory.get('phi', []),
                    "r0": replanned_trajectory.get('r0', []),
                    "l": replanned_trajectory.get('l', []),
                    "phi_new": replanned_trajectory.get('phi_new', []),
                    "time_segments": [],
                    "Flagb": replanned_trajectory.get('Flagb', []),
                    "waypoint_positions": replanned_trajectory.get('waypoint_positions', []),
                    "total_time": 0.0
                }
                print(f"[{self.name}] Creating new structure (original not found)")
            
            # Only replace the optimized time_segments and total_time from replanned_trajectory
            save_data["time_segments"] = replanned_trajectory.get('time_segments', [])
            save_data["total_time"] = replanned_trajectory.get('total_time', 0.0)
            
            # Update metadata to indicate this is a replanned version
            if "metadata" not in save_data:
                save_data["metadata"] = {}
            
            save_data["metadata"]["replanned"] = True
            save_data["metadata"]["replan_timestamp"] = str(np.datetime64('now'))
            if 'original_total_time' in replanned_trajectory.get('metadata', {}):
                save_data["metadata"]["original_total_time"] = replanned_trajectory['metadata']['original_total_time']
            save_data["metadata"]["replanned_total_time"] = replanned_trajectory.get('total_time', 0.0)
            save_data["metadata"]["detailed_optimization"] = True
        
            # Save replanned trajectory parameters
            output_file = f'{output_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}.json'
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"[{self.name}] ✓ Replanned trajectory parameters saved: {output_file}")
            print(f"[{self.name}]   Preserved original structure, updated time_segments and total_time")
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
    
    def discretize_single_segment(self, segment_idx, waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph):
        """
        Discretize a single major segment (arc+line) independently using just the endpoints of each subsegment.
        Simplified version that works directly with JSON graph structure.
        
        Args:
            segment_idx: Index of the segment to discretize (0 to N-2)
            waypoints: List of waypoint indices in the reeb graph
            phi: List of angles at each waypoint
            r0: List of arc radii for each segment
            l: List of line lengths for each segment
            phi_new: List of adjusted angles accounting for flag values
            time_segments: List of dictionaries with 'arc' and 'line' time values for each segment
            Flagb: List of flag values for each waypoint
            reeb_graph: Dictionary containing waypoint coordinates
            
        Returns:
            x_discrete: List of x-coordinates for the discretized segment
            y_discrete: List of y-coordinates for the discretized segment
            t_discrete: List of time points for the discretized segment
        """
        # Extract segment info
        i = segment_idx
        seg_times = time_segments[i]
        
        # Get waypoint position from JSON structure - coordinates already converted to meters
        waypoint_id = waypoints[i]
        pos = reeb_graph[waypoint_id]['configuration']  # Already in meters
        angle = phi[i] + Flagb[i]*np.pi/2
        
        # Separate arc and line parts
        arc_times = seg_times.get('arc', [])
        line_times = seg_times.get('line', [])
        
        x_discrete, y_discrete, t_discrete = [], [], []
        t_curr = 0.0
        
        # Starting point is always included
        x_discrete.append(pos[0])
        y_discrete.append(pos[1])
        t_discrete.append(t_curr)
        
        # Process arc segment (if it exists)
        if arc_times and len(arc_times) > 0:
            dphi = phi[i+1] - phi_new[i]
            r = r0[i]
            # Calculate center of arc
            cx = pos[0] - r*np.cos(angle + np.pi/2)
            cy = pos[1] - r*np.sin(angle + np.pi/2)
            n_arc = len(arc_times)
            
            for j in range(n_arc):
                t_curr += arc_times[j]
                # Calculate position on arc
                theta_curr = angle + (j+1) * dphi / n_arc
                x_curr = cx + r * np.cos(theta_curr + np.pi/2)
                y_curr = cy + r * np.sin(theta_curr + np.pi/2)
                
                x_discrete.append(x_curr)
                y_discrete.append(y_curr)
                t_discrete.append(t_curr)
            
            # Update position and angle for line segment
            pos = np.array([x_discrete[-1], y_discrete[-1]])
            angle = phi[i+1]
        
        # Process line segment (if it exists)
        if line_times and len(line_times) > 0:
            line_length = l[i]
            line_direction = np.array([np.cos(angle), np.sin(angle)])
            n_line = len(line_times)
            
            for j in range(n_line):
                t_curr += line_times[j]
                # Calculate position along line
                distance = (j+1) * line_length / n_line
                x_curr = pos[0] + distance * line_direction[0]
                y_curr = pos[1] + distance * line_direction[1]
                
                x_discrete.append(x_curr)
                y_discrete.append(y_curr)
                t_discrete.append(t_curr)
        
        return x_discrete, y_discrete, t_discrete
