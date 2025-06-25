import numpy as np
import casadi as ca
import json
import matplotlib.pyplot as plt
import copy
import os
import traceback
import time
from scipy.interpolate import CubicSpline

# Import for reeb graph loading
import sys
sys.path.append('/root/workspace/src/Replanning/scripts')
from Graph import load_reeb_graph_from_file

# Differential drive constants (from Planning_deltaT.py)
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

def load_trajectory_parameters_individual(case, robot_id=None):
    """
    Load trajectory parameters from individual robot trajectory parameter files
        
    Returns:
        Dictionary containing trajectory data for each robot
    """
    data_dir = f'/root/workspace/data/{case}/'
    
    robot_file = f'{data_dir}robot_{robot_id}_trajectory_parameters_{case}.json'
        
    if os.path.exists(robot_file):
        print(f"Loading trajectory parameters for Robot {robot_id}")
        with open(robot_file, 'r') as f:
            robot_data = json.load(f)
            trajectory_data = robot_data
    else:
        print(f"Warning: Robot {robot_id} trajectory file not found: {robot_file}")
        trajectory_data = None
    
    return trajectory_data

def replan_trajectory_parameters_to_target(case, target_time, robot_id, save_results=True):
    """
    Load trajectory parameters from individual robot file and replan to achieve target time
    
    Args:
        case: Case name (e.g., "simple_maze")
        target_time: Target total time for the robot (seconds)
        robot_id: Robot ID
        save_results: Whether to save replanned results
        
    Returns:
        Dictionary with replanning results for the specified robot
    """
    print(f"=== Replanning Trajectory Parameters for {case} ===")
    print(f"Robot ID: {robot_id}")
    print(f"Target time: {target_time:.3f}s")
    
    # Load trajectory parameters for the specific robot
    print(f"\n1. Loading trajectory parameters for Robot {robot_id}...")
    trajectory_data = load_trajectory_parameters_individual(case, robot_id)
    
    if not trajectory_data:
        print(f"No trajectory parameter data found for Robot {robot_id}!")
        return None
    
    print(f"✓ Loaded trajectory data for Robot {robot_id}")
    
    
    robot_data = trajectory_data
    
    # Calculate current time
    print(f"\n2. Analyzing current trajectory...")
    current_robot_time = robot_data.get('total_time', 0)
    print(f"   Current time: {current_robot_time:.3f}s")
    print(f"   Target time: {target_time:.3f}s")
    
    # Check if replanning is needed based on conditional logic
    if current_robot_time <= target_time:
        print(f"\n✓ Current time ({current_robot_time:.3f}s) is already ≤ target time ({target_time:.3f}s)")
        print("   Using original trajectory data without optimization")
        
        # Create result structure with original data
        optimization_results = {
            'original_total_time': current_robot_time,
            'target_time': target_time,
            'optimized_total_time': current_robot_time,  # No change
            'segment_times': [robot_data.get('total_time', 0)],  # Use original total time as single segment
            'deviation': abs(current_robot_time - target_time),
            'improvement': 0.0,  # No improvement since no optimization
            'optimization_skipped': True
        }
        
        # Use original trajectory as "replanned" (no changes)
        replanned_trajectory = robot_data.copy()
        
        # Create result structure
        result = {
            'robot_id': robot_id,
            'original_trajectory': robot_data,
            'replanned_trajectory': replanned_trajectory,
            'optimization_results': optimization_results
        }
        
        # Save results if requested (saves original data as replanned)
        if save_results:
            print(f"\n4. Saving original data as replanned results...")
            save_single_robot_trajectory_parameters(result, case, robot_id)
            
            # Also copy the original discrete trajectory file directly
            print(f"\n5. Copying original discrete trajectory file...")
            original_discrete_file = f'/root/workspace/data/{case}/tb{robot_id}_Trajectory.json'
            replanned_discrete_file = f'/root/workspace/data/{case}/tb{robot_id}_Trajectory_replanned.json'
            
            if os.path.exists(original_discrete_file):
                import shutil
                shutil.copy2(original_discrete_file, replanned_discrete_file)
                print(f"✓ Copied original discrete trajectory: {original_discrete_file}")
                print(f"  → {replanned_discrete_file}")
            else:
                print(f"⚠ Original discrete trajectory file not found: {original_discrete_file}")
        
        # Print summary
        print(f"\n=== Replanning Summary ===")
        print(f"Robot {robot_id} results:")
        print(f"  Original → Replanned: {optimization_results['original_total_time']:.3f}s → {optimization_results['optimized_total_time']:.3f}s")
        print(f"  Target: {optimization_results['target_time']:.3f}s")
        print(f"  Deviation: {optimization_results['deviation']:.3f}s")
        print(f"  Improvement: {optimization_results['improvement']:.3f}s")
        print(f"  Status: Optimization skipped (current time already meets target)")
        
        return result
    
    print(f"\n⚠️  Current time ({current_robot_time:.3f}s) > target time ({target_time:.3f}s)")
    print("   Proceeding with trajectory optimization...")
    
    # Extract time segments
    time_segments = robot_data.get('time_segments', [])
    num_segments = len(time_segments)
    
    if num_segments == 0:
        print(f"  Warning: No time segments found for Robot {robot_id}")
        return None
    
    print(f"   Number of segments: {num_segments}")
    
    # Calculate current segment times
    current_segment_times = []
    for segment in time_segments:
        segment_time = 0.0
        if 'arc' in segment and isinstance(segment['arc'], list):
            segment_time += sum(segment['arc'])
        if 'line' in segment and isinstance(segment['line'], list):
            segment_time += sum(segment['line'])
        current_segment_times.append(segment_time)
    
    # Replan using CasADi optimization
    print(f"\n3. Optimizing trajectory times...")
    
    try:
        opti = ca.Opti()
        
        # Extract robot trajectory parameters for constraint calculations
        waypoints = robot_data.get('waypoints', [])
        phi = np.array(robot_data.get('phi', []))
        r0 = np.array(robot_data.get('r0', []))
        l = np.array(robot_data.get('l', []))
        phi_new = np.array(robot_data.get('phi_new', []))
        Flagb = robot_data.get('Flagb', [])
        
        # Create decision variables for detailed time segments
        delta_t_arcs = []
        delta_t_lines = []
        segment_arc_indices = []
        segment_line_indices = []
        all_vars_flat = []
        
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
                
                # Calculate number of subsegments
                N_arc = max(1, int(arc_length / Deltal)) if arc_length > 0.03 else 0
                N_line = max(1, int(line_length / Deltal)) if line_length > 0.03 else 0
                
                # Create arc variables
                if N_arc > 0 and len(orig_arc_times) > 0:
                    arc_vars = [opti.variable() for _ in range(N_arc)]
                    delta_t_arcs.append(arc_vars)
                    segment_arc_indices.append(i)
                    all_vars_flat.extend(arc_vars)
                else:
                    delta_t_arcs.append([])
                
                # Create line variables  
                if N_line > 0 and len(orig_line_times) > 0:
                    line_vars = [opti.variable() for _ in range(N_line)]
                    delta_t_lines.append(line_vars)
                    segment_line_indices.append(i)
                    all_vars_flat.extend(line_vars)
                else:
                    delta_t_lines.append([])
        
        # Optimized constraint formulation - batch constraints for efficiency
        all_accelerations = []
        
        # Pre-compute common values once to avoid repeated calculations
        max_segments = len(time_segments)
        r0_valid = np.array([abs(r0[i]) if i < len(r0) else 0 for i in range(max_segments)])
        l_valid = np.array([l[i] if i < len(l) else 0 for i in range(max_segments)])
        
        # Cache angular limits to avoid repeated calculations
        angular_limits = {}
        for i in range(max_segments):
            if r0_valid[i] > 0:
                angular_limits[i] = {
                    'w_max': calculate_angular_velocity_limit(r0_valid[i]),
                    'aw_max': calculate_angular_acceleration_limit(r0[i])
                }
        
        # Batch time bounds for all variables at once
        for i in range(max_segments):
            # Arc time bounds (vectorized)
            if i < len(delta_t_arcs) and delta_t_arcs[i]:
                arc_vars = ca.vertcat(*delta_t_arcs[i])
                opti.subject_to(opti.bounded(0.20, arc_vars, 10.0))
            
            # Line time bounds (vectorized)  
            if i < len(delta_t_lines) and delta_t_lines[i]:
                line_vars = ca.vertcat(*delta_t_lines[i])
                opti.subject_to(opti.bounded(0.1, line_vars, 5.0))
        
        # Optimized velocity and acceleration constraints
        for i in range(max_segments):
            if i >= len(r0) or i >= len(l) or i >= len(phi):
                continue
                
            # Arc constraints - optimized processing
            if i < len(delta_t_arcs) and delta_t_arcs[i] and r0_valid[i] > 0:
                delta_phi = phi[i+1] - phi_new[i] if i+1 < len(phi) else 0
                arc_length = r0_valid[i] * abs(delta_phi)
                arc_segment_length = arc_length / len(delta_t_arcs[i])
                
                if arc_segment_length > 0:
                    # Vectorized angular velocity constraints
                    arc_vars = ca.vertcat(*delta_t_arcs[i])
                    omega_terms = arc_segment_length / r0_valid[i] / arc_vars
                    opti.subject_to(omega_terms <= angular_limits[i]['w_max'])
                    
                    # Simplified acceleration constraints (consecutive pairs only)
                    if len(delta_t_arcs[i]) > 1:
                        for j in range(1, len(delta_t_arcs[i])):
                            dt_prev, dt_curr = delta_t_arcs[i][j-1], delta_t_arcs[i][j]
                            v_diff = arc_segment_length * (1/dt_curr - 1/dt_prev)
                            t_avg = (dt_prev + dt_curr) / 2
                            alpha = v_diff / (r0_valid[i] * t_avg)
                            
                            opti.subject_to(opti.bounded(-angular_limits[i]['aw_max'], alpha, angular_limits[i]['aw_max']))
                            all_accelerations.append(alpha)
                    
                    # Initial state constraints for first segments
                    if i == 0 or (i < len(Flagb) and Flagb[i] != 0):
                        min_t_initial = np.sqrt(2*arc_segment_length / angular_limits[i]['aw_max'] / r0_valid[i])
                        opti.subject_to(delta_t_arcs[i][0] >= min_t_initial)
                    
                    # Ensure stopping at the end of trajectory for arc segments (final segment only)
                    # Only apply if this is the last segment AND there are no line segments after it
                    if i == len(time_segments)-1 and (i >= len(delta_t_lines) or not delta_t_lines[i]):
                        # Minimum time for final arc segment to allow deceleration to zero
                        t_min_arc = np.sqrt(2*arc_segment_length / angular_limits[i]['aw_max'] / r0_valid[i])
                        opti.subject_to(delta_t_arcs[i][-1] >= t_min_arc)
                        opti.subject_to(delta_t_arcs[i][-1] <= 10.0)
            
            # Line constraints - optimized processing  
            if i < len(delta_t_lines) and delta_t_lines[i] and l_valid[i] > 0:
                line_segment_length = l_valid[i] / len(delta_t_lines[i])
                
                # Vectorized velocity constraints
                line_vars = ca.vertcat(*delta_t_lines[i])
                velocity_terms = line_segment_length / line_vars
                opti.subject_to(velocity_terms <= v_max)
                
                # Simplified acceleration constraints (consecutive pairs only)
                if len(delta_t_lines[i]) > 1:
                    for j in range(1, len(delta_t_lines[i])):
                        dt_prev, dt_curr = delta_t_lines[i][j-1], delta_t_lines[i][j]
                        accel_term = (dt_curr**2 - dt_prev**2) / (dt_prev * dt_curr)
                        constraint_bound = a_max / (2 * line_segment_length)
                        opti.subject_to(opti.bounded(-constraint_bound, accel_term, constraint_bound))
                        
                        # Store simplified acceleration for penalty
                        a_simple = line_segment_length * (1/dt_curr - 1/dt_prev) / ((dt_curr + dt_prev)/2)
                        all_accelerations.append(a_simple)
                
                # Arc-to-line continuity (simplified)
                if i < len(delta_t_arcs) and delta_t_arcs[i] and r0_valid[i] > 0:
                    arc_length = r0_valid[i] * abs(phi[i+1] - phi_new[i] if i+1 < len(phi) else 0)
                    arc_seg_len = arc_length / len(delta_t_arcs[i]) if arc_length > 0 else 0
                    
                    if arc_seg_len > 0:
                        v_arc_end = arc_seg_len / delta_t_arcs[i][-1]
                        v_line_start = line_segment_length / delta_t_lines[i][0]
                        t_transition = (delta_t_arcs[i][-1] + delta_t_lines[i][0]) / 2
                        a_transition = (v_line_start - v_arc_end) / t_transition
                        opti.subject_to(opti.bounded(-a_max, a_transition, a_max))
                        all_accelerations.append(a_transition)
                
                # Ensure stopping at the end of trajectory (final segment only)
                if i == len(time_segments)-1:
                    # Minimum time for final line segment to allow deceleration to zero
                    t_min = np.sqrt(2*line_segment_length/a_max)
                    opti.subject_to(delta_t_lines[i][-1] >= t_min)
                    opti.subject_to(delta_t_lines[i][-1] <= 5.0)
            
            # Calculate total time for detailed optimization
            total_time = 0
            for i in range(len(time_segments)):
                if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                    total_time += ca.sum1(ca.vertcat(*delta_t_arcs[i]))
                if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                    total_time += ca.sum1(ca.vertcat(*delta_t_lines[i]))
        
        # Objective: minimize deviation from target time + acceleration penalty
        objective = (total_time - target_time)**2
        
        
        # Add acceleration penalty for smoother trajectories (from Planning_deltaT.py)
        if 'all_accelerations' in locals() and all_accelerations:
            acceleration_penalty_weight = 1000.0  # Same as Planning_deltaT.py
            accel_terms_vector = ca.vertcat(*all_accelerations)
            objective += acceleration_penalty_weight * ca.sumsqr(accel_terms_vector)
        
        opti.minimize(objective)
        
        # Total time constraints (allow 20% flexibility for more robust optimization)
        opti.subject_to(total_time >= target_time * 0.8)
        opti.subject_to(total_time <= target_time * 1.2)
        

        # Detailed subsegment optimization - use loaded data and physics-based estimates
        for i in range(len(time_segments)):
            segment = time_segments[i]
            
            # Set initial values for arc variables using loaded data and physical constraints
            if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                orig_arc_times = segment.get('arc', [])
                if orig_arc_times:
                    # Calculate average velocity from the loaded data for initial guess
                    delta_phi = phi[i+1] - phi_new[i] if i+1 < len(phi) else 0
                    arc_length = abs(r0[i] * delta_phi) if abs(r0[i]) > 0 else 0
                    arc_segment_length = arc_length / len(delta_t_arcs[i]) if len(delta_t_arcs[i]) > 0 else 0
                    
                    # Use loaded times as baseline but ensure they respect physical limits
                    for j, arc_var in enumerate(delta_t_arcs[i]):
                        if j < len(orig_arc_times):
                            # Use loaded time but ensure it meets minimum time constraints
                            loaded_time = orig_arc_times[j]
                            
                            # Check angular velocity constraint
                            if abs(r0[i]) > 0 and arc_segment_length > 0:
                                omega_c = arc_segment_length / abs(r0[i]) / loaded_time
                                w_max_arc = calculate_angular_velocity_limit(abs(r0[i]))
                                
                                # If loaded time violates velocity constraint, adjust it
                                if omega_c > w_max_arc:
                                    min_time_vel = arc_segment_length / abs(r0[i]) / w_max_arc
                                    initial_time = max(loaded_time, min_time_vel * 1.1)  # 10% safety margin
                                else:
                                    initial_time = max(0.05, loaded_time)
                            else:
                                initial_time = max(0.05, loaded_time)
                            
                            opti.set_initial(arc_var, initial_time)
                        else:
                            # For extra variables beyond loaded data, use physics-based estimate
                            if abs(r0[i]) > 0 and arc_segment_length > 0:
                                # Conservative initial guess based on 50% of max angular velocity
                                w_conservative = calculate_angular_velocity_limit(abs(r0[i])) * 0.5
                                initial_time = arc_segment_length / abs(r0[i]) / w_conservative
                            else:
                                initial_time = 0.5
                            opti.set_initial(arc_var, max(0.05, initial_time))
                else:
                    # No loaded arc data, use physics-based estimates
                    delta_phi = phi[i+1] - phi_new[i] if i+1 < len(phi) else 0
                    arc_length = abs(r0[i] * delta_phi) if abs(r0[i]) > 0 else 0
                    arc_segment_length = arc_length / len(delta_t_arcs[i]) if len(delta_t_arcs[i]) > 0 else 0
                    
                    for arc_var in delta_t_arcs[i]:
                        if abs(r0[i]) > 0 and arc_segment_length > 0:
                            # Conservative initial guess based on 50% of max angular velocity
                            w_conservative = calculate_angular_velocity_limit(abs(r0[i])) * 0.5
                            initial_time = arc_segment_length / abs(r0[i]) / w_conservative
                        else:
                            initial_time = 0.5
                        opti.set_initial(arc_var, max(0.05, initial_time))
                
                # Ensure stopping at the end of trajectory for arc segments (final segment only)
                # Only apply if this is the last segment AND there are no line segments after it
                if i == len(time_segments)-1 and len(delta_t_arcs[i]) > 0 and (i >= len(delta_t_lines) or not delta_t_lines[i]):
                    # Minimum time for final arc segment
                    delta_phi = phi[i+1] - phi_new[i] if i+1 < len(phi) else 0
                    arc_length = abs(r0[i] * delta_phi) if abs(r0[i]) > 0 else 0
                    arc_segment_length = arc_length / len(delta_t_arcs[i]) if len(delta_t_arcs[i]) > 0 else 0
                    if arc_segment_length > 0 and abs(r0[i]) > 0:
                        aw_max_arc = calculate_angular_acceleration_limit(abs(r0[i]))
                        t_min = np.sqrt(2*arc_segment_length / aw_max_arc / abs(r0[i]))
                        # Ensure the last arc segment has enough time to stop
                        opti.set_initial(delta_t_arcs[i][-1], max(t_min * 1.2, 0.5))
            
            # Set initial values for line variables using loaded data and physical constraints
            if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                orig_line_times = segment.get('line', [])
                if orig_line_times:
                    # Calculate line segment parameters
                    line_length = l[i] if i < len(l) else 0
                    line_segment_length = line_length / len(delta_t_lines[i]) if len(delta_t_lines[i]) > 0 else 0
                    
                    # Use loaded times as baseline but ensure they respect physical limits
                    for j, line_var in enumerate(delta_t_lines[i]):
                        if j < len(orig_line_times):
                            # Use loaded time but ensure it meets minimum time constraints
                            loaded_time = orig_line_times[j]
                            
                            # Check velocity constraint
                            if line_segment_length > 0:
                                velocity = line_segment_length / loaded_time
                                
                                # If loaded time violates velocity constraint, adjust it
                                if velocity > v_max:
                                    min_time_vel = line_segment_length / v_max
                                    initial_time = max(loaded_time, min_time_vel * 1.1)  # 10% safety margin
                                else:
                                    initial_time = max(0.05, loaded_time)
                            else:
                                initial_time = max(0.05, loaded_time)
                            
                            opti.set_initial(line_var, initial_time)
                        else:
                            # For extra variables beyond loaded data, use physics-based estimate
                            if line_segment_length > 0:
                                # Conservative initial guess based on 50% of max velocity
                                initial_time = line_segment_length / (v_max * 0.5)
                            else:
                                initial_time = 0.5
                            opti.set_initial(line_var, max(0.05, initial_time))
                else:
                    # No loaded line data, use physics-based estimates
                    line_length = l[i] if i < len(l) else 0
                    line_segment_length = line_length / len(delta_t_lines[i]) if len(delta_t_lines[i]) > 0 else 0
                    
                    for line_var in delta_t_lines[i]:
                        if line_segment_length > 0:
                            # Conservative initial guess based on 50% of max velocity
                            initial_time = line_segment_length / (v_max * 0.5)
                        else:
                            initial_time = 0.5
                        opti.set_initial(line_var, max(0.05, initial_time))
                
                # Ensure stopping at the end of trajectory (final segment only)
                if i == len(time_segments)-1:
                    # Minimum time for final line segment (similar to Planning_deltaT.py)
                    line_length = l[i] if i < len(l) else 0
                    line_segment_length = line_length / len(delta_t_lines[i]) if len(delta_t_lines[i]) > 0 else 0
                    if line_segment_length > 0:
                        t_min = np.sqrt(2*line_segment_length/a_max)
                        # Ensure the last line segment has enough time to stop
                        opti.set_initial(delta_t_lines[i][-1], max(t_min * 1.2, 0.5))
            if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                orig_arc_times = segment.get('arc', [])
                if orig_arc_times and len(orig_arc_times) == len(delta_t_arcs[i]):
                    # Use exact loaded arc times WITHOUT scaling
                    for j, arc_var in enumerate(delta_t_arcs[i]):
                        initial_value = max(0.05, orig_arc_times[j])  # No scaling
                        opti.set_initial(arc_var, initial_value)
                elif orig_arc_times:
                    # Distribute total arc time evenly if mismatch in count
                    total_arc_time = sum(orig_arc_times)  # No scaling
                    avg_arc_time = total_arc_time / len(delta_t_arcs[i])
                    for arc_var in delta_t_arcs[i]:
                        opti.set_initial(arc_var, max(0.05, avg_arc_time))
                else:
                    # Default fallback
                    for arc_var in delta_t_arcs[i]:
                        opti.set_initial(arc_var, 0.5)
            
            # Use actual line times from loaded data WITHOUT scaling
            if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                orig_line_times = segment.get('line', [])
                if orig_line_times and len(orig_line_times) == len(delta_t_lines[i]):
                    # Use exact loaded line times WITHOUT scaling
                    for j, line_var in enumerate(delta_t_lines[i]):
                        initial_value = max(0.05, orig_line_times[j])  # No scaling
                        opti.set_initial(line_var, initial_value)
                elif orig_line_times:
                    # Distribute total line time evenly if mismatch in count
                    total_line_time = sum(orig_line_times)  # No scaling
                    avg_line_time = total_line_time / len(delta_t_lines[i])
                    for line_var in delta_t_lines[i]:
                        opti.set_initial(line_var, max(0.05, avg_line_time))
                else:
                    # Default fallback
                    for line_var in delta_t_lines[i]:
                        opti.set_initial(line_var, 0.5)
        
        # Optimized solver settings for faster convergence
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 1500,              # Reduced from 3000
            'ipopt.acceptable_tol': 1e-3,        # Relaxed from 1e-4
            'ipopt.tol': 1e-5,                   # Added main tolerance
            'ipopt.dual_inf_tol': 1e-3,          # Added dual tolerance
            'ipopt.constr_viol_tol': 1e-3,       # Added constraint violation tolerance
            'ipopt.acceptable_iter': 15,         # Accept solution after 15 iterations at acceptable tolerance
            'ipopt.warm_start_init_point': 'yes' # Enable warm start for faster convergence
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
        replanned_trajectory = create_detailed_replanned_trajectory(
            robot_data, delta_t_arcs, delta_t_lines, sol, optimized_total_time
        )
        
        
        optimization_results = {
            'original_total_time': current_robot_time,
            'target_time': target_time,
            'optimized_total_time': optimized_total_time,
            'segment_times': T_opt_list,
            'deviation': abs(optimized_total_time - target_time),
            'improvement': current_robot_time - optimized_total_time
        }
        
        print(f"  ✓ Optimization successful!")
        print(f"    Optimized time: {optimized_total_time:.3f}s")
        print(f"    Deviation: {abs(optimized_total_time - target_time):.3f}s")
        print(f"    Improvement: {current_robot_time - optimized_total_time:.3f}s")
        
    except Exception as e:
        print(f"  ✗ Optimization failed: {e}")
        print(f"  ⚠️  Warning: Could not optimize trajectory parameters for Robot {robot_id}")
        print(f"  ⚠️  Reason: {str(e)}")
        return None
    
    # Create result structure
    result = {
        'robot_id': robot_id,
        'original_trajectory': robot_data,
        'replanned_trajectory': replanned_trajectory,
        'optimization_results': optimization_results
    }
    
    # Save results if requested
    if save_results:
        print(f"\n4. Saving replanned results...")
        save_single_robot_trajectory_parameters(result, case, robot_id)
        
        # Also generate discrete trajectory from replanned parameters
        print(f"\n5. Generating discrete trajectory...")
        try:
            # Load reeb graph for discrete trajectory generation
            graph_file = f'/root/workspace/data/Graph_new_{case}.json'
            if os.path.exists(graph_file):
                sys.path.append('/root/workspace/src/Replanning/scripts')
                from Graph import load_reeb_graph_from_file
                
                # Change to the data directory so load_reeb_graph_from_file can find the file
                old_cwd = os.getcwd()
                os.chdir('/root/workspace/data')
                try:
                    reeb_graph = load_reeb_graph_from_file(f'Graph_new_{case}.json')
                finally:
                    os.chdir(old_cwd)  # Restore original directory
                
                # Extract trajectory parameters from replanned data
                waypoints = replanned_trajectory['waypoints']
                phi = replanned_trajectory['phi']
                r0 = replanned_trajectory['r0']
                l = replanned_trajectory['l']
                phi_new = replanned_trajectory['phi_new']
                time_segments = replanned_trajectory['time_segments']
                Flagb = replanned_trajectory['Flagb']
                
                # Generate discrete trajectory
                discrete_result = generate_single_robot_discrete_trajectory(
                    robot_id=robot_id,
                    waypoints=waypoints,
                    phi=phi,
                    r0=r0,
                    l=l,
                    phi_new=phi_new,
                    time_segments=time_segments,
                    Flagb=Flagb,
                    reeb_graph=reeb_graph,
                    dt=0.1,
                    save_dir=f'/root/workspace/data/{case}/',
                    case=case
                )
                
                if discrete_result:
                    print(f"✓ Discrete trajectory generated and saved to: {discrete_result['discrete_trajectory_file']}")
                    print(f"  Total points: {discrete_result['num_points']}")
                    print(f"  Total time: {discrete_result['total_time']:.3f}s")
                else:
                    print(f"⚠ Failed to generate discrete trajectory")
            else:
                print(f"⚠ Graph file not found: {graph_file} - skipping discrete trajectory generation")
                
        except Exception as e:
            print(f"⚠ Error generating discrete trajectory: {e}")
            print(f"  Continuing with trajectory parameter saving only...")
    
    # Print summary
    print(f"\n=== Replanning Summary ===")
    print(f"Robot {robot_id} results:")
    print(f"  Original → Replanned: {optimization_results['original_total_time']:.3f}s → {optimization_results['optimized_total_time']:.3f}s")
    print(f"  Target: {optimization_results['target_time']:.3f}s")
    print(f"  Deviation: {optimization_results['deviation']:.3f}s")
    print(f"  Improvement: {optimization_results['improvement']:.3f}s")
    
    return result

def create_detailed_replanned_trajectory(original_data, delta_t_arcs, delta_t_lines, sol, total_time):
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

def create_replanned_trajectory_from_optimization(original_data, optimized_times, total_time):
    """
    Create new trajectory data with optimized times maintaining original structure
    
    Args:
        original_data: Original robot trajectory data
        optimized_times: List of optimized segment times
        total_time: Total optimized time
        
    Returns:
        Updated trajectory data with new timing
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

def save_single_robot_trajectory_parameters(result, case, robot_id):
    """
    Save replanned trajectory parameters for a single robot to file
    
    Args:
        result: Dictionary containing replanned results for one robot
        case: Case name
        robot_id: Robot ID
        
    Returns:
        Saved file path
    """
    output_dir = f'/root/workspace/data/{case}/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save replanned trajectory for this robot
    output_file = f'{output_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}.json'
    
    # Save the replanned trajectory data
    with open(output_file, 'w') as f:
        json.dump(result['replanned_trajectory'], f, indent=2)
    
    print(f"✓ Replanned trajectory saved: {output_file}")
    
    
    return output_file

def generate_discrete_trajectories_from_replanned_data(case, N, dt=0.1, save_dir=None):
    """
    Generate discrete trajectories for each robot from replanned trajectory data.
    Similar to compare_discretization_with_spline but processes individual robot data.
    
    Args:
        case: Case name (e.g., "simple_maze", "simulation", "maze")
        N: Number of robots
        dt: Time step for uniform sampling (default: 0.1s)
        save_dir: Directory to save trajectory files (default: /home/zhihui/data/{case}/)
    
    Returns:
        Dictionary containing results for each robot
    """
    if save_dir is None:
        save_dir = f'/home/zhihui/data/{case}/'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load reeb graph for this case
    reeb_graph_file = f"Graph_new_{case}.json"
    try:
        reeb_graph = load_reeb_graph_from_file(reeb_graph_file)
    except Exception as e:
        print(f"Error loading reeb graph: {e}")
        return None
    
    results = {}
    
    for robot_id in range(N):
        print(f"\nProcessing Robot {robot_id}...")
        
        # Load replanned trajectory data for this robot
        replanned_file = f'{save_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}.json'
        
        if not os.path.exists(replanned_file):
            print(f"Warning: Replanned data not found for Robot {robot_id} at {replanned_file}")
            continue
            
        try:
            with open(replanned_file, 'r') as f:
                replanned_data = json.load(f)
            
            # Extract trajectory parameters
            waypoints = replanned_data['waypoints']
            phi = replanned_data['phi']
            r0 = replanned_data['r0']
            l = replanned_data['l']
            phi_new = replanned_data['phi_new']
            time_segments = replanned_data['time_segments']
            Flagb = replanned_data['Flagb']
            total_time = replanned_data['total_time']
            
            print(f"  Loaded trajectory with {len(waypoints)} waypoints, total time: {total_time:.3f}s")
            
            # Generate discrete trajectory using the same approach as compare_discretization_with_spline
            robot_result = generate_single_robot_discrete_trajectory(
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
                save_dir=save_dir,
                case=case
            )
            
            if robot_result:
                results[f'robot_{robot_id}'] = robot_result
                print(f"  ✓ Robot {robot_id} discrete trajectory generated successfully")
            else:
                print(f"  ✗ Failed to generate discrete trajectory for Robot {robot_id}")
                
        except Exception as e:
            print(f"  ✗ Error processing Robot {robot_id}: {e}")
            traceback.print_exc()
    
    print(f"\n=== Discrete Trajectory Generation Summary ===")
    print(f"Successfully processed {len(results)} out of {N} robots")
    
    return results

def generate_single_robot_discrete_trajectory(robot_id, waypoints, phi, r0, l, phi_new, 
                                            time_segments, Flagb, reeb_graph, dt, save_dir, case):
    """
    Generate discrete trajectory for a single robot using optimization results with proper arc and line segments.
    
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
    from discretization import discretize_segment
    
    N = len(waypoints) - 1  # Number of segments
    
    # Process each segment independently with its own spline
    segment_boundaries = [0]
    current_time = 0.0
    segment_times = []
    segment_xs = []
    segment_ys = []
    total_time = 0.0
    
    # Collect boundary indices for each segment
    for i in range(N):
        # Discretize this segment
        x_seg, y_seg, t_seg = discretize_segment(
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
            total_time = current_time
    
    # Create uniform time grid
    t_uniform = np.arange(0, total_time, dt)
    x_uniform = np.zeros_like(t_uniform)
    y_uniform = np.zeros_like(t_uniform)
    
    # Interpolate each segment separately and combine
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
            
            # Calculate orientation using delta x and delta y instead of derivatives
            if i == 0:
                theta = phi_new[0]  # Use phi_new directly for first point
            else:
                # Calculate delta x and delta y between current and previous point
                delta_x = x_uniform[i] - x_uniform[i-1]
                delta_y = y_uniform[i] - y_uniform[i-1]
                
                # Calculate orientation from delta values
                if abs(delta_x) > 1e-10 or abs(delta_y) > 1e-10:
                    theta = np.arctan2(delta_y, delta_x)
                else:
                    # If no movement, keep previous theta
                    theta = thetas[-1] if thetas else phi_new[0]
            thetas.append(float(theta))
            
            # Calculate linear velocity using derivatives
            dx_dt = cs_x_seg(t, 1)
            dy_dt = cs_y_seg(t, 1)
            velocity = np.sqrt(dx_dt**2 + dy_dt**2)
            velocities.append(float(velocity))
            
            # Calculate angular velocity using delta theta instead of second derivatives
            if i == 0:
                angular_velocity = 0.0  # No angular velocity at first point
            else:
                # Calculate delta theta between current and previous point
                delta_theta = theta - thetas[-1]
                
                # Normalize delta theta to [-pi, pi] range
                while delta_theta > np.pi:
                    delta_theta -= 2 * np.pi
                while delta_theta < -np.pi:
                    delta_theta += 2 * np.pi
                
                # Angular velocity = delta_theta / delta_time
                angular_velocity = delta_theta / dt
                
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
    
    # Save discrete trajectory to JSON file (tb0_Trajectory.json format)
    discrete_trajectory_file = os.path.join(save_dir, f'tb{robot_id}_Trajectory_replanned.json')
    trajectory_data = {
        "Trajectory": trajectory_points
    }
    
    with open(discrete_trajectory_file, 'w') as f:
        json.dump(trajectory_data, f)
    

    
    return {
        "robot_id": robot_id,
        "discrete_trajectory_file": discrete_trajectory_file,
        "trajectory_points": trajectory_points,
        "total_time": total_time,
        "num_points": len(trajectory_points)
    }

def load_trajectory_from_json(file_path):
    """
    Load trajectory data from JSON file in tb0_Trajectory.json format
    
    Args:
        file_path: Path to the JSON trajectory file
        
    Returns:
        List of trajectory points in format [x, y, theta, v, w]
        Returns None if file doesn't exist or has wrong format
    """
    try:
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if 'Trajectory' not in data:
            print(f"Warning: No 'Trajectory' key found in {file_path}")
            return None
            
        trajectory = data['Trajectory']
        
        # Validate trajectory format
        if not trajectory or len(trajectory[0]) != 5:
            print(f"Warning: Invalid trajectory format in {file_path}")
            return None
            
        return trajectory
        
    except Exception as e:
        print(f"Error loading trajectory from {file_path}: {e}")
        return None