import numpy as np
import casadi as ca
import json
import matplotlib.pyplot as plt
import copy
import os
import traceback
from scipy.interpolate import CubicSpline
from GenerateMatrix import load_reeb_graph_from_file

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

def main_replanning(case, N, target_time=45.0):
    """
    Main function for trajectory parameter replanning
    
    Args:
        case: Case name (e.g., "simple_maze", "simulation", "maze")
        N: Number of robots
        target_time: Target coordination time (seconds)
    
    Returns:
        List with replanning results for each robot
    """
    print(f"=== Trajectory Parameter Replanning ===")
    print(f"Case: {case}")
    print(f"Number of robots: {N}")
    print(f"Target time: {target_time:.3f}s")
    
    results = []
    
    for i in range(N):
        print(f"\nRobot {i}: Optimizing trajectory parameters")
        # Use the trajectory parameter replanning function for single robot
        result = replan_trajectory_parameters_to_target(
            case=case,
            target_time=target_time,
            robot_id=i,
            save_results=True
        )
        
        if result:
            results.append(result)
            print(f"✓ Robot {i} optimization completed")
        else:
            print(f"✗ Robot {i} optimization failed")
    
    # Print overall summary
    print(f"\n=== Overall Summary ===")
    print(f"Successfully optimized {len(results)} out of {N} robots")
    
    if results:
        total_improvement = sum([r['optimization_results']['improvement'] for r in results])
        avg_deviation = sum([r['optimization_results']['deviation'] for r in results]) / len(results)
        print(f"Total time improvement: {total_improvement:.3f}s")
        print(f"Average deviation from target: {avg_deviation:.3f}s")
        
        # Generate discrete trajectories from replanned data
        print(f"\n=== Generating Discrete Trajectories ===")
        discrete_results = generate_discrete_trajectories_from_replanned_data(
            case=case, 
            N=N, 
            dt=0.1
        )
        
        if discrete_results:
            print(f"✓ Generated discrete trajectories for {len(discrete_results)} robots")
            # Add discrete trajectory results to the main results
            for robot_key, discrete_result in discrete_results.items():
                robot_id = discrete_result['robot_id']
                # Find corresponding result and add discrete trajectory info
                for result in results:
                    if result.get('robot_id') == robot_id:
                        result['discrete_trajectory'] = discrete_result
                        break
            
            # Generate velocity comparison plots
            print(f"\n=== Generating Velocity Comparisons ===")
            velocity_results = plot_velocity_comparison_original_vs_replanned(
                case=case,
                N=N,
                dt=0.1
            )
            
            if velocity_results:
                print(f"✓ Generated velocity comparisons for {len(velocity_results)} robots")
                # Add velocity comparison results to the main results
                for robot_key, velocity_result in velocity_results.items():
                    robot_id = velocity_result['robot_id']
                    # Find corresponding result and add velocity comparison info
                    for result in results:
                        if result.get('robot_id') == robot_id:
                            result['velocity_comparison'] = velocity_result
                            break
            else:
                print("✗ No velocity comparisons generated")
        else:
            print("✗ No discrete trajectories generated")
    
    return results

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
    
    return trajectory_data

def replan_trajectory_parameters_to_target(case, target_time, robot_id, save_results=True):
    """
    Load trajectory parameters from individual robot file and replan to achieve target time
    
    Args:
        case: Case name (e.g., "simple_maze")
        target_time: Target total time for the robot (seconds)
        robot_id: Robot ID to process (e.g., 0, 1, 2)
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
        
        # If no detailed variables were created, fall back to simple segment optimization
        if not all_vars_flat:
            # Simple segment-based optimization (original approach)
            T = opti.variable(num_segments)
            all_vars_flat = [T[i] for i in range(num_segments)]
            
            # Simple constraints
            for i in range(num_segments):
                opti.subject_to(T[i] >= 0.1)  # Minimum segment time
                opti.subject_to(T[i] <= target_time)  # Maximum segment time
            
            total_time = ca.sum1(T)
        else:
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
                        if i == 0 or (i < len(Flagb) and Flagb[i] != 0):  # First segment or relay point - start from rest
                            min_t_initial = np.sqrt(2*arc_segment_length / calculate_angular_acceleration_limit(r0[i]) / abs(r0[i])) if abs(r0[i]) > 0 else 0.2
                            opti.subject_to(delta_t_arcs[i][0] >= min_t_initial)
                            opti.subject_to(delta_t_arcs[i][0] <= 10.0)
                        
                        # Acceleration-based continuity between segments (simplified to match Planning_deltaT.py)
                        if i > 0 and (i >= len(Flagb) or Flagb[i] == 0):  # Not first segment and not relay point
                            if i-1 < len(delta_t_lines) and len(delta_t_lines[i-1]) > 0:  # Previous segment ends with line
                                prev_line_segment_length = l[i-1] / len(delta_t_lines[i-1])
                                v_end_prev_line = prev_line_segment_length / delta_t_lines[i-1][-1]
                                v_start_curr_arc = arc_segment_length / delta_t_arcs[i][0]
                                
                                t_avg_transition = (delta_t_lines[i-1][-1] + delta_t_arcs[i][0])/2
                                a_transition = (v_start_curr_arc - v_end_prev_line) / t_avg_transition
                                all_accelerations.append(a_transition)
                                
                                opti.subject_to(a_transition >= -a_max)
                                opti.subject_to(a_transition <= a_max)
                        
                        for j, dt_arc in enumerate(delta_t_arcs[i]):
                            # Time bounds (similar to Planning_deltaT.py)
                            opti.subject_to(dt_arc >= 0.20)  # Minimum time (same as Planning_deltaT.py)
                            opti.subject_to(dt_arc <= 10.0)   # Maximum time (same as Planning_deltaT.py)
                            
                            # Angular velocity constraint: omega = arc_segment_length / r / delta_t <= w_max_arc
                            if abs(r0[i]) > 0:
                                omega_c = arc_segment_length / abs(r0[i]) / dt_arc
                                w_max_arc = calculate_angular_velocity_limit(abs(r0[i]))
                                opti.subject_to(omega_c >= 0)
                                opti.subject_to(omega_c <= w_max_arc)
                            
                            # Angular acceleration constraint between consecutive arc subsegments
                            if j > 0:
                                # Similar to Planning_deltaT.py: tangential acceleration constraint
                                v1 = arc_segment_length / delta_t_arcs[i][j-1]
                                v2 = arc_segment_length / delta_t_arcs[i][j]
                                t_avg = (delta_t_arcs[i][j-1] + delta_t_arcs[i][j])/2
                                a_tangential = (v2 - v1) / t_avg
                                all_accelerations.append(a_tangential)
                                
                                # Convert to angular acceleration: alpha = a_tangential / radius
                                alpha = a_tangential / abs(r0[i])
                                aw_max_arc = calculate_angular_acceleration_limit(r0[i])
                                opti.subject_to(alpha >= -aw_max_arc)
                                opti.subject_to(alpha <= aw_max_arc)
                        
                        # Enhanced constraints for arc segments ending at relay points or final point
                        if (i == len(time_segments)-1 or (i+1 < len(Flagb) and Flagb[i+1] != 0)) and (i >= len(delta_t_lines) or len(delta_t_lines[i]) == 0):
                            # Ensure minimum time for final arc segment (similar to Planning_deltaT.py)
                            t_min_final = np.sqrt(2*arc_segment_length / calculate_angular_acceleration_limit(r0[i]) / abs(r0[i]))
                            opti.subject_to(delta_t_arcs[i][-1] >= t_min_final)
                            opti.subject_to(delta_t_arcs[i][-1] <= 10.0)
                    
                    # Line constraints
                    if i < len(delta_t_lines) and len(delta_t_lines[i]) > 0:
                        line_length = l[i] if i < len(l) else 0
                        line_segment_length = line_length / len(delta_t_lines[i])
                        
                        # Arc-to-line continuity within the same segment i
                        if i < len(delta_t_arcs) and len(delta_t_arcs[i]) > 0:
                            # Velocity transition from arc end to line start within same segment
                            arc_segment_length = abs(r0[i] * (phi[i+1] - phi_new[i])) / len(delta_t_arcs[i])
                            v_end_arc_i = arc_segment_length / delta_t_arcs[i][-1]  # Last arc subsegment
                            v_start_line_i = line_segment_length / delta_t_lines[i][0]  # First line subsegment
                            
                            t_avg = (delta_t_arcs[i][-1] + delta_t_lines[i][0])/2
                            a_transition = (v_start_line_i - v_end_arc_i) / t_avg
                            all_accelerations.append(a_transition)
                            
                            opti.subject_to(a_transition >= -a_max)
                            opti.subject_to(a_transition <= a_max)
                        
                        # Ensure starting from zero velocity for line segments at relay points
                        if len(delta_t_arcs[i]) == 0 and (i == 0 or (i < len(Flagb) and Flagb[i] != 0)):
                            # Line segment starting from zero velocity (similar to Planning_deltaT.py)
                            t_min = np.sqrt(2*line_segment_length/a_max)
                            opti.subject_to(delta_t_lines[i][0] >= t_min)
                            opti.subject_to(delta_t_lines[i][0] <= 5.0)
                        
                        for j, dt_line in enumerate(delta_t_lines[i]):
                            # Time bounds (similar to Planning_deltaT.py)
                            opti.subject_to(dt_line >= 0.1)  # Minimum time (same as Planning_deltaT.py)
                            opti.subject_to(dt_line <= 5.0)   # Maximum time (same as Planning_deltaT.py)
                            
                            # Linear velocity constraint: v = line_segment_length / delta_t <= v_max
                            velocity_expr = line_segment_length / dt_line
                            opti.subject_to(velocity_expr >= 0)
                            opti.subject_to(velocity_expr <= v_max)
                            
                            # Linear acceleration constraint between consecutive line subsegments
                            if j > 0:
                                # Similar to Planning_deltaT.py: acceleration constraint
                                a_lin = line_segment_length * (1/dt_line - 1/delta_t_lines[i][j-1]) / ((dt_line + delta_t_lines[i][j-1])/2)
                                all_accelerations.append(a_lin)
                                
                                # Constraint using the same form as Planning_deltaT.py
                                constraint_expr = (dt_line**2 - delta_t_lines[i][j-1]**2) / delta_t_lines[i][j-1] / dt_line
                                opti.subject_to(constraint_expr >= -a_max/2/line_segment_length)
                                opti.subject_to(constraint_expr <= a_max/2/line_segment_length)
                        
                        # Ensure stopping at the end of line segments at relay points or final point
                        if i == len(time_segments)-1 or (i+1 < len(Flagb) and Flagb[i+1] != 0):
                            # Minimum time for final line segment (similar to Planning_deltaT.py)
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
        
        # Initial guess: use loaded time segments and calculate initial values based on constraints
        if 'T' in locals():
            # Simple segment optimization - use current segment times as initial guess
            for i in range(num_segments):
                initial_time = max(0.1, current_segment_times[i])  # No scaling
                opti.set_initial(T[i], initial_time)
        else:
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
        
        if 'T' in locals():
            # Simple segment optimization
            T_opt = sol.value(T)
            if num_segments == 1:
                T_opt_list = [float(T_opt)]
            else:
                T_opt_list = T_opt.tolist()
        else:
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
        
        # Create replanned trajectory data
        if 'T' in locals():
            # Use simple segment approach
            replanned_trajectory = create_replanned_trajectory_from_optimization(
                robot_data, T_opt_list, optimized_total_time
            )
        else:
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
    output_dir =f'/root/workspace/data/{case}/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save replanned trajectory for this robot
    output_file = f'{output_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}.json'
    
    # Save the replanned trajectory data
    with open(output_file, 'w') as f:
        json.dump(result['replanned_trajectory'], f, indent=2)
    
    print(f"✓ Replanned trajectory saved: {output_file}")
    
    # # Also save optimization results summary
    # summary_file = f'{output_dir}robot_{robot_id}_optimization_summary_{case}.json'
    # summary_data = {
    #     'robot_id': robot_id,
    #     'case': case,
    #     'optimization_results': result['optimization_results'],
    #     'metadata': {
    #         'description': f'Optimization summary for Robot {robot_id}',
    #         'timestamp': str(np.datetime64('now'))
    #     }
    # }
    
    # with open(summary_file, 'w') as f:
    #     json.dump(summary_data, f, indent=2)
    
    # print(f"✓ Optimization summary saved: {summary_file}")
    
    return output_file

def generate_discrete_trajectories_from_replanned_data(case, N, dt=0.1, save_dir=None):
    """
    Generate discrete trajectories for each robot from replanned trajectory data.
    Similar to compare_discretization_with_spline but processes individual robot data.
    
    Args:
        case: Case name (e.g., "simple_maze", "simulation", "maze")
        N: Number of robots
        dt: Time step for uniform sampling (default: 0.1s)
        save_dir: Directory to save trajectory files (default: f'/root/workspace/data/{case}/')
    
    Returns:
        Dictionary containing results for each robot
    """
    if save_dir is None:
        save_dir = f'/root/workspace/data/{case}/'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load reeb graph for this case
    reeb_graph_file = f"/root/workspace/data/Graph_new_{case}.json"
    try:
        reeb_graph = load_reeb_graph_from_file(reeb_graph_file)
    except Exception as e:
        print(f"Error loading reeb graph: {e}")
        print(f"Looking for file: {reeb_graph_file}")
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
    Generate discrete trajectory for a single robot using cubic spline interpolation.
    Based on the compare_discretization_with_spline function from discretization.py.
    
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
    
    # Collect all discretized segments (original discrete points)
    all_times, all_xs, all_ys = [], [], []
    total_time = 0.0
    
    # Process each segment independently to get the original discrete points
    for i in range(N):
        # Discretize this segment - only getting endpoints
        x_seg, y_seg, t_seg = discretize_segment(
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
    
    # Create uniform time grid
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
            segment_boundaries.append(len(all_times))
    
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
    
    # Save discrete trajectory to JSON file (tb0_Trajectory.json format)
    discrete_trajectory_file = os.path.join(save_dir, f'tb{robot_id}_Trajectory_replanned.json')
    trajectory_data = {
        "Trajectory": trajectory_points
    }
    
    with open(discrete_trajectory_file, 'w') as f:
        json.dump(trajectory_data, f)
    
    # Create and save visualization plots
    plot_file = create_robot_trajectory_plots(
        robot_id=robot_id,
        all_times=all_times,
        all_xs=all_xs, 
        all_ys=all_ys,
        t_uniform=t_uniform,
        x_uniform=x_uniform,
        y_uniform=y_uniform,
        waypoints=waypoints,
        reeb_graph=reeb_graph,
        save_dir=save_dir,
        case=case
    )
    
    # Calculate statistics
    position_errors = []
    for i, (orig_x, orig_y) in enumerate(zip(all_xs, all_ys)):
        # Find closest point in interpolated trajectory
        distances = [(orig_x - x)**2 + (orig_y - y)**2 for x, y in zip(x_uniform, y_uniform)]
        min_idx = np.argmin(distances)
        error = np.sqrt((orig_x - x_uniform[min_idx])**2 + (orig_y - y_uniform[min_idx])**2)
        position_errors.append(error)
    
    position_errors = np.array(position_errors)
    
    stats = {
        "max_position_error": float(np.max(position_errors)),
        "mean_position_error": float(np.mean(position_errors)),
        "rms_position_error": float(np.sqrt(np.mean(position_errors**2)))
    }
    
    print(f"    Max position error: {stats['max_position_error']:.6f} m")
    print(f"    Mean position error: {stats['mean_position_error']:.6f} m")
    print(f"    RMS position error: {stats['rms_position_error']:.6f} m")
    
    return {
        "robot_id": robot_id,
        "discrete_trajectory_file": discrete_trajectory_file,
        "plot_file": plot_file,
        "trajectory_points": trajectory_points,
        "statistics": stats,
        "total_time": total_time,
        "num_points": len(trajectory_points)
    }

def create_robot_trajectory_plots(robot_id, all_times, all_xs, all_ys, t_uniform, 
                                x_uniform, y_uniform, waypoints, reeb_graph, save_dir, case):
    """
    Create and save trajectory comparison plots for a single robot
    
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(15, 12))
    
    # X position vs time plot
    plt.subplot(221)
    plt.plot(all_times, all_xs, 'ro-', label='Original Discrete Points', markersize=6)
    plt.plot(t_uniform, x_uniform, 'b-', label='Cubic Spline Interpolation', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title(f'Robot {robot_id} - X Position vs Time')
    plt.grid(True)
    plt.legend()
    
    # Y position vs time plot
    plt.subplot(222)
    plt.plot(all_times, all_ys, 'ro-', label='Original Discrete Points', markersize=6)
    plt.plot(t_uniform, y_uniform, 'b-', label='Cubic Spline Interpolation', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Robot {robot_id} - Y Position vs Time')
    plt.grid(True)
    plt.legend()
    
    # 2D trajectory plot
    plt.subplot(223)
    plt.plot(all_xs, all_ys, 'ro-', label='Original Discrete Points', markersize=6)
    plt.plot(x_uniform, y_uniform, 'b-', label='Cubic Spline Interpolation', linewidth=2)
    
    # Plot waypoints
    for i, wp_idx in enumerate(waypoints):
        pos = reeb_graph.nodes[wp_idx].configuration/100.0
        plt.plot(pos[0], pos[1], 'ko', markersize=8)
        plt.text(pos[0], pos[1], f'W{wp_idx}', fontsize=10)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Robot {robot_id} - 2D Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Velocity profile
    plt.subplot(224)
    # Calculate velocity from discrete points
    discrete_velocities = []
    for i in range(len(all_times)-1):
        dt = all_times[i+1] - all_times[i]
        dx = all_xs[i+1] - all_xs[i]
        dy = all_ys[i+1] - all_ys[i]
        v = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
        discrete_velocities.append(v)
    
    if len(discrete_velocities) > 0:
        plt.plot(all_times[:-1], discrete_velocities, 'ro-', label='Discrete Velocity', markersize=4)
    
    # Calculate velocity from spline
    spline_velocities = []
    for i in range(len(t_uniform)-1):
        dt = t_uniform[i+1] - t_uniform[i]
        dx = x_uniform[i+1] - x_uniform[i]
        dy = y_uniform[i+1] - y_uniform[i]
        v = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
        spline_velocities.append(v)
    
    if len(spline_velocities) > 0:
        plt.plot(t_uniform[:-1], spline_velocities, 'b-', label='Spline Velocity', linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Robot {robot_id} - Velocity Profile')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(save_dir, f'robot_{robot_id}_trajectory_comparison_{case}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"    ✓ Plot saved: {plot_file}")
    
    return plot_file

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

def extract_velocity_profile_from_json(trajectory_data, dt=0.1):
    """
    Extract velocity profile from trajectory data loaded from JSON
    
    Args:
        trajectory_data: List of trajectory points [x, y, theta, v, w]
        dt: Time step for uniform sampling
        
    Returns:
        Dictionary containing velocity analysis data
    """
    if not trajectory_data:
        return None
        
    # Extract positions, velocities and angular velocities directly from JSON data
    positions = np.array([[point[0], point[1]] for point in trajectory_data])
    velocities = np.array([point[3] for point in trajectory_data])  # v (already computed)
    angular_velocities = np.array([point[4] for point in trajectory_data])  # w (already computed)
    
    # Calculate distances between consecutive points
    distances = np.zeros(len(positions))
    if len(positions) > 1:
        distances[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    cumulative_distance = np.cumsum(distances)
    
    # Create time points assuming uniform time intervals
    total_points = len(trajectory_data)
    time_points = np.linspace(0, (total_points - 1) * dt, total_points)
    
    # Use the velocities directly from JSON - no need to calculate
    # Both discrete and spline velocities are the same since they come from the JSON
    discrete_times = time_points[:-1] if len(time_points) > 1 else []
    discrete_velocities = velocities[:-1].tolist() if len(velocities) > 1 else []
    spline_times = time_points
    spline_velocities = velocities
    
    # Calculate statistics
    total_time = time_points[-1] if len(time_points) > 0 else 0
    max_velocity = np.max(velocities) if len(velocities) > 0 else 0
    avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0
    max_angular_velocity = np.max(np.abs(angular_velocities)) if len(angular_velocities) > 0 else 0
    avg_angular_velocity = np.mean(np.abs(angular_velocities)) if len(angular_velocities) > 0 else 0
    
    return {
        'time': time_points,
        'positions': positions,
        'velocities': velocities,
        'angular_velocities': angular_velocities,
        'distances': distances,
        'cumulative_distance': cumulative_distance,
        'total_time': total_time,
        'total_distance': cumulative_distance[-1] if len(cumulative_distance) > 0 else 0,
        'max_velocity': max_velocity,
        'avg_velocity': avg_velocity,
        'max_angular_velocity': max_angular_velocity,
        'avg_angular_velocity': avg_angular_velocity,
        # Add the required keys for the plot function (using velocities from JSON)
        'discrete_times': discrete_times,
        'discrete_velocities': discrete_velocities,
        'spline_times': spline_times,
        'spline_velocities': spline_velocities
    }

def plot_velocity_comparison_original_vs_replanned(case, N, dt=0.1):
    """
    Generate velocity comparison plots between original and replanned trajectories by reading from JSON files
    
    Args:
        case: Case name (e.g., "simple_maze", "simulation", "maze")
        N: Number of robots
        dt: Time step for uniform sampling (default: 0.1s)
    
    Returns:
        Dictionary containing velocity comparison results for each robot
    """
    base_path =f'/root/workspace/data/{case}/'
    save_dir = os.path.join(base_path)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== Generating Velocity Comparisons from JSON Files ===")
    print(f"Case: {case}")
    print(f"Loading trajectories from: {base_path}")
    print(f"Saving plots to: {save_dir}")
    
    velocity_results = {}
    
    for robot_id in range(N):
        print(f"\nProcessing Robot {robot_id}...")
        
        # Define file paths
        original_file = os.path.join(base_path, f"tb{robot_id}_Trajectory.json")
        replanned_file = os.path.join(base_path, f"tb{robot_id}_Trajectory_replanned.json")
        
        print(f"  Original: {original_file}")
        print(f"  Replanned: {replanned_file}")
        
        # Load trajectory data
        original_trajectory = load_trajectory_from_json(original_file)
        replanned_trajectory = load_trajectory_from_json(replanned_file)
        
        if original_trajectory is None:
            print(f"  ✗ Could not load original trajectory for robot {robot_id}")
            continue
            
        if replanned_trajectory is None:
            print(f"  ✗ Could not load replanned trajectory for robot {robot_id}")
            continue
            
        print(f"  ✓ Loaded trajectories: Original({len(original_trajectory)} points), Replanned({len(replanned_trajectory)} points)")
        
        # Extract velocity profiles
        original_velocity_data = extract_velocity_profile_from_json(original_trajectory, dt)
        replanned_velocity_data = extract_velocity_profile_from_json(replanned_trajectory, dt);
        
        if original_velocity_data is None or replanned_velocity_data is None:
            print(f"  ✗ Could not extract velocity profiles for robot {robot_id}")
            continue
        
        # Create velocity comparison plot
        plot_file = create_velocity_comparison_plot(
            original_velocity_data, 
            replanned_velocity_data, 
            robot_id, 
            case, 
            save_dir
        )
        
        # Store results
        velocity_results[f'robot_{robot_id}'] = {
            'robot_id': robot_id,
            'original_velocity_data': original_velocity_data,
            'replanned_velocity_data': replanned_velocity_data,
            'plot_file': plot_file
        }
        
        print(f"  ✓ Velocity comparison completed for robot {robot_id}")
    
    return velocity_results

def create_velocity_comparison_plot(original_data, replanned_data, robot_id, case, save_dir):
    """
    Create velocity comparison plot between original and replanned trajectories
    
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Discrete velocity comparison
    plt.subplot(221)
    plt.plot(original_data['discrete_times'], original_data['discrete_velocities'], 
             'ro-', label='Original Discrete Velocity', markersize=4, alpha=0.7)
    plt.plot(replanned_data['discrete_times'], replanned_data['discrete_velocities'], 
             'bo-', label='Replanned Discrete Velocity', markersize=4, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Robot {robot_id} - Discrete Velocity Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Spline velocity comparison
    plt.subplot(222)
    plt.plot(original_data['spline_times'], original_data['spline_velocities'], 
             'r-', label='Original Spline Velocity', linewidth=2, alpha=0.8)
    plt.plot(replanned_data['spline_times'], replanned_data['spline_velocities'], 
             'b-', label='Replanned Spline Velocity', linewidth=2, alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Robot {robot_id} - Spline Velocity Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Combined velocity comparison (spline only)
    plt.subplot(223)
    plt.plot(original_data['spline_times'], original_data['spline_velocities'], 
             'r-', label=f'Original (Total: {original_data["total_time"]:.2f}s)', linewidth=2)
    plt.plot(replanned_data['spline_times'], replanned_data['spline_velocities'], 
             'b-', label=f'Replanned (Total: {replanned_data["total_time"]:.2f}s)', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Robot {robot_id} - Velocity Profile Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Velocity statistics comparison
    plt.subplot(224)
    categories = ['Max Velocity', 'Avg Velocity']
    original_stats = [original_data['max_velocity'], original_data['avg_velocity']]
    replanned_stats = [replanned_data['max_velocity'], replanned_data['avg_velocity']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, original_stats, width, label='Original', color='red', alpha=0.7)
    plt.bar(x + width/2, replanned_stats, width, label='Replanned', color='blue', alpha=0.7)
    
    plt.xlabel('Velocity Metrics')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Robot {robot_id} - Velocity Statistics Comparison')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(original_stats):
        plt.text(i - width/2, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    for i, v in enumerate(replanned_stats):
        plt.text(i + width/2, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(save_dir, f'robot_{robot_id}_velocity_comparison_{case}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"    ✓ Velocity comparison plot saved: {plot_file}")
    
    return plot_file

# Call the main function
if __name__ == "__main__":
    # Main function call with case, N, and robot_id parameters
    case = "simple_maze"
    # case="simulation"
    N = 3
    target_time = 45.0
    main_replanning(case=case, N=N, target_time=target_time)
