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

def replan_trajectory_parameters_to_target_optimized(case, target_time, robot_id, save_results=True):
    """
    OPTIMIZED VERSION: Load trajectory parameters from individual robot file and replan to achieve target time
    with significant performance improvements in constraint building and solver configuration.
    
    Args:
        case: Case name (e.g., "simple_maze")
        target_time: Target total time for the robot (seconds)
        robot_id: Robot ID
        save_results: Whether to save replanned results
        
    Returns:
        Dictionary with replanning results for the specified robot
    """
    print(f"=== Replanning Trajectory Parameters for {case} (OPTIMIZED) ===")
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
    
    # Replan using CasADi optimization with performance optimizations
    print(f"\n3. Optimizing trajectory (FAST MODE)...")
    
    try:
        start_time = time.time()
        opti = ca.Opti()
        
        # Extract robot trajectory parameters for constraint calculations
        waypoints = robot_data.get('waypoints', [])
        phi = np.array(robot_data.get('phi', []))
        r0 = np.array(robot_data.get('r0', []))
        l = np.array(robot_data.get('l', []))
        phi_new = np.array(robot_data.get('phi_new', []))
        Flagb = robot_data.get('Flagb', [])
        
        # OPTIMIZATION 1: Pre-compute all segment parameters in a single pass
        print(f"   Precomputing segment parameters...")
        Deltal = 0.02  # Small segment length (from Planning_deltaT.py)
        segment_params = []
        total_arc_vars = 0
        total_line_vars = 0
        
        # Single pass to compute all parameters and count variables
        for i, segment in enumerate(time_segments):
            if i >= len(r0) or i >= len(l) or i >= len(phi):
                segment_params.append(None)
                continue
                
            # Pre-compute geometric parameters
            delta_phi = phi[i+1] - phi_new[i] if i+1 < len(phi) else 0
            arc_length = abs(r0[i] * delta_phi) if abs(r0[i]) > 0 else 0
            line_length = l[i]
            
            # Pre-compute subsegment counts
            N_arc = max(1, int(arc_length / Deltal)) if arc_length > 0.03 else 0
            N_line = max(1, int(line_length / Deltal)) if line_length > 0.03 else 0
            
            # Pre-compute physical limits
            w_max_arc = calculate_angular_velocity_limit(abs(r0[i])) if abs(r0[i]) > 0 else 0
            aw_max_arc = calculate_angular_acceleration_limit(r0[i]) if abs(r0[i]) > 0 else 0
            
            # Check if we need variables for this segment
            orig_arc_times = segment.get('arc', [])
            orig_line_times = segment.get('line', [])
            has_arc_vars = N_arc > 0 and len(orig_arc_times) > 0
            has_line_vars = N_line > 0 and len(orig_line_times) > 0
            
            if has_arc_vars:
                total_arc_vars += N_arc
            if has_line_vars:
                total_line_vars += N_line
            
            segment_params.append({
                'delta_phi': delta_phi,
                'arc_length': arc_length,
                'line_length': line_length,
                'N_arc': N_arc,
                'N_line': N_line,
                'w_max_arc': w_max_arc,
                'aw_max_arc': aw_max_arc,
                'has_arc_vars': has_arc_vars,
                'has_line_vars': has_line_vars,
                'arc_segment_length': arc_length / N_arc if N_arc > 0 else 0,
                'line_segment_length': line_length / N_line if N_line > 0 else 0,
                'orig_arc_times': orig_arc_times,
                'orig_line_times': orig_line_times
            })
        
        print(f"   Total variables: {total_arc_vars} arc + {total_line_vars} line = {total_arc_vars + total_line_vars}")
        
        # OPTIMIZATION 2: Choose between simple and detailed optimization based on problem size
        if total_arc_vars + total_line_vars == 0 or total_arc_vars + total_line_vars > 1000:
            # Simple segment-based optimization for large problems or no detailed segments
            print(f"   Using SIMPLE optimization (faster)")
            T = opti.variable(num_segments)
            
            # Vectorized constraints for speed
            opti.subject_to(T >= 0.1)
            opti.subject_to(T <= target_time * 2.0)  # Generous upper bound
            
            total_time = ca.sum1(T)
            all_accelerations = []
            
            # Simple initial guess
            for i in range(num_segments):
                initial_time = max(0.1, current_segment_times[i])
                opti.set_initial(T[i], initial_time)
                
        else:
            # OPTIMIZATION 3: Detailed optimization with vectorized variable creation
            print(f"   Using DETAILED optimization with vectorized constraints")
            
            # Create all variables at once for better memory management
            all_arc_vars = opti.variable(total_arc_vars) if total_arc_vars > 0 else ca.DM([])
            all_line_vars = opti.variable(total_line_vars) if total_line_vars > 0 else ca.DM([])
            
            # OPTIMIZATION 4: Pre-allocate constraint vectors for vectorized operations
            lower_bounds = []
            upper_bounds = []
            constraint_vars = []
            all_accelerations = []
            
            # Build constraint vectors efficiently
            arc_idx = 0
            line_idx = 0
            
            for i, params in enumerate(segment_params):
                if params is None:
                    continue
                    
                # Arc variables constraints
                if params['has_arc_vars']:
                    N_arc = params['N_arc']
                    arc_segment_length = params['arc_segment_length']
                    w_max_arc = params['w_max_arc']
                    aw_max_arc = params['aw_max_arc']
                    
                    # Add time bounds for all arc variables in this segment
                    for j in range(N_arc):
                        var = all_arc_vars[arc_idx + j]
                        constraint_vars.append(var)
                        
                        # Determine minimum time based on physics
                        if (i == 0 or (i < len(Flagb) and Flagb[i] != 0)) and j == 0:
                            min_t = np.sqrt(2*arc_segment_length / aw_max_arc / abs(r0[i])) if abs(r0[i]) > 0 else 0.2
                        else:
                            min_t = 0.05
                            
                        lower_bounds.append(min_t)
                        upper_bounds.append(10.0)
                        
                        # Angular velocity constraint
                        if abs(r0[i]) > 0:
                            omega_constraint = arc_segment_length / abs(r0[i]) / var
                            constraint_vars.append(omega_constraint)
                            lower_bounds.append(0.0)
                            upper_bounds.append(w_max_arc)
                    
                    arc_idx += N_arc
                
                # Line variables constraints
                if params['has_line_vars']:
                    N_line = params['N_line']
                    line_segment_length = params['line_segment_length']
                    
                    # Add time bounds for all line variables in this segment
                    for j in range(N_line):
                        var = all_line_vars[line_idx + j]
                        constraint_vars.append(var)
                        
                        # Determine minimum time based on physics
                        if (not params['has_arc_vars'] and (i == 0 or (i < len(Flagb) and Flagb[i] != 0))) and j == 0:
                            min_t = np.sqrt(2*line_segment_length/a_max)
                        else:
                            min_t = 0.05
                            
                        lower_bounds.append(min_t)
                        upper_bounds.append(5.0)
                        
                        # Linear velocity constraint
                        velocity_constraint = line_segment_length / var
                        constraint_vars.append(velocity_constraint)
                        lower_bounds.append(0.0)
                        upper_bounds.append(v_max)
                    
                    line_idx += N_line
            
            # OPTIMIZATION 5: Apply all constraints in single vectorized operations
            if constraint_vars:
                constraint_vector = ca.vertcat(*constraint_vars)
                lower_vector = ca.DM(lower_bounds)
                upper_vector = ca.DM(upper_bounds)
                
                opti.subject_to(constraint_vector >= lower_vector)
                opti.subject_to(constraint_vector <= upper_vector)
            
            # Calculate total time efficiently
            total_time = 0
            if total_arc_vars > 0:
                total_time += ca.sum1(all_arc_vars)
            if total_line_vars > 0:
                total_time += ca.sum1(all_line_vars)
            
            # OPTIMIZATION 6: Efficient batch initial value setting
            if total_arc_vars > 0:
                arc_initials = []
                arc_idx = 0
                for i, params in enumerate(segment_params):
                    if params and params['has_arc_vars']:
                        orig_arc_times = params['orig_arc_times']
                        N_arc = params['N_arc']
                        
                        if orig_arc_times and len(orig_arc_times) == N_arc:
                            arc_initials.extend([max(0.05, t) for t in orig_arc_times])
                        elif orig_arc_times:
                            avg_time = sum(orig_arc_times) / N_arc
                            arc_initials.extend([max(0.05, avg_time)] * N_arc)
                        else:
                            conservative_time = (params['arc_segment_length'] / abs(r0[i]) / params['w_max_arc'] * 2.0 
                                               if abs(r0[i]) > 0 and params['w_max_arc'] > 0 else 0.5)
                            arc_initials.extend([max(0.05, conservative_time)] * N_arc)
                        arc_idx += N_arc
                
                opti.set_initial(all_arc_vars, arc_initials)
            
            if total_line_vars > 0:
                line_initials = []
                line_idx = 0
                for i, params in enumerate(segment_params):
                    if params and params['has_line_vars']:
                        orig_line_times = params['orig_line_times']
                        N_line = params['N_line']
                        
                        if orig_line_times and len(orig_line_times) == N_line:
                            line_initials.extend([max(0.05, t) for t in orig_line_times])
                        elif orig_line_times:
                            avg_time = sum(orig_line_times) / N_line
                            line_initials.extend([max(0.05, avg_time)] * N_line)
                        else:
                            conservative_time = params['line_segment_length'] / (v_max * 0.5) if params['line_segment_length'] > 0 else 0.5
                            line_initials.extend([max(0.05, conservative_time)] * N_line)
                        line_idx += N_line
                
                opti.set_initial(all_line_vars, line_initials)
        
        # OPTIMIZATION 7: Simplified objective function
        objective = (total_time - target_time)**2
        
        # Reduced acceleration penalty for speed (optional - can be disabled for max speed)
        if all_accelerations:
            acceleration_penalty_weight = 100.0  # Reduced from 1000.0 for faster solving
            accel_terms_vector = ca.vertcat(*all_accelerations)
            objective += acceleration_penalty_weight * ca.sumsqr(accel_terms_vector)
        
        opti.minimize(objective)
        
        # Total time constraints with flexibility
        opti.subject_to(total_time >= target_time * 0.7)  # More flexible
        opti.subject_to(total_time <= target_time * 1.3)
        
        # OPTIMIZATION 8: Highly optimized solver settings for maximum speed
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 1000,              # Reduced for speed
            'ipopt.acceptable_tol': 1e-2,        # Relaxed tolerance for speed
            'ipopt.tol': 1e-4,                   # Relaxed main tolerance
            'ipopt.dual_inf_tol': 1e-4,          # Relaxed dual feasibility tolerance
            'ipopt.constr_viol_tol': 1e-4,       # Relaxed constraint violation tolerance
            'ipopt.warm_start_init_point': 'yes', # Use warm start
            'ipopt.mu_strategy': 'adaptive',      # Adaptive barrier parameter
            'ipopt.linear_solver': 'mumps',       # Fast linear solver
            'ipopt.hessian_approximation': 'limited-memory',  # BFGS for speed
            'ipopt.limited_memory_max_history': 6,            # Reduced memory for speed
            'ipopt.obj_scaling_factor': 1.0,      # No objective scaling
            'ipopt.nlp_scaling_method': 'none'    # No NLP scaling for speed
        }
        opti.solver('ipopt', opts)
        
        setup_time = time.time() - start_time
        print(f"   Setup completed in {setup_time:.3f}s")
        
        # Solve
        solve_start_time = time.time()
        sol = opti.solve()
        solve_time = time.time() - solve_start_time
        
        print(f"   Solver completed in {solve_time:.3f}s")
        
        # Extract solution efficiently
        optimized_total_time = float(sol.value(total_time))
        
        if total_arc_vars + total_line_vars == 0 or total_arc_vars + total_line_vars > 1000:
            # Simple segment optimization
            T_opt = sol.value(T)
            T_opt_list = [float(T_opt)] if num_segments == 1 else T_opt.tolist()
            replanned_trajectory = create_replanned_trajectory_from_optimization(
                robot_data, T_opt_list, optimized_total_time
            )
        else:
            # Detailed subsegment optimization - reconstruct segment times efficiently
            T_opt_list = []
            arc_sol = sol.value(all_arc_vars) if total_arc_vars > 0 else []
            line_sol = sol.value(all_line_vars) if total_line_vars > 0 else []
            
            arc_idx = 0
            line_idx = 0
            for i, params in enumerate(segment_params):
                if params is None:
                    T_opt_list.append(0.1)
                    continue
                    
                segment_time = 0.0
                if params['has_arc_vars']:
                    N_arc = params['N_arc']
                    segment_time += sum(float(arc_sol[arc_idx + j]) for j in range(N_arc))
                    arc_idx += N_arc
                if params['has_line_vars']:
                    N_line = params['N_line']
                    segment_time += sum(float(line_sol[line_idx + j]) for j in range(N_line))
                    line_idx += N_line
                T_opt_list.append(max(0.1, segment_time))
            
            # Create detailed trajectory
            replanned_trajectory = create_detailed_replanned_trajectory_optimized(
                robot_data, arc_sol, line_sol, segment_params, optimized_total_time
            )
        
        optimization_results = {
            'original_total_time': current_robot_time,
            'target_time': target_time,
            'optimized_total_time': optimized_total_time,
            'segment_times': T_opt_list,
            'deviation': abs(optimized_total_time - target_time),
            'improvement': current_robot_time - optimized_total_time,
            'setup_time': setup_time,
            'solve_time': solve_time,
            'total_optimization_time': setup_time + solve_time
        }
        
        print(f"  ✓ Optimization successful!")
        print(f"    Optimized time: {optimized_total_time:.3f}s")
        print(f"    Deviation: {abs(optimized_total_time - target_time):.3f}s")
        print(f"    Improvement: {current_robot_time - optimized_total_time:.3f}s")
        print(f"    Total optimization time: {setup_time + solve_time:.3f}s")
        
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
    print(f"\n=== Replanning Summary (OPTIMIZED) ===")
    print(f"Robot {robot_id} results:")
    print(f"  Original → Replanned: {optimization_results['original_total_time']:.3f}s → {optimization_results['optimized_total_time']:.3f}s")
    print(f"  Target: {optimization_results['target_time']:.3f}s")
    print(f"  Deviation: {optimization_results['deviation']:.3f}s")
    print(f"  Improvement: {optimization_results['improvement']:.3f}s")
    print(f"  Optimization Time: {optimization_results['total_optimization_time']:.3f}s")
    
    return result

def create_detailed_replanned_trajectory_optimized(original_data, arc_sol, line_sol, segment_params, total_time):
    """
    OPTIMIZED VERSION: Create new trajectory data with optimized detailed subsegment times
    """
    # Deep copy original data
    replanned_data = copy.deepcopy(original_data)
    
    # Update time segments with detailed subsegment times efficiently
    time_segments = replanned_data.get('time_segments', [])
    
    arc_idx = 0
    line_idx = 0
    
    for i, segment in enumerate(time_segments):
        if i >= len(segment_params) or segment_params[i] is None:
            continue
            
        params = segment_params[i]
        
        # Update arc times
        if params['has_arc_vars']:
            N_arc = params['N_arc']
            new_arc_times = []
            for j in range(N_arc):
                new_arc_times.append(float(arc_sol[arc_idx + j]))
            segment['arc'] = new_arc_times
            arc_idx += N_arc
        
        # Update line times
        if params['has_line_vars']:
            N_line = params['N_line']
            new_line_times = []
            for j in range(N_line):
                new_line_times.append(float(line_sol[line_idx + j]))
            segment['line'] = new_line_times
            line_idx += N_line
    
    # Update total time
    replanned_data['total_time'] = total_time
    
    # Update metadata
    if 'metadata' not in replanned_data:
        replanned_data['metadata'] = {}
    
    replanned_data['metadata']['replanned'] = True
    replanned_data['metadata']['replan_timestamp'] = str(np.datetime64('now'))
    replanned_data['metadata']['original_total_time'] = original_data.get('total_time', 0)
    replanned_data['metadata']['replanned_total_time'] = total_time
    replanned_data['metadata']['detailed_optimization'] = True
    replanned_data['metadata']['optimization_method'] = 'vectorized_fast'
    
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
    if 'metadata' not in replanned_data:
        replanned_data['metadata'] = {}
    
    replanned_data['metadata']['replanned'] = True
    replanned_data['metadata']['replan_timestamp'] = str(np.datetime64('now'))
    replanned_data['metadata']['original_total_time'] = original_data.get('total_time', 0)
    replanned_data['metadata']['replanned_total_time'] = total_time
    replanned_data['metadata']['optimization_method'] = 'simple_fast'
    
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
    output_file = f'{output_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}_optimized.json'
    
    # Save the replanned trajectory data
    with open(output_file, 'w') as f:
        json.dump(result['replanned_trajectory'], f, indent=2)
    
    print(f"✓ Replanned trajectory saved: {output_file}")
    
    return output_file

# Test function to compare performance
def test_optimization_performance(case='simple_maze', robot_id=0, target_time=10.0):
    """
    Test function to compare optimization performance
    """
    print("=== OPTIMIZATION PERFORMANCE TEST ===")
    
    # Test optimized version
    print("\n1. Testing OPTIMIZED version...")
    start_time = time.time()
    result_optimized = replan_trajectory_parameters_to_target_optimized(case, target_time, robot_id, save_results=False)
    optimized_time = time.time() - start_time
    
    if result_optimized:
        opt_results = result_optimized['optimization_results']
        print(f"\nOPTIMIZED Results:")
        print(f"  Total time: {optimized_time:.3f}s")
        print(f"  Setup time: {opt_results.get('setup_time', 0):.3f}s")
        print(f"  Solve time: {opt_results.get('solve_time', 0):.3f}s")
        print(f"  Optimized trajectory time: {opt_results['optimized_total_time']:.3f}s")
        print(f"  Deviation from target: {opt_results['deviation']:.3f}s")
    
    print(f"\n=== PERFORMANCE SUMMARY ===")
    print(f"Optimized version: {optimized_time:.3f}s total")
    
    return result_optimized

if __name__ == "__main__":
    # Example usage
    test_optimization_performance()
