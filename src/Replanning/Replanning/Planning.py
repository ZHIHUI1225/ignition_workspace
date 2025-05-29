# optimization the time of deleta l of the trajectory
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import json
import casadi as ca
import casadi.tools as ca_tools

# Import the discretization tools
from discretization import compare_discretization_with_spline

# Import the trajectory saving function
from save_spline_trajectory import save_spline_trajectory

aw_max=0.2*np.pi# the maximum angular acceleration
w_max=0.5*np.pi # the maximum angular velocity
r_limit=0.75 # m
r_w=0.033 # the radius of the wheel
v_max=w_max*r_w# m/s
a_max=aw_max*r_w
l_r=0.14 # the wheel base



def load_WayPointFlag_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Waypoints'], data['Flags'],data["FlagB"]

def load_reeb_graph(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    nodes = data['nodes']
    in_neighbors = data['in_neighbors']
    out_neighbors = data['out_neighbors']
    return nodes, in_neighbors, out_neighbors

def load_optimization_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    phi = data["Optimization_phi"]
    l = data["Optimization_l"]
    r = data["Optimization_r"]
    v = data["Optimization_v"]
    a = data["Optimization_a"]
    return phi,l,r,v,a

def load_robot_path_data(file_path):
    """
    Load robot path data from a JSON file containing segment information.
    
    Args:
        file_path: Path to the robot path JSON file
        
    Returns:
        Dictionary containing the robot path data
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Functions to calculate constraints based on differential drive kinematics
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

def load_WayPointFlag_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Waypoints'], data['Flags'],data["FlagB"]

def load_matrices_from_file(file_path):
    data = np.load(file_path)
    Ec = data['Ec']
    El = data['El']
    Ad = data['Ad']
    return Ec, El, Ad
def load_phi_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Optimization_phi']
def load_r_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Optimization_r']
def load_l_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Optimization_l']

def load_trajectory_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Optimization_phi'],data['Optimization_l'],data['Optimization_r'],data['Optimization_v'],data['Optimization_a']

def Get_index(r0, l, delta_phi, Deltal):
    """
    Calculate the number of small segments needed for arc and line parts.
    
    Args:
        r0: Radius of the arc
        l: Length of the straight line
        delta_phi: Angular change of the arc
        Deltal: Small segment length
    
    Returns:
        Tuple (N_arc, N_line) with number of segments for arc and line
    """
    # Calculate number of arc segments, ensuring at least 1 if arc exists
    arc_length = abs(r0 * delta_phi)
    N_arc = max(1, int(arc_length / Deltal)) if arc_length > 0.03 else 0
    
    # Calculate number of line segments, ensuring at least 1 if line exists
    N_line = max(1, int(l / Deltal)) if l > 0.03 else 0
    
    return N_arc, N_line
    

def Planning_deltaT(waypoints_file_path, Initial_Guess_file_path, Target_time):
    """
    Optimize the timing of each segment to match the target time
    
    Args:
        waypoints_file_path: Path to the Waypoints.json file
        Initial_Guess_file_path: Path to the robot_path.json file
        Target_time: Target completion time for the trajectory
        
    Returns:
        time_segments: Optimized time segments
        total_time: Total trajectory time after optimization
    """
    Deltal = 0.02 #m 
    
    # Load the waypoints data
    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)

    phi_data=np.array(load_phi_from_file(Initial_Guess_file_path))
    r_guess=np.array(load_r_from_file(Initial_Guess_file_path))/100 #cm to m 
    l_guess=np.array(load_l_from_file(Initial_Guess_file_path))/100
    
    # Number of variables
    N = len(Waypoints)
    N_relays = np.nonzero(Flagb)[0].shape[0]
    
    # Create symbolic variables
    phi=phi_data # the angel at each waypoints
    r0=r_guess # the radius of each arc
    l=l_guess # the length of each straight line
    phi_new=np.zeros(np.shape(phi)) # the angle of each straight line
    
    # Arrays to store the number of small segments
    ArcIndex=np.zeros(N-1, dtype=int)
    LineIndex=np.zeros(N-1, dtype=int)
    
    # Calculate the number of arc and line segments
    for i in range(N-1):
        phi_new[i]=phi[i]+Flagb[i]*np.pi/2
        delta_phi=(phi[i+1] - phi_new[i])
        Index=Get_index(r0[i],l[i],delta_phi,Deltal)
        ArcIndex[i]=Index[0]
        LineIndex[i]=Index[1]
    
    # Create symbolic variables for delta times only
    delta_t_arcs = []  # List to hold delta time variables for arcs
    delta_t_lines = [] # List to hold delta time variables for lines
    
    # Constraints
    g = []    # Constraints
    lbg = []  # Lower bounds for constraints
    ubg = []  # Upper bounds for constraints
    all_accelerations = [] # NEW: List to store all acceleration terms for the objective
    arc_counter = 0
    line_counter = 0
    # For each segment, create symbolic variables for delta times
    for i in range(N-1):
        # Create variables for arc segments
        if ArcIndex[i] > 0:
            arc_counter += 1
            # Delta times for arc segments - use a unique name for each segment
            # delta_t_arc_i = ca.SX.sym(f'delta_t_arc_seg{i}', ArcIndex[i])
            # delta_t_arcs.append(delta_t_arc_i)  # Still add to the list, but with a segment-specific name
            delta_t_arcs.append([ca.SX.sym(f'delta_t_arc{arc_counter}_{j}') for j in range(ArcIndex[i])])
            # Arc segment - compute segment length
            delta_phi = phi[i+1] - phi_new[i]
            arc_length = abs(r0[i] * delta_phi)
            arc_segment_length = arc_length / ArcIndex[i]
            
            # Acceleration-based continuity between segments
            if i > 0 and Flagb[i] == 0:  # Not the first segment and not a relay point
                # Connect with previous segment's final velocity using acceleration constraints
                if LineIndex[i-1] > 0:  # Previous segment ends with a line
                    # Convert from line to arc velocity continuity
                    # v_end_prev = line_segment_length / delta_t_prev
                    prev_line_subsegment_length = l[i-1] / LineIndex[i-1]
                    # Velocity at the end of the last subsegment of the previous line (i-1)
                    v_end_prev_line = prev_line_subsegment_length / delta_t_lines[i-1][-1] # Corrected: use last element of the last line segment times
                    
                    # Tangential velocity at the start of the first subsegment of the current arc (i)
                    current_arc_subsegment_length = abs(r0[i] * (phi[i+1] - phi_new[i])) / ArcIndex[i]
                    v_start_curr_arc = current_arc_subsegment_length / delta_t_arcs[i][0]
                    
                    # For acceleration-based continuity: (v_start_curr_arc - v_end_prev_line)/t_avg_transition ∈ [-a_max, a_max]
                    # Note: This is a transition from linear to tangential velocity.
                    # The acceleration limit should be a_max (linear equivalent for the robot chassis)
                    t_avg_transition = (delta_t_lines[i-1][-1] + delta_t_arcs[i][0])/2 # Corrected
                    a_transition = (v_start_curr_arc - v_end_prev_line) / t_avg_transition
                    all_accelerations.append(a_transition) # NEW: Add to list
                    
                    # Calculate appropriate acceleration limit for this transition
                    # For line to arc transition, use the arc's acceleration limit
                    a_max_transition = calculate_angular_acceleration_limit(r0[i]) * r_w
                    
                    g.append(a_transition)
                    lbg.append(-a_max_transition)  # Limit the acceleration
                    ubg.append(a_max_transition)
                    
            else:  # First segment or relay point - start from rest
                # For arc segments starting from zero velocity, the maximum achievable angular velocity
                # is constrained by ω^2 = 2*α*θ where α is max angular acceleration and θ is angle
                g.append(delta_t_arcs[i][0])
            
                min_t = np.sqrt(2*arc_segment_length/ calculate_angular_acceleration_limit(r0[i]) /abs(r0[i]))
                lbg.append(min_t)  # Standard minimum time constrain
                ubg.append(5.0)  # Maximum physically achievable angular velocity
            min_t = np.sqrt(2*arc_segment_length/ calculate_angular_acceleration_limit(r0[i]) /abs(r0[i]))
            # Constraints for each segment within the arc
            for j in range(ArcIndex[i]):
                # Delta time lower bound (positive time)
                g.append(delta_t_arcs[i][j])
                lbg.append(0.20)  # Minimum time
                ubg.append(5.0)   # Maximum time
                
                # Angular velocity constraint: omega = arc_segment_length / r / delta_t <= w_max_arc
                omega_c = arc_segment_length / abs(r0[i]) / delta_t_arcs[i][j]
                w_max_arc = calculate_angular_velocity_limit(abs(r0[i]))
                g.append(omega_c)
                lbg.append(0)      # Non-negative angular velocity
                ubg.append(w_max_arc)  # Maximum angular velocity based on differential drive constraints
                
                # Angular acceleration constraint between consecutive segments
                if j > 0:
                    # a_ang = (v2-v1)/delta_t = (L/t2 - L/t1)/t_avg
                    # where L is arc_segment_length/r, and t_avg is (t1+t2)/2
                    a_ang = (arc_segment_length/abs(r0[i])) * (1/delta_t_arcs[i][j] - 1/delta_t_arcs[i][j-1]) / ((delta_t_arcs[i][j] + delta_t_arcs[i][j-1])/2)
                    all_accelerations.append(a_ang*abs(r0[i])) # NEW: Add to list
                    aw_max_arc = calculate_angular_acceleration_limit(r0[i])
                    g.append(a_ang)
                    lbg.append(-aw_max_arc)
                    ubg.append(aw_max_arc)
        
        # Create variables for line segments
        if LineIndex[i] > 0:
            line_counter +=1
            # Delta times for line segments
            # delta_t_line = ca.SX.sym(f'delta_t_line_{i}', LineIndex[i])
            # delta_t_lines.append(delta_t_line)
            delta_t_lines.append([ca.SX.sym(f'delta_t_line{line_counter}_{j}') for j in range(LineIndex[i])])
            # Line segment - compute segment length
            line_segment_length = l[i] / LineIndex[i]
            
            # ACCELERATION CONTINUITY: Arc (i) to Line (i)
            # This is for the case where segment i has BOTH an arc and a line component.
            # We need to ensure smooth velocity transition from the end of the arc to the start of the line.
            if ArcIndex[i] > 0:  # If this same segment i also has an arc component
                
                # Tangential velocity at the end of the arc's last subsegment in THIS segment i
                arc_segment_length = abs(r0[i] * (phi[i+1] - phi_new[i])) / ArcIndex[i]
                v_end_arc_i = arc_segment_length / delta_t_arcs[i][ArcIndex[i]-1]  # Last subsegment of current arc
                
                # Linear velocity at the start of the line's first subsegment in THIS segment i
                v_start_line_i = line_segment_length / delta_t_lines[line_counter-1][0]  # First subsegment of current line
                
                # For acceleration-based continuity: (v_start_line_i - v_end_arc_i)/t_avg ∈ [-a_max, a_max]
                t_avg = (delta_t_arcs[i][ArcIndex[i]-1] + delta_t_lines[line_counter-1][0])/2
                a_transition = (v_start_line_i - v_end_arc_i) / t_avg
                all_accelerations.append(a_transition) # Add to list for objective function penalty
                
                # Use linear acceleration limit for this transition
                g.append(a_transition)
                lbg.append(-a_max)
                ubg.append(a_max)
            
            # Constraints for each segment within the line
            for j in range(LineIndex[i]):
                # Delta time lower bound (positive time) - Increased minimum time to prevent very small values
                g.append(delta_t_lines[line_counter-1][j])
                lbg.append(0.1)  # Default minimum time (increased from 0.001 to 0.01)
                ubg.append(5.0)   # Maximum time
                
                # Linear velocity constraint: v = line_segment_length / delta_t <= v_max
                velocity_expr = line_segment_length / delta_t_lines[line_counter-1][j]
                g.append(velocity_expr)
                lbg.append(0)      # Non-negative velocity
                ubg.append(v_max)  # Maximum velocity based on differential drive constraints
                
                # Linear acceleration constraint between consecutive segments
                if j > 0:
                    # a_lin = (v2-v1)/delta_t = (L/t2 - L/t1)/t_avg
                    # where L is line_segment_length, and t_avg is (t1+t2)/2
                    a_lin = line_segment_length * (1/delta_t_lines[line_counter-1][j] - 1/delta_t_lines[line_counter-1][j-1]) / ((delta_t_lines[line_counter-1][j] + delta_t_lines[line_counter-1][j-1])/2)
                    all_accelerations.append(a_lin) # NEW: Add to list
                    g.append((delta_t_lines[line_counter-1][j]**2-delta_t_lines[line_counter-1][j-1]**2)/delta_t_lines[line_counter-1][j-1]/delta_t_lines[line_counter-1][j])
                    lbg.append(-a_max/2/line_segment_length)
                    ubg.append(a_max/2/line_segment_length)
            
            # Ensure stopping at the end of a line segment if it's a relay point or the final point.
            if i == N-2 or Flagb[i+1] != 0:
                # If the line segment 'i' needs to stop at its end (waypoint i+1) 
                # Second constraint: ensure minimum time for final segment to prevent excessive deceleration
                g.append(delta_t_lines[line_counter-1][-1])
                t_min=np.sqrt(2*line_segment_length/a_max)
                lbg.append(t_min)  # Enforce slightly higher minimum time for final segment
                ubg.append(5.0)  # Maximum time
                
    # Combine all delta time variables in an interleaved manner (arc + line for each segment)
    all_vars_flat = []
    for i in range(N-1):
        # Add arc variables for this segment if they exist
        arc_idx = -1
        for j, arc_vars in enumerate(delta_t_arcs):
            if j == sum(1 for k in range(i) if ArcIndex[k] > 0):
                arc_idx = j
                break
                
        if arc_idx != -1 and ArcIndex[i] > 0:
            if isinstance(delta_t_arcs[arc_idx], list):
                all_vars_flat.extend(delta_t_arcs[arc_idx])
            else:
                all_vars_flat.append(delta_t_arcs[arc_idx])
        
        # Add line variables for this segment if they exist
        line_idx = -1
        for j, line_vars in enumerate(delta_t_lines):
            if j == sum(1 for k in range(i) if LineIndex[k] > 0):
                line_idx = j
                break
                
        if line_idx != -1 and LineIndex[i] > 0:
            if isinstance(delta_t_lines[line_idx], list):
                all_vars_flat.extend(delta_t_lines[line_idx])
            else:
                all_vars_flat.append(delta_t_lines[line_idx])
    
    # Flatten all variables into a single optimization vector
    if all_vars_flat:
        opt_vars = ca.vertcat(*all_vars_flat)
        
        # Calculate total time for each relay-to-relay segment
        T_RL = ca.SX.sym('T_RL', N_relays+1)
        
        # Track segments between relay points
        relay_indices = [i for i, flag in enumerate(Flagb) if flag != 0]
        relay_indices = [0] + relay_indices  # Add start point
        if N-1 not in relay_indices:
            relay_indices.append(N-1)  # Add end point if not already included
        
        # Calculate total time for each path segment
        segment_times = []

        # NEW CORRECTED MAPPING:
        correct_arc_map_idx = []
        correct_line_map_idx = []
        
        arc_list_counter = 0
        line_list_counter = 0
        for i in range(N-1): # Iterate through N-1 major segments
            if ArcIndex[i] > 0:
                correct_arc_map_idx.append(arc_list_counter)
                arc_list_counter += 1
            else:
                correct_arc_map_idx.append(-1)
            
            if LineIndex[i] > 0:
                correct_line_map_idx.append(line_list_counter)
                line_list_counter += 1
            else:
                correct_line_map_idx.append(-1)
        
        # Now calculate time for each segment with proper indexing
        for i in range(N-1):
            segment_time_for_major_segment_i = 0 # Symbolic expression for total time of major segment i
            
            # Add arc time if exists
            arc_map_list_idx = correct_arc_map_idx[i]
            if arc_map_list_idx != -1:
                # Handle the new structure of delta_t_arcs (list of individual variables)
                if isinstance(delta_t_arcs[arc_map_list_idx], list):
                    segment_time_for_major_segment_i += ca.sum1(ca.vertcat(*delta_t_arcs[arc_map_list_idx]))
                else:
                    segment_time_for_major_segment_i += ca.sum1(delta_t_arcs[arc_map_list_idx])
                
            # Add line time if exists
            line_map_list_idx = correct_line_map_idx[i]
            if line_map_list_idx != -1:
                # Handle the new structure of delta_t_lines (list of individual variables)
                if isinstance(delta_t_lines[line_map_list_idx], list):
                    segment_time_for_major_segment_i += ca.sum1(ca.vertcat(*delta_t_lines[line_map_list_idx]))
                else:
                    segment_time_for_major_segment_i += ca.sum1(delta_t_lines[line_map_list_idx])
                
            segment_times.append(segment_time_for_major_segment_i)
        
        # Calculate relay-to-relay times
        for i in range(len(relay_indices)-1):
            start_idx = relay_indices[i]
            end_idx = relay_indices[i+1]
            
            # Sum times for segments between these relay points
            if end_idx > start_idx:
                # Create a list of symbolic expressions for segment times between relays
                relay_segment_times = segment_times[start_idx:end_idx]
                
                # Only perform vertcat if there are multiple segments
                if len(relay_segment_times) > 1:
                    T_RL[i] = ca.sum1(ca.vertcat(*relay_segment_times))
                elif len(relay_segment_times) == 1:
                    T_RL[i] = relay_segment_times[0]  # Just use the single segment time
                else:
                    T_RL[i] = 0
            else:
                T_RL[i] = 0
        
        # Calculate average relay-to-relay time for even distribution
        t_average = ca.sum1(T_RL) / (N_relays + 1)
        
        # Calculate total trajectory time
        total_time_expr = ca.sum1(opt_vars)
        
        # NEW PRIMARY OBJECTIVE: Match Target_time
        # High weight for matching the target time
        time_matching_weight = 10000.0
        objective = time_matching_weight * (total_time_expr - Target_time)**2
        
        # Secondary objective: Keep even distribution between relays
        objective += 20.0 * ca.sum1((T_RL - t_average)**2)
        
        # Add acceleration penalty to the objective for smoothness
        acceleration_penalty_weight = 500 # Reduced from 1000 to balance with the target time objective
        if all_accelerations: # Check if the list is not empty
            accel_terms_vector = ca.vertcat(*all_accelerations)
            objective += acceleration_penalty_weight * ca.sumsqr(accel_terms_vector)
            
        # Create an optimization problem
        nlp = {'x': opt_vars, 'f': objective, 'g': ca.vertcat(*g)}
        
        # Set solver options
        opts = {
            'ipopt.print_level': 5,   # More detailed output for debugging
            'print_time': 1,
            'ipopt.max_iter': 5000,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Initial guess: time distribution proportional to segment lengths
        x0 = np.ones(opt_vars.size1()) * 0.1  # Base value
        
        # Calculate total length of the trajectory for normalization
        total_trajectory_length = 0
        for i in range(N-1):
            # Add arc length if exists
            if ArcIndex[i] > 0:
                arc_length = abs(r0[i] * (phi[i+1] - phi_new[i]))
                total_trajectory_length += arc_length
            
            # Add line length if exists
            if LineIndex[i] > 0:
                total_trajectory_length += l[i]
        
        # Estimate a reasonable average velocity (e.g., half of max velocity)
        avg_velocity = v_max * 0.5
        
        
        # Adjust initial guess to be more realistic based on segment lengths
        idx = 0
        for i in range(N-1):
            if ArcIndex[i] > 0:
                arc_length = abs(r0[i] * (phi[i+1] - phi_new[i]))
                arc_segment_length = arc_length / ArcIndex[i]
                
                # Set initial delta_t based on a reasonable velocity
                for j in range(ArcIndex[i]):
                    # Time estimate: distance/velocity
                    x0[idx] = arc_segment_length / (avg_velocity * 0.1)
                    idx += 1
            
            if LineIndex[i] > 0:
                line_segment_length = l[i] / LineIndex[i]
                
                # Set initial delta_t based on a reasonable velocity
                for j in range(LineIndex[i]):
                    # Time estimate: distance/velocity
                    x0[idx] = line_segment_length / (avg_velocity*0.1)
                    idx += 1
        
        # Solve the optimization problem
        sol = solver(x0=x0, lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg))
        
        # Extract solution
        opt_time = sol['x'].full().flatten()
        
        # Calculate time segments for each part
        time_segments = []
        idx = 0
        
        for i in range(N-1):
            segment_times = {}
            if ArcIndex[i] > 0:
                segment_times['arc'] = opt_time[idx:idx+ArcIndex[i]].tolist()
                idx += ArcIndex[i]
            else:
                segment_times['arc'] = []
                
            if LineIndex[i] > 0:
                segment_times['line'] = opt_time[idx:idx+LineIndex[i]].tolist()
                idx += LineIndex[i]
            else:
                segment_times['line'] = []
                
            time_segments.append(segment_times)
        
        # Calculate total trajectory time
        total_time = float(np.sum(opt_time))
        
        # Save results
        results = {
            'time_segments': time_segments,
            'total_time': total_time,
        }
        
        
        return time_segments, total_time
    else:
        print("No variables to optimize!")
        return [], 0.0

def save_waypoints(case,N):
    nodes, in_neighbors, out_neighbors = load_reeb_graph('Graph_new_'+case+'.json')
    Waypoints, Flags, FlagB = load_WayPointFlag_from_file(f'WayPointFlag{N}'+case+'.json')
    phi,l,r,v,a=load_optimization_data(f'Optimization_ST_path{N}'+case+'.json')

    # Load environment obstacles
    try:
        with open(f'environment_{case}.json', 'r') as file:
            env_data = json.load(file)
        obstacles = env_data.get('obstacles', [])
        print(f"Loaded {len(obstacles)} obstacles from environment file")
    except Exception as e:
        print(f"Could not load environment file: {e}")
        obstacles = []
    WP=[]
    RP=[]
    for i in range(len(Waypoints)-1):
        Node={'Node':i,'Position':nodes[Waypoints[i]][1],'Orientation':phi[i],
                'Radius':r[i],
                'Length':l[i],'ArcVelocity':abs(v[2*i]/r[i]),'Velocity':v[2*i+1],
                'ArcAngular Velocity':abs(a[2*i]/r[i]),'Angular Velocity':a[2*i+1]}
        WP.append(Node)
        if Flags[i]==1 or i==0:
            Theta=phi[i]+FlagB[i]*np.pi/2
            RP_Ini={'Node':i,'Position':nodes[Waypoints[i]][1],'Orientation':Theta}
            RP.append(RP_Ini)
    Node={'Node':len(Waypoints)-1,'Position':nodes[Waypoints[len(Waypoints)-1]][1]}
    WP.append(Node)
    data={'Waypoints':WP,'RelayPoints':RP}
    save_file='/home/zhihui/data/'+case+'/Waypoints.json'
    with open(save_file, 'w') as file:
        json.dump(data, file)


def main():
    """
    Main function to be called when this script is run as a ROS2 node
    """
    import rclpy
    from rclpy.node import Node
    import os
    
    rclpy.init()
    
    # Create a ROS2 node
    planning_node = Node("planning_node")
    
    # Get parameters from ROS2 or use defaults
    planning_node.declare_parameter('case', 'simulation')
    planning_node.declare_parameter('n', 5)
    planning_node.declare_parameter('namespace', 'tb0')
    planning_node.declare_parameter('robot_id', 0)
    planning_node.declare_parameter('target_time', 30.0)  # Default target time (seconds)
    
    case = planning_node.get_parameter('case').value
    N = int(planning_node.get_parameter('n').value)
    namespace = planning_node.get_parameter('namespace').value
    robot_id = int(planning_node.get_parameter('robot_id').value)
    target_time = float(planning_node.get_parameter('target_time').value)
    # Set data directory
    data_dir = '/root/workspace/data/' + case + '/'
    
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    planning_node.get_logger().info(f"Starting planning for robot {namespace} with case '{case}' and N={N}")
    
    # Set up file paths
    waypoints_file_path= os.path.join(data_dir, f'Waypoints.json')
    path_file_path = os.path.join(data_dir, f'robot{robot_id}_path.json')
    result_file_path = os.path.join(data_dir, f'robot{robot_id}_optimized_path.json')
    
    # Log the paths we're using
    planning_node.get_logger().info(f"Using waypoints file: {waypoints_file_path}")
    planning_node.get_logger().info(f"Using robot path file: {path_file_path}")
    planning_node.get_logger().info(f"Will save results to: {result_file_path}")
    planning_node.get_logger().info(f"Target time: {target_time} seconds")
    
    # Check if the files exist
    if not os.path.exists(waypoints_file_path):
        planning_node.get_logger().warn(f"Waypoints file {waypoints_file_path} not found!")
        return 1
        
    if not os.path.exists(path_file_path):
        planning_node.get_logger().warn(f"Robot path file {path_file_path} not found!")
        return 1
    
    # Run the optimization
    time_segments, total_time = Planning_deltaT(
        waypoints_file_path=waypoints_file_path,
        Initial_Guess_file_path=path_file_path,
        Target_time=target_time
    )
    
    planning_node.get_logger().info(f"Optimization completed successfully!")
    planning_node.get_logger().info(f"Total trajectory time: {total_time:.4f} seconds")
    
    # Save the optimized path
    if os.path.exists(path_file_path):
        try:
            # Load the original robot path
            with open(path_file_path, 'r') as f:
                robot_path = json.load(f)
                
            # Update the time segments in the path
            if 'path_segments' in robot_path and len(time_segments) > 0:
                for i, segment in enumerate(robot_path['path_segments']):
                    if i < len(time_segments):
                        segment['time_segments'] = time_segments[i]
            
            # Save the updated path
            with open(result_file_path, 'w') as f:
                json.dump(robot_path, f, indent=2)
                
            planning_node.get_logger().info(f"Saved optimized path to {result_file_path}")
        except Exception as e:
            planning_node.get_logger().error(f"Error saving optimized path: {str(e)}")
    
    # Skip visualization steps that require missing dependencies
    # Just log that we're done
    planning_node.get_logger().info("Planning and time optimization complete")
    # Shutdown ROS2
    rclpy.shutdown()
    return 0

if __name__ == '__main__':
    main()





