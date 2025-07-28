# optimization the time of deleta l of the trajectory
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import json
import os
from GenerateMatrix import load_reeb_graph_from_file
import casadi as ca
from Environment import Environment
from Planing_functions import get_safe_corridor
import casadi.tools as ca_tools
from BarriersOriginal import generate_barriers_test
from trajectory_visualization import plot_trajectory_with_time
# Import the discretization tools
from discretization import compare_discretization_with_spline

# Import the trajectory saving function
from save_spline_trajectory import save_spline_trajectory

# Import trajectory parameter functions
from trajectory_parameters import save_trajectory_parameters, load_trajectory_parameters, plot_from_saved_trajectory, generate_spline_from_saved_trajectory

aw_max=0.2*np.pi# the maximum angular acceleration
w_max=0.5*np.pi # the maximum angular velocity
r_limit=0.75 # m
r_w=0.033 # the radius of the wheel
v_max=w_max*r_w# m/s
a_max=aw_max*r_w
l_r=0.14 # the wheel base
mu = 0.7  # Coefficient of friction (typical for rubber on concrete)
mu_f = 0.8  # Safety factor
g = 9.81  # Gravitational acceleration (m/s²)
mu_mu_f = mu * mu_f * g  


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
    """
    Load optimization data from a JSON file.
    For Initial_Guess files, only phi, l, r are expected.
    For ST path files, v and a might also be available.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Required fields
    phi = data["Optimization_phi"]
    l = data["Optimization_l"]
    r = data["Optimization_r"]
    
    # Optional fields - set to default values if not present
    if "Optimization_v" in data:
        v = data["Optimization_v"]
    else:
        print(f"Warning: 'Optimization_v' not found in {file_path}")
        # Create default values (zeros) with double the length of phi
        v = [0.0] * (len(phi) * 2)
    
    if "Optimization_a" in data:
        a = data["Optimization_a"]
    else:
        print(f"Warning: 'Optimization_a' not found in {file_path}")
        # Create default values (zeros) with double the length of phi
        a = [0.0] * (len(phi) * 2)
    
    return phi, l, r, v, a

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
    

def Planning_deltaT(waypoints_file_path,reeb_graph,Initial_Guess_file_path,Result_file,figure_file):
    Deltal=0.02 #m 
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
                    current_arc_idx = sum(1 for k in range(i+1) if ArcIndex[k] > 0) - 1
                    v_start_curr_arc = current_arc_subsegment_length / delta_t_arcs[current_arc_idx][0]
                    
                    # For acceleration-based continuity: (v_start_curr_arc - v_end_prev_line)/t_avg_transition ∈ [-a_max, a_max]
                    # Note: This is a transition from linear to tangential velocity.
                    # The acceleration limit should be a_max (linear equivalent for the robot chassis)
                    t_avg_transition = (delta_t_lines[i-1][-1] + delta_t_arcs[current_arc_idx][0])/2 # Corrected
                    a_transition = (v_start_curr_arc - v_end_prev_line) / t_avg_transition
                    all_accelerations.append(a_transition) # NEW: Add to list
                    
                    # Calculate appropriate acceleration limit for this transition
                    # For line to arc transition, use linear acceleration limit (robot chassis acceleration)
                    a_max_transition = a_max
                    
                    g.append(a_transition)
                    lbg.append(-a_max_transition)  # Limit the acceleration
                    ubg.append(a_max_transition)
                
                elif ArcIndex[i-1] > 0 and LineIndex[i-1] == 0:  # Previous segment ends with an arc (no line after it)
                    # Arc to arc continuity when there's no line segment between them
                    # Tangential velocity at the end of the previous arc's last subsegment
                    prev_arc_delta_phi = phi[i] - (phi_data[i-1] + Flagb[i-1]*np.pi/2)
                    prev_arc_length = abs(r0[i-1] * prev_arc_delta_phi)
                    prev_arc_subsegment_length = prev_arc_length / ArcIndex[i-1]
                    
                    # Find the correct index for the previous arc in delta_t_arcs
                    prev_arc_idx = sum(1 for k in range(i-1) if ArcIndex[k] > 0) - 1
                    v_end_prev_arc = prev_arc_subsegment_length / delta_t_arcs[prev_arc_idx][-1]
                    
                    # Tangential velocity at the start of the current arc's first subsegment
                    current_arc_subsegment_length = abs(r0[i] * (phi[i+1] - phi_new[i])) / ArcIndex[i]
                    current_arc_idx = sum(1 for k in range(i) if ArcIndex[k] > 0) - 1
                    v_start_curr_arc = current_arc_subsegment_length / delta_t_arcs[current_arc_idx][0]
                    
                    # For arc-to-arc continuity: ensure smooth velocity transition
                    t_avg_transition = (delta_t_arcs[prev_arc_idx][-1] + delta_t_arcs[current_arc_idx][0])/2
                    a_transition = (v_start_curr_arc - v_end_prev_arc) / t_avg_transition
                    all_accelerations.append(a_transition) # Add to list for objective function penalty
                    
                    # Use the more restrictive acceleration limit between the two arcs
                    # Convert angular acceleration limits to tangential acceleration limits
                    a_max_prev = calculate_angular_acceleration_limit(r0[i-1]) * abs(r0[i-1])
                    a_max_curr = calculate_angular_acceleration_limit(r0[i]) * abs(r0[i])
                    a_max_transition = min(a_max_prev, a_max_curr)
                    
                    g.append(a_transition)
                    lbg.append(-a_max_transition)  # Limit the acceleration
                    ubg.append(a_max_transition)
                    
            # Ensure arc segments that start at relay points begin from zero velocity  
            if i == 0 or Flagb[i] != 0:  # First segment or relay point - start from rest
                # For arc segments starting from zero velocity, the maximum achievable angular velocity
                # is constrained by ω^2 = 2*α*θ where α is max angular acceleration and θ is angle
                g.append(delta_t_arcs[arc_counter-1][0])
                min_t = np.sqrt(2*arc_segment_length / calculate_angular_acceleration_limit(r0[i]) / abs(r0[i]))
                lbg.append(min_t)  # Enforce minimum time for starting from zero velocity
                ubg.append(10.0)  # Maximum time
            # Constraints for each segment within the arc
            for j in range(ArcIndex[i]):
                # Delta time lower bound (positive time)
                g.append(delta_t_arcs[arc_counter-1][j])
                lbg.append(0.20)  # Minimum time
                ubg.append(10.0)   # Maximum time
                
                # Angular velocity constraint: omega = arc_segment_length / r / delta_t <= w_max_arc
                omega_c = arc_segment_length / abs(r0[i]) / delta_t_arcs[arc_counter-1][j]
                w_max_arc = calculate_angular_velocity_limit(abs(r0[i]))
                g.append(omega_c)
                lbg.append(0)      # Non-negative angular velocity
                ubg.append(w_max_arc)  # Maximum angular velocity based on differential drive constraints
                
                # Centripetal force constraint: ω_c²|r| ≤ μ μ_f g
                # This ensures the robot doesn't slide during curved motion
                g.append(omega_c)
                lbg.append(0)      # Non-negative centripetal force
                ubg.append(np.sqrt(mu_mu_f/ abs(r0[i]) )) # Maximum allowable centripetal force based on friction
                
                # Angular acceleration constraint between consecutive segments
                if j > 0:
                    # For arc motion: tangential acceleration = radius * angular acceleration
                    # a_tangential = (v2-v1)/t_avg where v = arc_segment_length/delta_t
                    v1 = arc_segment_length / delta_t_arcs[arc_counter-1][j-1]
                    v2 = arc_segment_length / delta_t_arcs[arc_counter-1][j]
                    t_avg = (delta_t_arcs[arc_counter-1][j-1] + delta_t_arcs[arc_counter-1][j])/2
                    a_tangential = (v2 - v1) / t_avg
                    all_accelerations.append(a_tangential) # Add tangential acceleration to list
                    
                    # Convert to angular acceleration: alpha = a_tangential / radius
                    alpha = a_tangential / abs(r0[i])
                    aw_max_arc = calculate_angular_acceleration_limit(r0[i])
                    g.append(alpha)
                    lbg.append(-aw_max_arc)
                    ubg.append(aw_max_arc)
                    
                    # Additional angular velocity and acceleration constraint: |a_c| ≥ ω²_c/μ - μ_f/|r|
                    # This ensures balanced centripetal and angular acceleration for safe curved motion
                    # Rearranged as: |a_c| - ω²_c/μ ≥ -μ_f/|r|
                    omega_c_current = arc_segment_length / abs(r0[i]) / delta_t_arcs[arc_counter-1][j]
                    centripetal_term = omega_c_current**2 / mu
                    friction_term = -mu_f / abs(r0[i])
                    
                    # For positive angular acceleration: a_c - ω²_c/μ ≥ -μ_f/|r|
                    g.append(alpha - centripetal_term)
                    lbg.append(friction_term)
                    ubg.append(ca.inf)
                    
                    # For negative angular acceleration: -a_c - ω²_c/μ ≥ -μ_f/|r|
                    g.append(-alpha - centripetal_term)
                    lbg.append(friction_term)
                    ubg.append(ca.inf)
            
            # Enhanced constraints for arc segments ending at relay points or final point (no line segment following)
            if (i == N-2 or Flagb[i+1] != 0) and LineIndex[i] == 0:
                # If the arc segment 'i' needs to stop at its end (waypoint i+1) and there's no line segment
                
                # 1. Ensure minimum time for final arc segment to prevent excessive deceleration
                g.append(delta_t_arcs[arc_counter-1][-1])
                t_min_final = np.sqrt(2*arc_segment_length / calculate_angular_acceleration_limit(r0[i]) / abs(r0[i]))
                # Calculate maximum time based on "stop" velocity threshold (0.01 m/s)
                t_max_stop_velocity = arc_segment_length / 0.01  # Time to reach stop velocity
                # Ensure t_max is always greater than or equal to t_min to avoid infeasible constraints
                t_max_final = max(t_max_stop_velocity, t_min_final * 1.1)  # Add 10% buffer above minimum
                lbg.append(t_min_final)  # Enforce minimum time for final arc segment
                ubg.append(t_max_final)  # Maximum time based on stop velocity threshold with feasibility check
        
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
                v_end_arc_i = arc_segment_length / delta_t_arcs[arc_counter-1][ArcIndex[i]-1]  # Last subsegment of current arc
                
                # Linear velocity at the start of the line's first subsegment in THIS segment i
                v_start_line_i = line_segment_length / delta_t_lines[line_counter-1][0]  # First subsegment of current line
                
                # For acceleration-based continuity: (v_start_line_i - v_end_arc_i)/t_avg ∈ [-a_max, a_max]
                t_avg = (delta_t_arcs[arc_counter-1][ArcIndex[i]-1] + delta_t_lines[line_counter-1][0])/2
                a_transition = (v_start_line_i - v_end_arc_i) / t_avg
                all_accelerations.append(a_transition) # Add to list for objective function penalty
                
                # Use linear acceleration limit for this transition
                g.append(a_transition)
                lbg.append(-a_max)
                ubg.append(a_max)
            
            # Ensure starting from zero velocity for line segments that start at relay points or first segment AND have no arc before them
            if ArcIndex[i] == 0 and (i == 0 or Flagb[i] != 0):
                # If the line segment 'i' starts at the beginning (waypoint i) and there's no arc segment before it
                # Ensure minimum time for first line segment to start from zero velocity
                g.append(delta_t_lines[line_counter-1][0])
                t_min = np.sqrt(2*line_segment_length/a_max)
                lbg.append(t_min)  # Enforce minimum time for first line segment starting from zero
                ubg.append(5.0)  # Maximum time
            
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
                # Calculate maximum time based on "stop" velocity threshold (0.01 m/s)
                t_max_stop_velocity_line = line_segment_length / 0.01  # Time to reach stop velocity
                # Ensure t_max is always greater than or equal to t_min to avoid infeasible constraints
                t_max_final_line = max(t_max_stop_velocity_line, t_min * 1.1)  # Add 10% buffer above minimum
                lbg.append(t_min)  # Enforce slightly higher minimum time for final segment
                ubg.append(t_max_final_line)  # Maximum time based on stop velocity threshold with feasibility check
                

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
        
        # Objective function: minimize total time + deviation from average time
        objective = ca.sum1(opt_vars) + 40.0 * ca.sum1((T_RL - t_average)**2)
        
        # NEW: Add acceleration penalty to the objective with significantly increased weight
        acceleration_penalty_weight = 1000 # Increased from 100 to 500 for stronger smoothness enforcement
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
        
        with open(Result_file, 'w') as file:
            json.dump(results, file)
        
        return time_segments, total_time
    else:
        print("No variables to optimize!")
        return [], 0.0

def save_waypoints(case,N,data_file=None):
    nodes, in_neighbors, out_neighbors = load_reeb_graph('Graph_new_'+case+'.json')
    Waypoints, Flags, FlagB = load_WayPointFlag_from_file(f'WayPointFlag{N}'+case+'.json')
    
    # Load only phi, l, r from the file (no v, a needed)
    with open(data_file, 'r') as file:
        data = json.load(file)
    phi = data["Optimization_phi"]
    l = data["Optimization_l"]
    r = data["Optimization_r"]

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
        # Simplified node structure without velocity and acceleration information
        Node={'Node':i,'Position':nodes[Waypoints[i]][1],'Orientation':phi[i],
                'Radius':r[i],
                'Length':l[i]}
        WP.append(Node)
        if Flags[i]==1 or i==0:
            Theta=phi[i]+FlagB[i]*np.pi/2
            RP_Ini={'Node':i,'Position':nodes[Waypoints[i]][1],'Orientation':Theta}
            RP.append(RP_Ini)
    
    Node={'Node':len(Waypoints)-1,'Position':nodes[Waypoints[len(Waypoints)-1]][1]}
    WP.append(Node)
    data={'Waypoints':WP,'RelayPoints':RP}
    
    # Ensure directory exists
    import os
    os.makedirs('/home/zhihui/data/'+case, exist_ok=True)
    
    save_file='/home/zhihui/data/'+case+'/Waypoints.json'
    with open(save_file, 'w') as file:
        json.dump(data, file)
    
    print(f"Waypoints saved to {save_file}")


if __name__ == '__main__':
    # Example usage
    # case="simple_maze"
    # case="simulation"
    case="maze_5"
    N=5
    # file_path = "Graph_"+case+".json"
    file_path = "Graph_new_"+case+".json"
    reeb_graph = load_reeb_graph_from_file(file_path)
    NumNodes=len(reeb_graph.nodes)
    environment_file= "environment_"+case+".json"
    Start_node=reeb_graph.nodes[NumNodes-2].configuration
    End_node=reeb_graph.nodes[NumNodes-1].configuration
    Distances=np.linalg.norm(End_node-Start_node)
    
    # Output files
    assignment_result_file = f"AssignmentResult{N}"+case+".json"
    Initial_Guess_file_path=f"Optimization_GA_{N}_IG_norma"+case+".json"
    Initial_Guess_figure_file=f"InitialGuess{N}"+case+".png"
    waypoints_file_path=f"WayPointFlag{N}"+case+".json"
    Initial_Guess_file=f"Optimization_withSC_path{N}"+case+".json"
    result_file=f"Optimization_deltaT_{N}"+case+".json"
    figure_file=f"Optimization_deltaT_{N}"+case+".png"
    
    # Save waypoints for visualization (with simplified structure)
    try:
        print(f"Saving waypoints from {Initial_Guess_file}...")
        save_waypoints(case, N, Initial_Guess_file)
    except Exception as e:
        print(f"Warning: Could not save waypoints: {e}")
        print("Continuing with optimization...")
    # Run the optimization
    time_segments, total_time = Planning_deltaT(
        waypoints_file_path=waypoints_file_path,
        reeb_graph=reeb_graph,
        Initial_Guess_file_path=Initial_Guess_file,
        Result_file=result_file,
        figure_file=figure_file
    )
    
    print(f"Optimization completed successfully!")
    print(f"Total trajectory time: {total_time:.4f} seconds")
    
    # Generate visualization of differential drive constraints
    # plot_differential_drive_limits()
    
    # Load waypoints to visualize trajectory with time information
    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
    phi_data = np.array(load_phi_from_file(Initial_Guess_file))
    r_guess = np.array(load_r_from_file(Initial_Guess_file))/100 #cm to m 
    l_guess = np.array(load_l_from_file(Initial_Guess_file))/100
    phi_new = np.zeros(np.shape(phi_data))
    
    # Calculate phi_new for visualization
    for i in range(len(Waypoints)-1):
        phi_new[i] = phi_data[i] + Flagb[i]*np.pi/2
    
    # NEW: Save trajectory parameters for later use
    trajectory_files = save_trajectory_parameters(
        waypoints=Waypoints,
        phi=phi_data,
        r0=r_guess,
        l=l_guess,
        phi_new=phi_new,
        time_segments=time_segments,
        Flagb=Flagb,
        reeb_graph=reeb_graph,
        case=case,
        N=N
    )
    
    # Plot trajectory with time information
    plot_trajectory_with_time(
        waypoints=Waypoints,
        phi=phi_data,
        r0=r_guess,
        l=l_guess,
        phi_new=phi_new,
        time_segments=time_segments,
        figure_file=figure_file,
        reeb_graph=reeb_graph,
        Flagb=Flagb,
        case=case,
        N=N
    )
    
    # Compare the original discrete trajectory with cubic spline interpolation
    # and save the trajectory to JSON files (one file per relay segment)
    saved_files = compare_discretization_with_spline(
        waypoints=Waypoints,
        phi=phi_data,
        r0=r_guess,
        l=l_guess,
        phi_new=phi_new,
        time_segments=time_segments,
        Flagb=Flagb,
        reeb_graph=reeb_graph,
        dt=0.5,  # For comparison
        save_dir='/home/zhihui/data/'+case+'/'
    )
    
    print(f"\nTrajectory parameters saved for {len(trajectory_files) - 1} robots")
    print("Saved files:")
    for file_path in trajectory_files:
        print(f"  - {file_path}")
    print("\nUse the following functions to reload and process:")
    print(f"  - load_complete_trajectory_parameters('{case}')")
    print(f"  - load_robot_trajectory_parameters('{case}', robot_id)")
    print(f"  - load_robot_trajectory_parameters('{case}')  # Load all robots")





