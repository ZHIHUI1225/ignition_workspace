# This file contains functions for visualizing the trajectory with velocity color coding
import numpy as np
import matplotlib.pyplot as plt

# Try to import 3D plotting with fallback
try:
    from mpl_toolkits.mplot3d import Axes3D  # This import is needed to register the '3d' projection
    HAS_3D = True
except ImportError:
    print("Warning: 3D plotting not available. Will use 2D visualization only.")
    HAS_3D = False

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import json
import os

# Import config to get robot parameters with fallback
import sys
sys.path.append('/root/workspace/config')

# Import coordinate transformation utilities
from coordinate_transform import convert_pixel_to_world_coordinates

try:
    from config_loader import config
    # Get robot physical parameters from config
    robot_params = config.get_robot_physical_params()
    aw_max = robot_params['aw_max']  # the maximum angular acceleration
    w_max = robot_params['w_max']   # the maximum angular velocity
    r_limit = robot_params['r_limit']  # m
    r_w = robot_params['r_w']       # the radius of the wheel
    v_max = robot_params['v_max']   # m/s (pre-calculated in config)
    a_max = robot_params['a_max']   # (pre-calculated in config)
    l_r = robot_params['l_r']       # the wheel base
    print("Successfully loaded robot parameters from config")
except ImportError as e:
    print(f"Warning: Could not import config ({e}). Using default robot parameters.")

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

def plot_trajectory_with_time(waypoints, phi, r0, l, phi_new, time_segments, figure_file, reeb_graph, Flagb=None, case=None, N=None):
    """
    Visualize the trajectory with time information using available plotting capabilities.
    
    Args:
        waypoints: List of waypoint indices
        phi: Array of angles at waypoints
        r0: Array of arc radii
        l: Array of straight line lengths
        phi_new: Array of angles after each waypoint (considering relay points)
        time_segments: List of dictionaries containing arc and line time segments
        figure_file: Output file path for the figure
        reeb_graph: The reeb graph containing node positions
        Flagb: Array of relay flags (if None, will try to load from file)
        case: Case name for determining the waypoints file path
        N: Number for determining the waypoints file path
        
    This function creates visualization panels based on available capabilities:
    - If 3D is available: 3D trajectory plot with time as Z-axis, plus 2D views
    - If 3D not available: 2D trajectory with velocity color coding and analysis plots
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    # Access global constants for velocity and acceleration limits
    global v_max, a_max, w_max, aw_max

    # Create figure for all visualizations
    if HAS_3D:
        fig1 = plt.figure(figsize=(20, 12)) # 4 subplots with 3D
        # 3D trajectory visualization (top-left) - key fix for 3D projection
        ax1 = fig1.add_subplot(221, projection='3d')
    else:
        fig1 = plt.figure(figsize=(15, 10)) # 3 subplots without 3D
        ax1 = None  # No 3D plot available
    
    # Prepare arrays to store all points for visualization
    all_points_x = []
    all_points_y = []
    all_points_z = []  # Time coordinates
    all_velocities = []  # Array to store velocity at each point
    
    # Arrays to store segment markers
    segment_markers_x = []
    segment_markers_y = []
    segment_markers_z = []
    segment_types = []
    
    # Arrays to track segment-specific velocities for improved visualization
    arc_times_plot = []
    arc_velocities_plot = []
    line_times_plot = []
    line_velocities_plot = []
    segment_transitions = []  # Store segment transition information
    
    # Extract waypoint positions from reeb graph
    wp_x = []
    wp_y = []
    
    # Get Flagb from parameter or load from reeb_graph to identify relay points
    # If Flagb is not provided, we need to extract it from the process we're analyzing
    if Flagb is None:
        # Determine the waypoints file path from case and N parameters
        try:
            if case and N:
                # Use config to get proper path
                waypoints_file_path = os.path.join(config.data_path, f"WayPointFlag{N}{case}.json")
            else:
                # Fallback to default using config.data_path
                waypoints_file_path = os.path.join(config.data_path, "WayPointFlag3experi.json")
            
            print(f"Looking for waypoints file at: {waypoints_file_path}")
            with open(waypoints_file_path, 'r') as file:
                data = json.load(file)
                Flagb = data["FlagB"]
        except (FileNotFoundError, NameError) as e:
            print(f"Warning: Could not load Flagb data: {e}")
            # Fallback: create a default Flagb array with all zeros
            Flagb = [0] * len(waypoints)
    
    for wp_idx in waypoints:
        # Convert from pixels to world coordinates using shared coordinate transformation
        node_pos_pixels = reeb_graph.nodes[wp_idx].configuration
        world_pos = convert_pixel_to_world_coordinates(node_pos_pixels)
        
        wp_x.append(world_pos[0])
        wp_y.append(world_pos[1])
    
    # Initialize variables for calculating cumulative time
    cumulative_time = 0.0
    
    # Store waypoint arrival times for plotting
    waypoint_times = [0]  # Starting time at first waypoint    # Process each segment between waypoints
    for i in range(len(waypoints) - 1):
        # Get starting position and angle - Match Get_Trajectory.py exactly
        # In Get_Trajectory.py: phi1=phi[node[j]] +FlagB[node[j]]*np.pi/2
        # Safely access Flagb[i] with bounds checking
        flagb_i = Flagb[i] if i < len(Flagb) else 0
        phi1 = phi[i] + flagb_i*np.pi/2
        angle_start = phi1
        
        # Get waypoint position - Match Get_Trajectory.py coordinate system
        x_start = wp_x[i]  # This is already converted from cm to m in wp_x calculation
        y_start = wp_y[i]
        
        # Calculate arc center - this is needed for both arc and line calculations
        # From Get_Trajectory.py: r_x=nodes[Waypoints[node[j]]][1][0]/100-r[j]*np.cos(phi1+np.pi/2)
        # r_y=nodes[Waypoints[node[j]]][1][1]/100-r[j]*np.sin(phi1+np.pi/2)
        r_x = x_start - r0[i] * np.cos(phi1 + np.pi/2)
        r_y = y_start - r0[i] * np.sin(phi1 + np.pi/2)
        
        # If we're at the beginning, add the starting point
        if i == 0:
            all_points_x.append(x_start)
            all_points_y.append(y_start)
            all_points_z.append(cumulative_time)
            all_velocities.append(0.0)  # Starting velocity is zero
        
        # Process arc segment if it exists
        if i < len(time_segments) and 'arc' in time_segments[i] and len(time_segments[i]['arc']) > 0:
            arc_times = time_segments[i]['arc']
            
            # Calculate arc properties
            delta_phi = phi[i+1] - phi_new[i]
            arc_radius = r0[i]
            
            # Calculate arc center - Match Get_Trajectory.py exactly
            if  len(arc_times) > 0:  # Only process meaningful arcs
                # From Get_Trajectory.py:
                # r_x=nodes[Waypoints[node[j]]][1][0]/100-r[j]*np.cos(phi1+np.pi/2)
                # r_y=nodes[Waypoints[node[j]]][1][1]/100-r[j]*np.sin(phi1+np.pi/2)
                # Note: r[j] in Get_Trajectory.py corresponds to r0[i] here, and the sign matters
                # The r_x, r_y were already calculated above, so we can use them directly
                
                # Number of points to plot for smooth arc visualization
                N_arc = len(arc_times)
                
                # Calculate arc points - use phi1 as the starting angle like Get_Trajectory.py
                # Generate points along the arc following Get_Trajectory.py logic
                for j in range(1, N_arc + 1):  # Start from 1 as we already added the starting point
                    # Calculate angle progression
                    angle_fraction = j / N_arc
                    current_angle = phi1 + delta_phi * angle_fraction
                    
                    # Calculate position along arc - Match Get_Trajectory.py exactly:
                    # T.append([r[j]*np.cos(phi1+np.pi/2)+r_x,r[j]*np.sin(phi1+np.pi/2)+r_y,...])
                    point_x = r0[i] * np.cos(current_angle + np.pi/2) + r_x
                    point_y = r0[i] * np.sin(current_angle + np.pi/2) + r_y
                    
                    # Calculate time
                    cumulative_time += arc_times[j-1]
                    
                    # Calculate arc segment length and velocity (unified calculation)
                    total_arc_length = abs(arc_radius * delta_phi)
                    arc_segment_length = total_arc_length / N_arc
                    velocity = arc_segment_length / arc_times[j-1] if arc_times[j-1] > 0 else 0
                    
                    # Debug output for verification (can be removed in production)
                    if j == 1:  # Only print for first segment to avoid spam
                        print(f"Arc {i}: radius={arc_radius:.3f}m, delta_phi={delta_phi:.3f}rad, "
                              f"total_length={total_arc_length:.3f}m, segment_length={arc_segment_length:.3f}m, "
                              f"time={arc_times[j-1]:.3f}s, velocity={velocity:.3f}m/s")
                    
                    # Store points
                    all_points_x.append(point_x)
                    all_points_y.append(point_y)
                    all_points_z.append(cumulative_time)
                    all_velocities.append(velocity)
                    
                    # Mark the endpoint of the arc
                    if j == N_arc:
                        segment_markers_x.append(point_x)
                        segment_markers_y.append(point_y)
                        segment_markers_z.append(cumulative_time)
                        segment_types.append('arc_end')
                
                # Plot arc segment with time as z-coordinate (only if 3D is available)
                if HAS_3D and ax1 is not None:
                    ax1.plot(all_points_x[-N_arc:], all_points_y[-N_arc:], all_points_z[-N_arc:], 
                           'r-', linewidth=2, label='Arc' if i == 0 else "")
        
        # Get current position after the arc (if there was one)
        if len(all_points_x) > 0:
            x_after_arc = all_points_x[-1]
            y_after_arc = all_points_y[-1]
            # Update angle to phi[i+1] after arc processing
            angle_after_arc = phi[i+1]
        else:
            # Fallback if no points were added
            x_after_arc = x_start
            y_after_arc = y_start
            angle_after_arc = phi1
        
        # Process line segment if it exists
        if i < len(time_segments) and 'line' in time_segments[i] and len(time_segments[i]['line']) > 0:
            line_times = time_segments[i]['line']
            
            # Calculate line properties
            line_length = l[i]
            
            if line_length > 0.001:  # Only process meaningful lines
                # Number of points for line
                N_line = len(line_times)
                
                # Calculate line start point using the approach from Get_Trajectory.py
                # l_x=r[j]*np.cos(phi[node[j]+1]+np.pi/2)+r_x
                # l_y=r[j]*np.sin(phi[node[j]+1]+np.pi/2)+r_y
                # phi1=phi[node[j]+1]
                l_x = r0[i] * np.cos(phi[i+1] + np.pi/2) + r_x
                l_y = r0[i] * np.sin(phi[i+1] + np.pi/2) + r_y
                phi1_line = phi[i+1]
                
                # Generate points along the line using the correct starting point
                for j in range(N_line + 1):
                    if j == 0:
                        # Starting point is calculated from the arc center and radius
                        point_x = l_x
                        point_y = l_y
                    else:
                        # Calculate position along line using phi1_line for direction
                        segment_length = line_length / N_line
                        l_delta = segment_length * j
                        point_x = l_x + l_delta * np.cos(phi1_line)
                        point_y = l_y + l_delta * np.sin(phi1_line)
                    
                    # Calculate time and velocity
                    if j > 0:
                        cumulative_time += line_times[j-1]
                        # Calculate velocity: v = distance / time
                        line_segment_length = line_length / N_line
                        velocity = line_segment_length / line_times[j-1] if line_times[j-1] > 0 else 0
                        all_velocities.append(velocity)
                    else:
                        # If it's the first point, we don't have a time segment yet
                        # We'll use the last velocity from previous segment or 0
                        if len(all_velocities) > 0:
                            all_velocities.append(all_velocities[-1])
                        else:
                            all_velocities.append(0.0)
                    
                    # Store points
                    all_points_x.append(point_x)
                    all_points_y.append(point_y)
                    all_points_z.append(cumulative_time)
                    
                    # Mark segment endpoints
                    if j == N_line:
                        segment_markers_x.append(point_x)
                        segment_markers_y.append(point_y)
                        segment_markers_z.append(cumulative_time)
                        segment_types.append('line_end')
                
                # Plot line segment with time as z-coordinate (only if 3D is available)
                if HAS_3D and ax1 is not None:
                    ax1.plot(all_points_x[-N_line-1:], all_points_y[-N_line-1:], all_points_z[-N_line-1:], 
                           'b-', linewidth=2, label='Line' if (i == 0 or i >= len(time_segments) or 
                                                               'arc' not in time_segments[i] or 
                                                               len(time_segments[i]['arc']) == 0) else "")
    
    # Calculate waypoint arrival times
    waypoint_times = [0]  # First waypoint is at time 0
    for i in range(len(waypoints) - 1):
        if i < len(time_segments):
            segment_time = 0
            if 'arc' in time_segments[i]:
                segment_time += sum(time_segments[i]['arc'])
            if 'line' in time_segments[i]:
                segment_time += sum(time_segments[i]['line'])
            waypoint_times.append(waypoint_times[-1] + segment_time)
    
    # 3D plotting section (only if 3D is available)
    if HAS_3D and ax1 is not None:
        # Plot waypoints
        ax1.scatter(wp_x, wp_y, [0] * len(wp_x), color='red', s=100, marker='o', label='Waypoints')
        
        # Add labels to waypoints
        for i, (x, y) in enumerate(zip(wp_x, wp_y)):
            ax1.text(x, y, 0, f'WP{i}', fontsize=10)
        
        # Highlight relay points
        relay_indices = [i for i in range(len(Flagb)) if i < len(Flagb) and Flagb[i] != 0]
        if relay_indices:
            relay_x = [wp_x[i] for i in relay_indices]
            relay_y = [wp_y[i] for i in relay_indices]
            relay_z = [waypoint_times[i] for i in relay_indices]
            ax1.scatter(relay_x, relay_y, relay_z, color='green', s=150, marker='^', label='Relay Points')
            
            # Add labels to relay points
            for i, (x, y, z, idx) in enumerate(zip(relay_x, relay_y, relay_z, relay_indices)):
                ax1.text(x, y, z, f'RP{idx}', fontsize=10)
        
        # Plot segment markers as points in 3D
        for x, y, z, type_ in zip(segment_markers_x, segment_markers_y, segment_markers_z, segment_types):
            marker_color = 'red' if 'arc' in type_ else 'blue'
            ax1.scatter(x, y, z, color=marker_color, s=50, marker='o', alpha=0.7)
        
        # Add labels to the transition points
        for i, (x, y, z, type_) in enumerate(zip(segment_markers_x, segment_markers_y, segment_markers_z, segment_types)):
            if i % 2 == 0:  # Label every other point to avoid crowding
                ax1.text(x, y, z, f'{z:.2f}s', fontsize=8)
        
        # Set axis labels and title
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_zlabel('Time (seconds)')
        ax1.set_title('3D Trajectory with Time Information')
        
        # Set equal aspect ratio for x and y (but not z, which is time)
        if all_points_x and all_points_y: # Check if lists are not empty
            max_range = max([
                max(all_points_x) - min(all_points_x) if all_points_x else 0,
                max(all_points_y) - min(all_points_y) if all_points_y else 0
            ])
            if max_range == 0: max_range = 1 # Avoid division by zero if all points are the same
            mid_x = (max(all_points_x) + min(all_points_x)) / 2 if all_points_x else 0
            mid_y = (max(all_points_y) + min(all_points_y)) / 2 if all_points_y else 0
            ax1.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax1.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        
        # Add a grid and legend
        ax1.grid(True)
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
    
    # Create a 2D projection with velocity color-coded
    subplot_pos = 222 if HAS_3D else 131  # Adjust subplot position based on 3D availability
    ax2 = fig1.add_subplot(subplot_pos)
    
    # Create a colormap
    cmap = plt.cm.get_cmap('viridis')  # Using viridis (blue to yellow) for velocity
    
    # Get min and max velocity, preventing division by zero or negative values
    min_velocity = 0.0
    max_velocity = v_max # Use the global v_max for a consistent scale
    if all_velocities:
        actual_min_vel = min(v for v in all_velocities if v is not None)
        actual_max_vel = max(v for v in all_velocities if v is not None)
        min_velocity = max(0.0, actual_min_vel) # Ensure min_velocity is not negative
        max_velocity = max(min_velocity + 0.001, actual_max_vel) # Ensure max_velocity > min_velocity

    norm = Normalize(vmin=min_velocity, vmax=max_velocity)
    
    # Plot the trajectory with color based on velocity
    if len(all_points_x) > 1: # Ensure there are at least two points to draw a line
        for i in range(len(all_points_x) - 1):
            line_seg = np.array([[all_points_x[i], all_points_y[i]], 
                                 [all_points_x[i+1], all_points_y[i+1]]])
            # Create color based on velocity at this point
            # Ensure all_velocities[i] is not None before normalizing
            current_velocity = all_velocities[i] if all_velocities[i] is not None else 0.0
            color = cmap(norm(current_velocity))
            ax2.plot([line_seg[0, 0], line_seg[1, 0]], [line_seg[0, 1], line_seg[1, 1]], 
                    color=color, linewidth=3)
    
    # Plot segment transition markers
    for i, (x, y, z, type_) in enumerate(zip(segment_markers_x, segment_markers_y, segment_markers_z, segment_types)):
        marker_color = 'black'
        ax2.scatter(x, y, color=marker_color, s=50, marker='o', alpha=0.7)
        # Add time labels (still showing time for reference)
        ax2.text(x, y, f'{z:.2f}s', fontsize=8, ha='right')
    
    # Plot waypoints
    ax2.scatter(wp_x, wp_y, color='red', s=100, marker='o', label='Waypoints')
    
    # Add labels to waypoints
    for i, (x, y) in enumerate(zip(wp_x, wp_y)):
        ax2.text(x, y, f'WP{i}', fontsize=10)
    
    # Highlight relay points in 2D
    if relay_indices:
        relay_x = [wp_x[i] for i in relay_indices]
        relay_y = [wp_y[i] for i in relay_indices]
        ax2.scatter(relay_x, relay_y, color='green', s=150, marker='^', label='Relay Points')
        
        # Add labels to relay points
        for i, (x, y, idx) in enumerate(zip(relay_x, relay_y, relay_indices)):
            ax2.text(x, y, f'RP{idx}', fontsize=10)
    
    # Set axis labels and title
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('2D Trajectory with Velocity Color Coding')
    
    # Set equal aspect ratio
    ax2.set_aspect('equal')
    
    # Add a colorbar
    cbar = fig1.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity (m/s)')
    
    # Add a grid and legend
    ax2.grid(True)
    ax2.legend()
    
    # Set axis limits
    if all_points_x and all_points_y: # Check if lists are not empty
        ax2.set_xlim(min(all_points_x) - 0.1, max(all_points_x) + 0.1)
        ax2.set_ylim(min(all_points_y) - 0.1, max(all_points_y) + 0.1)
    
    # Create improved velocity vs. time plot
    subplot_pos_3 = 223 if HAS_3D else 132  # Adjust subplot position based on 3D availability
    ax3 = fig1.add_subplot(subplot_pos_3)  # Velocity vs time plot
    
    # Prepare data for segment-specific velocity visualization
    arc_times = []
    arc_velocities = []
    line_times = []
    line_velocities = []
    segment_transition_times = []
    segment_transition_velocities = []
    segment_labels = []
    
    # Extract segment information from time_segments
    cumulative_time = 0.0
    for i in range(len(waypoints) - 1):
        # Process arc segments
        if i < len(time_segments) and 'arc' in time_segments[i] and len(time_segments[i]['arc']) > 0:
            arc_times_i = time_segments[i]['arc']
            # Calculate arc properties
            delta_phi = phi[i+1] - phi_new[i]
            arc_radius = r0[i]
            
            if abs(delta_phi) > 0.001 and len(arc_times_i) > 0:
                # Calculate lengths and velocities for each arc subsegment (unified calculation)
                total_arc_length = abs(arc_radius * delta_phi)
                arc_segment_length = total_arc_length / len(arc_times_i)
                start_time = cumulative_time
                
                for j in range(len(arc_times_i)):
                    end_time = start_time + arc_times_i[j]
                    velocity = arc_segment_length / arc_times_i[j] if arc_times_i[j] > 0 else 0
                    
                    # Debug output for verification (can be removed in production)
                    if j == 0:  # Only print for first segment to avoid spam
                        print(f"Arc {i} (ax3): radius={arc_radius:.3f}m, delta_phi={delta_phi:.3f}rad, "
                              f"total_length={total_arc_length:.3f}m, segment_length={arc_segment_length:.3f}m, "
                              f"time={arc_times_i[j]:.3f}s, velocity={velocity:.3f}m/s")
                    
                    # Add points at both start and end of segment for proper line drawing
                    arc_times.append(start_time)
                    arc_velocities.append(velocity)
                    arc_times.append(end_time)
                    arc_velocities.append(velocity)
                    
                    start_time = end_time
                
                # Update cumulative time
                cumulative_time += sum(arc_times_i)
                
                # Mark the end of the arc segment
                segment_transition_times.append(cumulative_time)
                # Use the last velocity of the arc segment
                segment_transition_velocities.append(velocity)
                segment_labels.append(f'Arc {i} end')
        
        # Process line segments
        if i < len(time_segments) and 'line' in time_segments[i] and len(time_segments[i]['line']) > 0:
            line_times_i = time_segments[i]['line']
            line_length = l[i]
            
            if line_length > 0.001 and len(line_times_i) > 0:
                # Calculate lengths and velocities for each line subsegment
                line_segment_length = line_length / len(line_times_i)
                start_time = cumulative_time
                
                for j in range(len(line_times_i)):
                    end_time = start_time + line_times_i[j]
                    velocity = line_segment_length / line_times_i[j] if line_times_i[j] > 0 else 0
                    
                    # Add points at both start and end of segment for proper line drawing
                    line_times.append(start_time)
                    line_velocities.append(velocity)
                    line_times.append(end_time)
                    line_velocities.append(velocity)
                    
                    start_time = end_time
                
                # Update cumulative time
                cumulative_time += sum(line_times_i)
                
                # Mark the end of the line segment
                segment_transition_times.append(cumulative_time)
                # Use the last velocity of the line segment
                segment_transition_velocities.append(velocity)
                segment_labels.append(f'Line {i} end')
    
    # Plot arc velocities with improved separation
    if arc_times:
        for i in range(0, len(arc_times), 2):  # Plot each segment separately
            if i + 1 < len(arc_times):
                ax3.plot([arc_times[i], arc_times[i+1]], [arc_velocities[i], arc_velocities[i+1]], 
                        '-r', linewidth=2)
    
    # Plot line velocities with improved separation
    if line_times:
        for i in range(0, len(line_times), 2):  # Plot each segment separately
            if i + 1 < len(line_times):
                ax3.plot([line_times[i], line_times[i+1]], [line_velocities[i], line_velocities[i+1]], 
                        '-b', linewidth=2)
    
    # Add legend entries for velocity plots
    if arc_times:
        ax3.plot([], [], '-r', linewidth=2, label='Arc Velocities')
    if line_times:
        ax3.plot([], [], '-b', linewidth=2, label='Line Velocities')
    
    # Mark segment transitions
    if segment_transition_times:
        ax3.scatter(segment_transition_times, segment_transition_velocities, color='purple', 
                   s=50, marker='o', label='Segment Transitions')
        
        # Add labels to transition points (optional - might get crowded)
        for i, (t, v, label) in enumerate(zip(segment_transition_times, segment_transition_velocities, segment_labels)):
            # Label every other point to prevent overcrowding
            if i % 2 == 0:
                ax3.annotate(label, (t, v), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=8)
    
    # Plot waypoint arrival times
    for i, t in enumerate(waypoint_times[1:], 1):  # Skip the first waypoint time (0)
        ax3.axvline(x=t, color='green', linestyle='--', alpha=0.5)
        ax3.text(t, ax3.get_ylim()[1] * 0.9, f'WP{i}', rotation=90, fontsize=8)
    
    # Highlight relay points
    if relay_indices:
        relay_times = [waypoint_times[i] for i in relay_indices]
        for i, t in enumerate(relay_times):
            ax3.axvline(x=t, color='green', linestyle='-', linewidth=2, alpha=0.5)
            ax3.text(t, ax3.get_ylim()[1] * 0.95, f'RP{relay_indices[i]}', 
                    rotation=90, fontsize=10, color='green')
    
    # Set axis labels and title
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs. Time by Segment Type')
    ax3.grid(True)
    ax3.legend()
    
    # Set y-axis limits to include 0 and have some margin
    if arc_velocities or line_velocities:
        all_velocities_combined = arc_velocities + line_velocities
        max_vel = max(all_velocities_combined) if all_velocities_combined else 0
        ax3.set_ylim(0, max_vel * 1.1)
    
    # Add a detailed subplot for time segments visualization
    subplot_pos_4 = 224 if HAS_3D else 133  # Adjust subplot position based on 3D availability
    ax4 = fig1.add_subplot(subplot_pos_4)  # Time segment distribution plot
    
    # Prepare data for detailed time segment visualization
    segment_indices = []
    segment_durations = []  # Individual segment durations, not cumulative time
    segment_types = []
    segment_labels = []
    
    # Extract and prepare subsegment time data
    segment_idx = 0
    
    # Collect individual segment times for both arc and line segments
    for i in range(len(waypoints) - 1):
        if i < len(time_segments):
            # Process arc segments
            if 'arc' in time_segments[i] and time_segments[i]['arc']:
                arc_times = time_segments[i]['arc']
                if arc_times:
                    # Add each individual arc subsegment
                    for j, dt in enumerate(arc_times):
                        segment_indices.append(segment_idx)
                        segment_durations.append(dt)  # Just the duration, not cumulative
                        segment_types.append('arc')
                        segment_labels.append(f'Arc {i}.{j}')
                        segment_idx += 1
            
            # Process line segments
            if 'line' in time_segments[i] and time_segments[i]['line']:
                line_times = time_segments[i]['line']
                if line_times:
                    # Add each individual line subsegment
                    for j, dt in enumerate(line_times):
                        segment_indices.append(segment_idx)
                        segment_durations.append(dt)  # Just the duration, not cumulative
                        segment_types.append('line')
                        segment_labels.append(f'Line {i}.{j}')
                        segment_idx += 1
    
    # Plot individual segment times as a bar chart
    # Create bars with different colors for arc and line segments
    arc_indices = [idx for idx, type_label in zip(segment_indices, segment_types) if type_label == 'arc']
    arc_durations = [dur for dur, type_label in zip(segment_durations, segment_types) if type_label == 'arc']
    
    line_indices = [idx for idx, type_label in zip(segment_indices, segment_types) if type_label == 'line']
    line_durations = [dur for dur, type_label in zip(segment_durations, segment_types) if type_label == 'line']
    
    # Create the bar chart
    bar_width = 0.8
    if arc_indices:
        ax4.bar(arc_indices, arc_durations, width=bar_width, color='red', alpha=0.7, label='Arc Segments')
    if line_indices:
        ax4.bar(line_indices, line_durations, width=bar_width, color='blue', alpha=0.7, label='Line Segments')
    
    # Add duration text above each bar
    for idx, dur in zip(segment_indices, segment_durations):
        ax4.text(idx, dur + 0.02, f"{dur:.2f}s", ha='center', rotation=90, fontsize=8)
    
    # Add segment labels below x-axis for important segments
    step = max(1, len(segment_indices) // 10)  # Show roughly 10 labels to avoid overcrowding
    for i in range(0, len(segment_indices), step):
        idx = segment_indices[i]
        label = segment_labels[i]
        ax4.text(idx, -0.05, label, ha='center', rotation=90, fontsize=8)
    
    # Set axis labels and title
    ax4.set_xlabel('Segment Index')
    ax4.set_ylabel('Segment Duration (seconds)')
    ax4.set_title('Individual Segment Time Distribution')
    ax4.grid(True, axis='y')
    
    # Set x-axis limits
    ax4.set_xlim(-0.5, len(segment_indices) - 0.5)
    
    # Set y-axis limits to include all bars plus some margin
    if segment_durations:
        max_duration = max(segment_durations)
        ax4.set_ylim(0, max_duration * 1.2)  # 20% margin on top
    
    # Add a legend
    ax4.legend(loc='upper right')
    
    # Add total time text box
    total_time = sum(segment_durations)
    ax4.text(0.5, -0.2, f"Total Time: {total_time:.2f} seconds", 
             ha='center', va='center', transform=ax4.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    fig1.tight_layout() # Adjust layout to prevent overlapping titles/labels
    
    # Determine the correct save path
    import os
    if case:
        try:
            # Use config if available
            save_dir = config.get_case_data_dir(case)
        except (NameError, AttributeError) as e:
            print(f"Warning: Could not use config.get_case_data_dir: {e}")
            # Fallback if config is not available or missing the method
            if hasattr(config, 'data_path'):
                save_dir = os.path.join(config.data_path, case)
            else:
                save_dir = os.path.join('/root/workspace/data', case)
            
        print(f"Saving visualization to directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract filename from figure_file and save to the data directory
        filename = os.path.basename(figure_file)
        save_path = os.path.join(save_dir, filename)
    else:
        # Use the original figure_file path
        save_path = figure_file
        if os.path.dirname(figure_file):
            os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    
    # Save the combined figure with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Trajectory visualization saved to: {save_path}")
    
    plt.show() # Show all subplots in the main figure window

    # Return the total trajectory time for reference
    return cumulative_time
def plot_differential_drive_limits():
    """
    Plot the angular velocity and acceleration limits for different arc radii
    to visualize how the differential drive constraints affect trajectory planning.
    """
    # Define radius values to plot
    radii = np.linspace(0.1, 2.0, 100)  # Radius values from 0.1m to 2.0m
    
    # Calculate limits for each radius
    w_limits = [calculate_angular_velocity_limit(r) for r in radii]
    a_limits = [calculate_angular_acceleration_limit(r) for r in radii]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot angular velocity limits
    ax1.plot(radii, w_limits, 'b-', linewidth=2)
    ax1.axhline(y=w_max, color='r', linestyle='--', label='Wheel max ω')
    ax1.set_xlabel('Arc Radius (m)')
    ax1.set_ylabel('Maximum Angular Velocity (rad/s)')
    ax1.set_title('Angular Velocity Limits vs. Arc Radius')
    ax1.grid(True)
    ax1.legend()
    
    # Plot angular acceleration limits
    ax2.plot(radii, a_limits, 'g-', linewidth=2)
    ax2.axhline(y=aw_max, color='r', linestyle='--', label='Wheel max a')
    ax2.set_xlabel('Arc Radius (m)')
    ax2.set_ylabel('Maximum Angular Acceleration (rad/s²)')
    ax2.set_title('Angular Acceleration Limits vs. Arc Radius')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('differential_drive_limits.png')
    plt.show()
    return

def visualize_original_and_replanned_trajectories(original_file, replanned_file, figure_file, reeb_graph):
    """
    Visualize both the original and replanned trajectories.

    Args:
        original_file: Path to the JSON file containing the original trajectory.
        replanned_file: Path to the JSON file containing the replanned trajectory.
        figure_file: Path to save the visualization figure.
        reeb_graph: The reeb graph containing node positions.
    """
    import json
    
    # Ensure file paths are absolute using config if available
    try:
        # Try to use config for proper file paths if paths are not absolute
        if not os.path.isabs(original_file) and hasattr(config, 'data_path'):
            original_file = os.path.join(config.data_path, original_file)
        if not os.path.isabs(replanned_file) and hasattr(config, 'data_path'):
            replanned_file = os.path.join(config.data_path, replanned_file)
    except (NameError, AttributeError) as e:
        print(f"Warning: Could not process file paths using config: {e}")
    
    print(f"Loading original trajectory from: {original_file}")
    print(f"Loading replanned trajectory from: {replanned_file}")

    # Load original and replanned data
    with open(original_file, 'r') as f:
        original_data = json.load(f)

    with open(replanned_file, 'r') as f:
        replanned_data = json.load(f)

    # Extract waypoints, angles, radii, and time segments for original and replanned trajectories
    original_waypoints = original_data['waypoints']
    original_phi = original_data['phi']
    original_r0 = original_data['r0']
    original_l = original_data['l']
    original_phi_new = original_data['phi_new']
    original_time_segments = original_data['time_segments']

    replanned_waypoints = replanned_data['waypoints']
    replanned_phi = replanned_data['phi']
    replanned_r0 = replanned_data['r0']
    replanned_l = replanned_data['l']
    replanned_phi_new = replanned_data['phi_new']
    replanned_time_segments = replanned_data['time_segments']

    # Visualize original trajectory
    plot_trajectory_with_time(
        original_waypoints, original_phi, original_r0, original_l,
        original_phi_new, original_time_segments, figure_file.replace('.png', '_original.png'), reeb_graph
    )

    # Visualize replanned trajectory
    plot_trajectory_with_time(
        replanned_waypoints, replanned_phi, replanned_r0, replanned_l,
        replanned_phi_new, replanned_time_segments, figure_file.replace('.png', '_replanned.png'), reeb_graph
    )

    print(f"Visualizations saved as {figure_file.replace('.png', '_original.png')} and {figure_file.replace('.png', '_replanned.png')}")