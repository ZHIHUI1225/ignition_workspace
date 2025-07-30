import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.interpolate import CubicSpline
from discretization import discretize_segment

import sys
sys.path.append('/root/workspace/config')
from config_loader import config

def save_spline_trajectory(waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph, dt=0.1, save_dir=None):
    # Use config data path if save_dir not specified
    if save_dir is None:
        save_dir = config.data_path
    """
    Generate discretized trajectory using cubic spline interpolation and save it to JSON files.
    Each trajectory between relay points will be saved as a separate file named "tb{i}_Trajectory.json".
    
    Args:
        waypoints: List of waypoint indices in the reeb graph
        phi: List of angles at each waypoint
        r0: List of arc radii for each segment
        l: List of line lengths for each segment
        phi_new: List of adjusted angles accounting for flag values
        time_segments: List of dictionaries with 'arc' and 'line' time values for each segment
        Flagb: List of flag values for each waypoint
        reeb_graph: The reeb graph containing waypoint coordinates
        dt: Time step for uniform sampling
        save_dir: Directory to save the output files
        
    Returns:
        List of file paths where the trajectories were saved
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get original discrete trajectory (just the endpoints from each segment)
    N = len(waypoints) - 1
    
    # Collect all discretized segments
    all_times, all_xs, all_ys = [], [], []
    total_time = 0.0
    
    # Store segment start/end indices for relay points
    segment_boundaries = {}
    
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
            all_times.extend(t_seg[1:])  # Skip the first point (already included from previous segment)
            all_xs.extend(x_seg[1:])
            all_ys.extend(y_seg[1:])
        else:
            all_times.extend(t_seg)
            all_xs.extend(x_seg)
            all_ys.extend(y_seg)
        
        # Update total time for next segment
        if len(t_seg) > 0:
            total_time = t_seg[-1]
        
        # Store boundary if this is a relay point
        if Flagb[i] != 0 or i == 0:
            segment_boundaries[i] = len(all_times) - 1
    
    # Add the last waypoint as a segment boundary
    segment_boundaries[N] = len(all_times) - 1
    
    # Get cubic spline interpolation at fixed time intervals
    t_uniform = np.arange(0, total_time, dt)
    
    # Spline interpolation
    cs_x = CubicSpline(all_times, all_xs)
    cs_y = CubicSpline(all_times, all_ys)
    
    x_uniform = cs_x(t_uniform)
    y_uniform = cs_y(t_uniform)
    
    # Calculate orientations (directions)
    thetas = []
    for i in range(len(x_uniform) - 1):
        dx = x_uniform[i+1] - x_uniform[i]
        dy = y_uniform[i+1] - y_uniform[i]
        theta = np.arctan2(dy, dx)
        thetas.append(theta)
    # For the last point, use the same orientation as the second-to-last point
    if len(x_uniform) > 1:
        thetas.append(thetas[-1])
    else:
        thetas.append(0)  # Default orientation if only one point
    
    # Calculate velocities and angular velocities
    velocities = []
    angular_velocities = []
    
    for i in range(len(x_uniform) - 1):
        dx = x_uniform[i+1] - x_uniform[i]
        dy = y_uniform[i+1] - y_uniform[i]
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance / dt
        velocities.append(velocity)
        
        # Angular velocity (change in orientation divided by dt)
        if i < len(thetas) - 2:
            dtheta = (thetas[i+1] - thetas[i])
            # Normalize to [-pi, pi]
            while dtheta > np.pi:
                dtheta -= 2 * np.pi
            while dtheta < -np.pi:
                dtheta += 2 * np.pi
            angular_velocity = dtheta / dt
        else:
            angular_velocity = 0  # Last point has no angular velocity
        
        angular_velocities.append(angular_velocity)
    
    # For the last point, use the same velocity as the second-to-last point
    if len(velocities) > 0:
        velocities.append(velocities[-1])
        angular_velocities.append(0)  # Angular velocity is likely zero at the end
    else:
        velocities.append(0)  # Default velocity if only one point
        angular_velocities.append(0)
    
    # Create trajectory points as [x, y, theta, v, w]
    trajectory_points = []
    for i in range(len(x_uniform)):
        point = [float(x_uniform[i]), float(y_uniform[i]), float(thetas[i]), 
                float(velocities[i]), float(angular_velocities[i])]
        trajectory_points.append(point)
    
    # Identify relay points (based on Flagb values)
    relay_indices = [i for i, flag in enumerate(Flagb) if flag != 0 or i == 0]
    relay_indices.append(len(waypoints)-1)  # Add the last waypoint
    relay_indices = sorted(list(set(relay_indices)))  # Remove duplicates and sort
    
    # Create time mappings for relay points
    relay_time_fractions = {}
    for idx in relay_indices:
        if idx == 0:
            relay_time_fractions[0] = 0.0  # Start point is at t=0
        elif idx == len(waypoints)-1:
            relay_time_fractions[idx] = 1.0  # End point is at t=total_time
        else:
            # Approximate the time fraction based on waypoint index
            relay_time_fractions[idx] = idx / (len(waypoints) - 1)
    
    # Save trajectories between relay points
    saved_files = []
    
    for i in range(len(relay_indices) - 1):
        start_idx = relay_indices[i]
        end_idx = relay_indices[i+1]
        
        # Calculate corresponding indices in the interpolated trajectory
        start_time_frac = relay_time_fractions[start_idx]
        end_time_frac = relay_time_fractions[end_idx]
        
        start_t_idx = int(start_time_frac * len(t_uniform))
        end_t_idx = int(end_time_frac * len(t_uniform))
        
        # Ensure we don't go out of bounds
        start_t_idx = max(0, start_t_idx)
        end_t_idx = min(end_t_idx, len(trajectory_points))
        
        # Extract the trajectory segment
        segment = trajectory_points[start_t_idx:end_t_idx+1]
        
        if segment:
            # Save to JSON file
            file_path = os.path.join(save_dir, f'tb{i}_Trajectory.json')
            with open(file_path, 'w') as f:
                json.dump({"Trajectory": segment}, f)
            saved_files.append(file_path)
            
            print(f"Saved trajectory segment {i} to {file_path}")
    
    # Save the full trajectory too
    full_path = os.path.join(save_dir, 'full_trajectory_spline.json')
    with open(full_path, 'w') as f:
        json.dump({"Trajectory": trajectory_points}, f)
    saved_files.append(full_path)
    
    print(f"Saved full trajectory to {full_path}")
    
    return saved_files

if __name__ == "__main__":
    from GenerateMatrix import load_reeb_graph_from_file
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        case = sys.argv[1]
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    else:
        case = "simple_maze"
        N = 3
    
    print(f"Processing case: {case}, N={N}")
    
    try:
        # Load required data
        from Planning_deltaT import load_WayPointFlag_from_file, load_phi_from_file, load_r_from_file, load_l_from_file
        
        # Load data files
        file_path = f"Graph_new_{case}.json"
        reeb_graph = load_reeb_graph_from_file(file_path)
        
        waypoints_file_path = f"WayPointFlag{N}{case}.json"
        Initial_Guess_file = f"Optimization_ST_path{N}{case}.json"
        result_file = f"Optimization_deltaT_{N}{case}.json"
        
        Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
        phi_data = np.array(load_phi_from_file(Initial_Guess_file))
        # Convert from pixels to meters using config utility
        r_pixels = np.array(load_r_from_file(Initial_Guess_file))
        l_pixels = np.array(load_l_from_file(Initial_Guess_file))
        r_guess = config.pixels_to_meters(r_pixels)  # pixels to m
        l_guess = config.pixels_to_meters(l_pixels)  # pixels to m
        
        # Calculate phi_new for interpolation
        phi_new = np.zeros(len(phi_data))
        for i in range(len(Waypoints)-1):
            phi_new[i] = phi_data[i] + Flagb[i]*np.pi/2
        
        # Load time segments
        with open(result_file, 'r') as f:
            result_data = json.load(f)
            time_segments = result_data["time_segments"]
        
        # Generate and save discretized trajectories
        save_spline_trajectory(
            waypoints=Waypoints,
            phi=phi_data,
            r0=r_guess,
            l=l_guess,
            phi_new=phi_new,
            time_segments=time_segments,
            Flagb=Flagb,
            reeb_graph=reeb_graph,
            dt=0.1,
            save_dir=config.data_path
        )
        
        print("Trajectory discretization and saving completed successfully.")
    
    except Exception as e:
        print(f"Error: {e}")
