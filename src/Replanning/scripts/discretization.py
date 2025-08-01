import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import sys
sys.path.append('/root/workspace/config')
from config_loader import config

# Import coordinate transformation utilities
from coordinate_transform import convert_world_pixel_to_world_meter

def discretize_segment(segment_idx, waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph):
    """
    Discretize a single major segment (arc+line) independently using just the endpoints of each subsegment.
    
    Args:
        segment_idx: Index of the segment to discretize (0 to N-2)
        waypoints: List of waypoint indices in the reeb graph
        phi: List of angles at each waypoint
        r0: List of arc radii for each segment
        l: List of line lengths for each segment
        phi_new: List of adjusted angles accounting for flag values
        time_segments: List of dictionaries with 'arc' and 'line' time values for each segment
        Flagb: List of flag values for each waypoint
        reeb_graph: The reeb graph containing waypoint coordinates
        
    Returns:
        x_discrete: List of x-coordinates for the discretized segment
        y_discrete: List of y-coordinates for the discretized segment
        t_discrete: List of time points for the discretized segment
    """
    # Extract segment info
    i = segment_idx
    seg_times = time_segments[i]
    # Convert from pixels to world coordinates using shared coordinate transformation
    pos_pixels = reeb_graph.nodes[waypoints[i]].configuration
    pos = convert_world_pixel_to_world_meter(pos_pixels)
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
    if arc_times:
        dphi = phi[i+1] - phi_new[i]
        r = r0[i]
        # Calculate center of arc
        cx = pos[0] - r*np.cos(angle + np.pi/2)
        cy = pos[1] - r*np.sin(angle + np.pi/2)
        n_arc = len(arc_times)
        
        # For each subsegment of the arc
        for j, dur in enumerate(arc_times):
            # Calculate end angle for this subsegment
            ang_end = angle + dphi*(j+1)/n_arc
            
            # Add only the endpoint
            x_end = cx + r*np.cos(ang_end + np.pi/2)
            y_end = cy + r*np.sin(ang_end + np.pi/2)
            t_curr += dur
            
            x_discrete.append(x_end)
            y_discrete.append(y_end)
            t_discrete.append(t_curr)
            
            # Update position for next subsegment
            pos = (x_end, y_end)
        
    # Process line segment (if it exists)
    if line_times:
        seglen = l[i]
        phi_line = phi[i+1]
        n_line = len(line_times)
        
        # For each subsegment of the line
        for j, dur in enumerate(line_times):
            # Calculate line subsegment distance
            dx = np.cos(phi_line) * (seglen/n_line)
            dy = np.sin(phi_line) * (seglen/n_line)
            
            # Add only the endpoint
            x_end = pos[0] + dx
            y_end = pos[1] + dy
            t_curr += dur
            
            x_discrete.append(x_end)
            y_discrete.append(y_end)
            t_discrete.append(t_curr)
            
            # Update position for next subsegment
            pos = (x_end, y_end)
    
    # Handle edge case: if this is the last segment (and has no line component)
    # Make sure position matches the endpoint of the trajectory
    if i == len(waypoints) - 2 and l[i] < 1e-3 and not line_times:
        # Get the final waypoint position
        # Convert from pixels to world coordinates using shared coordinate transformation
        final_pos_pixels = reeb_graph.nodes[waypoints[i+1]].configuration
        final_pos = convert_world_pixel_to_world_meter(final_pos_pixels)
        
        # If the final position is not close enough to our last point, add it
        if len(x_discrete) > 0:
            last_x, last_y = x_discrete[-1], y_discrete[-1]
            dist = np.sqrt((last_x - final_pos[0])**2 + (last_y - final_pos[1])**2)
            if dist > 1e-3:  # If points are not close enough
                x_discrete.append(float(final_pos[0]))
                y_discrete.append(float(final_pos[1]))
                t_discrete.append(float(t_curr))  # Use the same time as the last point
    
    return x_discrete, y_discrete, t_discrete



def visualize_segment_parts(waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph):
    """
    Visualize each segment with arc and line parts displayed separately.
    
    Args:
        waypoints: List of waypoint indices in the reeb graph
        phi: List of angles at each waypoint
        r0: List of arc radii for each segment
        l: List of line lengths for each segment
        phi_new: List of adjusted angles accounting for flag values
        time_segments: List of dictionaries with 'arc' and 'line' time values for each segment
        Flagb: List of flag values for each waypoint
        reeb_graph: The reeb graph containing waypoint coordinates
    """
    N = len(waypoints) - 1
    
    plt.figure(figsize=(14, 10))
    
    # Define markers and colors
    arc_marker = '-'
    line_marker = '--'
    arc_color = 'blue'
    line_color = 'red'
    
    # Prepare for legend
    arc_line = plt.Line2D([0], [0], color=arc_color, linestyle=arc_marker, label='Arc Segment')
    line_line = plt.Line2D([0], [0], color=line_color, linestyle=line_marker, label='Line Segment')
    
    # Process each segment
    for i in range(N):
        # Get the segment info
        seg_times = time_segments[i]
        # Convert from world_pixel to world_meter coordinates using shared coordinate transformation
        start_pos_pixels = reeb_graph.nodes[waypoints[i]].configuration
        start_pos_world_meter = convert_world_pixel_to_world_meter(start_pos_pixels)
        start_pos = [start_pos_world_meter[0], start_pos_world_meter[1]]
        
        angle = phi[i] + Flagb[i]*np.pi/2
        
        # Separate arc and line parts
        arc_times = seg_times.get('arc', [])
        line_times = seg_times.get('line', [])
        
        # Process arc segment (if it exists)
        if arc_times:
            dphi = phi[i+1] - phi_new[i]
            r = r0[i]
            
            # Calculate center of arc
            cx = start_pos[0] - r*np.cos(angle + np.pi/2)
            cy = start_pos[1] - r*np.sin(angle + np.pi/2)
            
            # Generate more points to visualize the arc smoothly
            arc_points = 50
            ang_range = np.linspace(angle, angle + dphi, arc_points)
            arc_x = [cx + r*np.cos(ang + np.pi/2) for ang in ang_range]
            arc_y = [cy + r*np.sin(ang + np.pi/2) for ang in ang_range]
            
            # Plot the arc
            plt.plot(arc_x, arc_y, arc_marker, color=arc_color, linewidth=2.5)
            
            # Update position for the line segment
            pos = (arc_x[-1], arc_y[-1])
        else:
            pos = (start_pos[0], start_pos[1])
        
        # Process line segment (if it exists)
        if line_times:
            seglen = l[i]
            phi_line = phi[i+1]
            
            # Calculate line endpoint
            dx = np.cos(phi_line) * seglen
            dy = np.sin(phi_line) * seglen
            line_x = [pos[0], pos[0] + dx]
            line_y = [pos[1], pos[1] + dy]
            
            # Plot the line
            plt.plot(line_x, line_y, line_marker, color=line_color, linewidth=2.5)
    
    # Plot waypoints
    for i, wp_idx in enumerate(waypoints):
        # Convert from pixels to world coordinates using shared coordinate transformation
        pos_pixels = reeb_graph.nodes[wp_idx].configuration
        pos = convert_world_pixel_to_world_meter(pos_pixels)
        plt.plot(pos[0], pos[1], 'ko', markersize=8)
        plt.text(pos[0], pos[1], f'W{wp_idx}', fontsize=12)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory with Separate Arc and Line Segments')
    plt.grid(True)
    plt.legend(handles=[arc_line, line_line])
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Time-based visualization of arcs and lines
    plt.figure(figsize=(12, 6))
    
    total_time = 0.0
    time_points, x_points, y_points = [], [], []
    arc_end_times, line_end_times = [], []
    
    # Process each segment
    for i in range(N):
        # Get the segment info
        seg_times = time_segments[i]
        arc_times = seg_times.get('arc', [])
        line_times = seg_times.get('line', [])
        
        # Add arc end times
        if arc_times:
            total_time += sum(arc_times)
            arc_end_times.append(total_time)
        
        # Add line end times
        if line_times:
            total_time += sum(line_times)
            line_end_times.append(total_time)
    
    # Define a function to discretize the full trajectory for this visualization
    def discretize_trajectory(waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph, dt):
        # Similar to discretize_segment but for the entire trajectory
        all_times, all_xs, all_ys = [], [], []
        total_time = 0.0
        
        # Process each segment
        for i in range(len(waypoints) - 1):
            # Discretize this segment
            x_seg, y_seg, t_seg = discretize_segment(
                i, waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph
            )
            
            # Adjust time values
            t_seg = [t + total_time for t in t_seg]
            
            # Add to the collections
            all_times.extend(t_seg)
            all_xs.extend(x_seg)
            all_ys.extend(y_seg)
            
            # Update total time
            if t_seg:
                total_time = t_seg[-1]
                
        # Create uniform time samples
        t_uniform = np.arange(0, total_time, dt)
        
        # Interpolate positions at uniform times
        from scipy.interpolate import interp1d
        x_interp = interp1d(all_times, all_xs, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_interp = interp1d(all_times, all_ys, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        x_uniform = x_interp(t_uniform)
        y_uniform = y_interp(t_uniform)
        
        return t_uniform, x_uniform, y_uniform
    
    # Get time-position data
    times, xs, ys = discretize_trajectory(
        waypoints=waypoints,
        phi=phi,
        r0=r0,
        l=l,
        phi_new=phi_new,
        time_segments=time_segments,
        Flagb=Flagb,
        reeb_graph=reeb_graph,
        dt=0.01  # Smaller dt for smoother plots
    )
    
    # Plot position vs time
    plt.plot(times, xs, 'b-', label='X position')
    plt.plot(times, ys, 'g-', label='Y position')
    
    # Mark segment transitions
    for t in arc_end_times:
        plt.axvline(x=t, color='blue', linestyle='--', alpha=0.7, label='Arc End' if t == arc_end_times[0] else "")
    
    for t in line_end_times:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.7, label='Line End' if t == line_end_times[0] else "")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time with Arc and Line Segments Highlighted')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_discretization_with_spline(waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph, dt, save_dir):
    """
    Compare the original discrete trajectory (segment endpoints) with cubic spline interpolation
    to visualize the time and position errors. Also saves the trajectory to JSON files.
    
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
        save_dir: Directory to save the trajectory JSON files
    """
    import os
    # Get original discrete trajectory (just the endpoints from each segment)
    N = len(waypoints) - 1
    
    # Collect all discretized segments
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
    
    # Instead of one global spline, we'll create splines for each arc+line pair
    # Define uniform time step across the whole trajectory
    t_uniform = np.arange(0, total_time, dt)
    x_uniform = np.zeros_like(t_uniform)
    y_uniform = np.zeros_like(t_uniform)
    
    # Process each segment independently with its own spline
    segment_boundaries = [0]  # Start with the first index
    current_time = 0.0
    segment_times = []
    segment_xs = []
    segment_ys = []
    
    # Collect boundary indices for each segment (arc+line pair)
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
            segment_boundaries.append(len(all_times))  # Store ending index
    
    # Now interpolate each segment separately and combine
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
    
    # Calculate orientations, velocities and angular velocities directly from cubic splines
    thetas = []
    velocities = []
    angular_velocities = []
    
    # Process each time point individually to calculate derivatives from splines
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
            dx_dt = cs_x_seg(t, 1)  # First derivative of x with respect to time
            dy_dt = cs_y_seg(t, 1)  # First derivative of y with respect to time
            
            # Calculate orientation (direction of motion)
            theta = np.arctan2(dy_dt, dx_dt)
            thetas.append(float(theta))
            
            # Calculate linear velocity (speed)
            velocity = np.sqrt(dx_dt**2 + dy_dt**2)
            velocities.append(float(velocity))
            
            # Second derivatives for angular velocity calculation
            if i < len(t_uniform) - 1:
                # Get the next time point
                t_next = t_uniform[i+1]
                
                # Calculate orientation at next time point if in the same segment
                if segment_times[segment_idx][0] <= t_next <= segment_times[segment_idx][-1]:
                    dx_dt_next = cs_x_seg(t_next, 1)
                    dy_dt_next = cs_y_seg(t_next, 1)
                    theta_next = np.arctan2(dy_dt_next, dx_dt_next)
                    
                    # Calculate change in orientation
                    dtheta = theta_next - theta
                    # Normalize to [-pi, pi]
                    while dtheta > np.pi:
                        dtheta -= 2 * np.pi
                    while dtheta < -np.pi:
                        dtheta += 2 * np.pi
                    
                    # Calculate angular velocity
                    angular_velocity = dtheta / dt
                else:
                    # We're at a segment boundary, use second derivatives instead
                    d2x_dt2 = cs_x_seg(t, 2)  # Second derivative of x
                    d2y_dt2 = cs_y_seg(t, 2)  # Second derivative of y
                    
                    # Formula for angular velocity from derivatives:
                    # ω = (dx/dt * d²y/dt² - dy/dt * d²x/dt²) / ((dx/dt)² + (dy/dt)²)
                    denominator = dx_dt**2 + dy_dt**2
                    if denominator > 1e-10:  # Avoid division by near-zero
                        angular_velocity = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / denominator
                    else:
                        angular_velocity = 0
            else:
                # For the last point, use second derivatives directly
                d2x_dt2 = cs_x_seg(t, 2)
                d2y_dt2 = cs_y_seg(t, 2)
                denominator = dx_dt**2 + dy_dt**2
                if denominator > 1e-10:
                    angular_velocity = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / denominator
                else:
                    angular_velocity = 0
                    
            angular_velocities.append(float(angular_velocity))
        else:
            # This time point doesn't fall within any segment (should not happen)
            # Use default values as fallback
            thetas.append(0.0)
            velocities.append(0.0)
            angular_velocities.append(0.0)
    
    # For the last point, use the same velocity as the second-to-last point
    if len(velocities) > 0:
        velocities.append(0)
        angular_velocities.append(0)  # Angular velocity is likely zero at the end
    else:
        velocities.append(0)  # Default velocity if only one point
        angular_velocities.append(0)
        
    # Create trajectory points as [x, y, theta, v, w]
    trajectory_points = []
    for i in range(len(t_uniform)):
        point = [float(x_uniform[i]), float(y_uniform[i]), float(thetas[i]), 
                float(velocities[i]), float(angular_velocities[i])]
        trajectory_points.append(point)
    
    # Save the trajectory to JSON files
    import json
    import os
    
    # Create directory if it doesn't exist
    # save_dir = '/home/zhihui/data/'
    # os.makedirs(save_dir, exist_ok=True)
    
    # Identify relay points (based on Flagb values)
    relay_indices = [i for i, flag in enumerate(Flagb) if flag != 0 or i == 0]
    relay_indices.append(N)  # Add the last waypoint
    relay_indices = sorted(list(set(relay_indices)))  # Remove duplicates and sort
    
    # Save trajectories between relay points
    saved_files = []
    
    for i in range(len(relay_indices) - 1):
        start_idx = relay_indices[i]
        end_idx = relay_indices[i+1]
        
        # Find corresponding segment in the trajectory
        # We'll use the waypoint positions to find the nearest points in the trajectory
        # Convert from pixels to world coordinates using shared coordinate transformation
        start_pos_pixels = reeb_graph.nodes[waypoints[start_idx]].configuration
        start_pos = convert_world_pixel_to_world_meter(start_pos_pixels)
        end_pos_pixels = reeb_graph.nodes[waypoints[end_idx]].configuration
        end_pos = convert_world_pixel_to_world_meter(end_pos_pixels)
        
        # Find closest trajectory points to these waypoints
        start_dists = [(x-start_pos[0])**2 + (y-start_pos[1])**2 for x, y in zip(x_uniform, y_uniform)]
        end_dists = [(x-end_pos[0])**2 + (y-end_pos[1])**2 for x, y in zip(x_uniform, y_uniform)]
        
        traj_start_idx = np.argmin(start_dists)
        traj_end_idx = np.argmin(end_dists)
        
        # Ensure start < end
        traj_start_idx = min(traj_start_idx, traj_end_idx)
        traj_end_idx = max(traj_start_idx, traj_end_idx)
        
        # Extract segment
        segment = trajectory_points[traj_start_idx:traj_end_idx+1]
        
        if segment:
            # Save to JSON file
            file_path = os.path.join(save_dir, f'tb{i}_Trajectory.json')
            with open(file_path, 'w') as f:
                json.dump({"Trajectory": segment}, f)
            saved_files.append(file_path)
            print(f"Saved trajectory segment {i} to {file_path}")
    
    # Save full trajectory
    full_path = os.path.join(save_dir, 'full_trajectory.json')
    with open(full_path, 'w') as f:
        json.dump({"Trajectory": trajectory_points}, f)
    saved_files.append(full_path)
    print(f"Saved full trajectory to {full_path}")
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # X position vs time plot
    plt.subplot(211)
    plt.plot(all_times, all_xs, 'ro-', label='Original Discrete Points (X)', markersize=6)
    plt.plot(t_uniform, x_uniform, 'bx', label='Cubic Spline Interpolation Points (X)', markersize=4)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('Comparison of Original Discrete Trajectory and Cubic Spline Interpolation - X Position')
    plt.grid(True)
    plt.legend()
    
    # Y position vs time plot
    plt.subplot(212)
    plt.plot(all_times, all_ys, 'ro-', label='Original Discrete Points (Y)', markersize=6)
    plt.plot(t_uniform, y_uniform, 'bx', label='Cubic Spline Interpolation Points (Y)', markersize=4)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Comparison of Original Discrete Trajectory and Cubic Spline Interpolation - Y Position')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2D trajectory
    plt.figure(figsize=(12, 8))
    plt.plot(all_xs, all_ys, 'ro-', label='Original Discrete Points', markersize=6)
    plt.plot(x_uniform, y_uniform, 'bx', label='Cubic Spline Interpolation Points', markersize=4)
    
    # Plot waypoints
    for i, wp_idx in enumerate(waypoints):
        # Convert from pixels to world coordinates using shared coordinate transformation
        pos_pixels = reeb_graph.nodes[wp_idx].configuration
        pos = convert_world_pixel_to_world_meter(pos_pixels)
        plt.plot(pos[0], pos[1], 'ko', markersize=8)
        plt.text(pos[0], pos[1], f'W{wp_idx}', fontsize=12)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Comparison of Original Discrete Trajectory and Cubic Spline Interpolation')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()
    
    # Calculate and display errors
    print("Interpolation Statistics:")
    
    # Calculate position errors by comparing the original points with the spline
    # For each original point, find the closest spline point
    position_errors = []
    for i, (orig_x, orig_y) in enumerate(zip(all_xs, all_ys)):
        # Find the closest point in the interpolated trajectory
        distances = [(orig_x - x)**2 + (orig_y - y)**2 for x, y in zip(x_uniform, y_uniform)]
        min_idx = np.argmin(distances)
        # Calculate error
        error = np.sqrt((orig_x - x_uniform[min_idx])**2 + (orig_y - y_uniform[min_idx])**2)
        position_errors.append(error)
    
    position_errors = np.array(position_errors)
    
    print(f"Max position error: {np.max(position_errors):.6f} m")
    print(f"Mean position error: {np.mean(position_errors):.6f} m")
    print(f"RMS position error: {np.sqrt(np.mean(position_errors**2)):.6f} m")
    
    return saved_files
