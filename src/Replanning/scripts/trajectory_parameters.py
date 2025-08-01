import json
import numpy as np
import os
import sys

# Add config path to sys.path and load configuration
sys.path.append('/root/workspace/config')
from config_loader import config

# Import coordinate transformation utilities
from coordinate_transform import convert_world_pixel_to_world_meter

def save_trajectory_parameters(waypoints, phi, r0, l, phi_new, time_segments, Flagb, 
                             reeb_graph=None, save_file_path=None, case=None, N=None):
    """
    Save trajectory parameters to JSON files for each robot separately based on relay points.
    
    Args:
        waypoints: List of waypoint indices
        phi: Array of angles at each waypoint
        r0: Array of arc radii for each segment
        l: Array of line lengths for each segment
        phi_new: Array of adjusted angles accounting for flag values
        time_segments: List of dictionaries with 'arc' and 'line' time values
        Flagb: Array of flag values for each waypoint (!=0 indicates relay points)
        reeb_graph: Optional reeb graph for saving waypoint positions
        save_file_path: Optional custom file path (will be ignored if saving per robot)
        case: Case name for default file naming
        N: Number of robots
    
    Returns:
        list: List of paths to the saved files for each robot
    """
    # Get waypoint positions from reeb_graph if available
    waypoint_positions = None
    if reeb_graph is not None:
        # Convert world_pixel coordinates to world_meter coordinates directly
        waypoint_positions = []
        
        for i in range(len(waypoints)):
            world_pixel_pos = reeb_graph.nodes[waypoints[i]].configuration
            world_meter_pos = convert_world_pixel_to_world_meter(world_pixel_pos)
            waypoint_positions.append(world_meter_pos.tolist())
    
    # If case is specified, use it for file naming and directory structure
    if save_file_path is None and case is not None:
        # Create directory for case if it doesn't exist
        save_dir = os.path.join(config.data_path, case)
        os.makedirs(save_dir, exist_ok=True)
        
        # For single-robot case, save to a common file
        if N is None or N <= 1:
            save_file_path = os.path.join(save_dir, f'trajectory_parameters_{case}.json')
    
    # Identify relay points (including start)
    relay_indices = [0]  # Start is always a relay point
    for i in range(1, len(waypoints)):
        if i < len(Flagb) and Flagb[i] != 0:
            relay_indices.append(i)
    
    # Add end if not already included
    if len(waypoints) - 1 not in relay_indices:
        relay_indices.append(len(waypoints) - 1)
    
    # Convert to python lists for JSON serialization
    phi_list = phi.tolist() if isinstance(phi, np.ndarray) else phi
    r0_list = r0.tolist() if isinstance(r0, np.ndarray) else r0
    l_list = l.tolist() if isinstance(l, np.ndarray) else l
    phi_new_list = phi_new.tolist() if isinstance(phi_new, np.ndarray) else phi_new
    Flagb_list = Flagb.tolist() if isinstance(Flagb, np.ndarray) else Flagb
    
    # Save complete trajectory parameters to a single file
    if save_file_path is not None:
        # Save all trajectory parameters (not split by robot)
        complete_params = {
            'waypoints': waypoints,
            'phi': phi_list,
            'r0': r0_list,
            'l': l_list,
            'phi_new': phi_new_list,
            'time_segments': time_segments,
            'Flagb': Flagb_list,
            'relay_indices': relay_indices,
            'waypoint_positions': waypoint_positions,
            'case': case,
            'N': N
        }
        
        with open(save_file_path, 'w') as f:
            json.dump(complete_params, f, indent=2)
        
        saved_files = [save_file_path]
    
    # If N is specified and > 1, save separate files for each robot (relay-to-relay segments)
    elif N is not None and N > 1 and case is not None:
        saved_files = []
        
        # Save separate files for each robot
        for i in range(len(relay_indices) - 1):
            start_idx = relay_indices[i]
            end_idx = relay_indices[i+1]
            
            # Extract parameters for this segment
            robot_params = {
                'waypoints': waypoints[start_idx:end_idx+1],
                'phi': phi_list[start_idx:end_idx+1],
                'r0': r0_list[start_idx:end_idx],
                'l': l_list[start_idx:end_idx],
                'phi_new': phi_new_list[start_idx:end_idx],
                'time_segments': time_segments[start_idx:end_idx],
                'Flagb': Flagb_list[start_idx:end_idx+1],
                'waypoint_positions': waypoint_positions[start_idx:end_idx+1] if waypoint_positions else None,
                'case': case,
                'robot_id': i
            }
            
            # Create file path for this robot
            robot_file = os.path.join(save_dir, f'robot_{i}_trajectory_parameters_{case}.json')
            
            # Save robot-specific parameters
            with open(robot_file, 'w') as f:
                json.dump(robot_params, f, indent=2)
            
            saved_files.append(robot_file)
        
        # Also save a complete parameters file
        complete_file = os.path.join(save_dir, f'complete_trajectory_parameters_{case}.json')
        complete_params = {
            'waypoints': waypoints,
            'phi': phi_list,
            'r0': r0_list,
            'l': l_list,
            'phi_new': phi_new_list,
            'time_segments': time_segments,
            'Flagb': Flagb_list,
            'relay_indices': relay_indices,
            'waypoint_positions': waypoint_positions,
            'case': case,
            'N': N
        }
        
        with open(complete_file, 'w') as f:
            json.dump(complete_params, f, indent=2)
        
        saved_files.append(complete_file)
    
    else:
        # Default case: save to a file in current directory
        save_file_path = 'trajectory_parameters.json'
        
        # Save all trajectory parameters
        complete_params = {
            'waypoints': waypoints,
            'phi': phi_list,
            'r0': r0_list,
            'l': l_list,
            'phi_new': phi_new_list,
            'time_segments': time_segments,
            'Flagb': Flagb_list,
            'relay_indices': relay_indices,
            'waypoint_positions': waypoint_positions
        }
        
        with open(save_file_path, 'w') as f:
            json.dump(complete_params, f, indent=2)
        
        saved_files = [save_file_path]
    
    # Convert lists back to numpy arrays for compatibility
    if isinstance(phi, np.ndarray):
        phi = np.array(phi_list)
    if isinstance(r0, np.ndarray):
        r0 = np.array(r0_list)
    if isinstance(l, np.ndarray):
        l = np.array(l_list)
    if isinstance(phi_new, np.ndarray):
        phi_new = np.array(phi_new_list)
    if isinstance(Flagb, np.ndarray):
        Flagb = np.array(Flagb_list)
    
    return saved_files

def load_trajectory_parameters(case, robot_id=None):
    """
    Load trajectory parameters from saved JSON files.
    
    Args:
        case: Case name used in file naming
        robot_id: Optional robot ID to load specific robot parameters
                 If None, all robot parameters are loaded
    
    Returns:
        dict or list: Parameters dictionary for single robot or list of dictionaries for all robots
    """
    # Create directory for case
    save_dir = os.path.join(config.data_path, case)
    
    if robot_id is not None:
        # Load parameters for specific robot
        robot_file = os.path.join(save_dir, f'robot_{robot_id}_trajectory_parameters_{case}.json')
        
        try:
            with open(robot_file, 'r') as f:
                params = json.load(f)
            
            # Convert lists back to numpy arrays for compatibility
            if 'phi' in params:
                params['phi'] = np.array(params['phi'])
            if 'r0' in params:
                params['r0'] = np.array(params['r0'])
            if 'l' in params:
                params['l'] = np.array(params['l'])
            if 'phi_new' in params:
                params['phi_new'] = np.array(params['phi_new'])
            if 'Flagb' in params:
                params['Flagb'] = np.array(params['Flagb'])
            
            return params
        
        except FileNotFoundError:
            print(f"Warning: Parameters file not found for robot {robot_id} in case {case}")
            return None
    
    else:
        # Try to load complete parameters file first
        complete_file = os.path.join(save_dir, f'complete_trajectory_parameters_{case}.json')
        
        try:
            with open(complete_file, 'r') as f:
                complete_params = json.load(f)
            
            # Convert lists back to numpy arrays
            if 'phi' in complete_params:
                complete_params['phi'] = np.array(complete_params['phi'])
            if 'r0' in complete_params:
                complete_params['r0'] = np.array(complete_params['r0'])
            if 'l' in complete_params:
                complete_params['l'] = np.array(complete_params['l'])
            if 'phi_new' in complete_params:
                complete_params['phi_new'] = np.array(complete_params['phi_new'])
            if 'Flagb' in complete_params:
                complete_params['Flagb'] = np.array(complete_params['Flagb'])
            
            # Check if we should load individual robot files
            if 'N' in complete_params and complete_params['N'] > 1:
                # Load parameters for all robots
                robot_params = []
                
                for i in range(complete_params['N']):
                    robot_file = os.path.join(save_dir, f'robot_{i}_trajectory_parameters_{case}.json')
                    
                    try:
                        with open(robot_file, 'r') as f:
                            params = json.load(f)
                        
                        # Convert lists to numpy arrays
                        if 'phi' in params:
                            params['phi'] = np.array(params['phi'])
                        if 'r0' in params:
                            params['r0'] = np.array(params['r0'])
                        if 'l' in params:
                            params['l'] = np.array(params['l'])
                        if 'phi_new' in params:
                            params['phi_new'] = np.array(params['phi_new'])
                        if 'Flagb' in params:
                            params['Flagb'] = np.array(params['Flagb'])
                        
                        robot_params.append(params)
                    
                    except FileNotFoundError:
                        print(f"Warning: Parameters file not found for robot {i}")
                        robot_params.append(None)
                
                return robot_params
            
            else:
                # Just return the complete parameters
                return complete_params
        
        except FileNotFoundError:
            # Try the old-style single file
            old_file = os.path.join(save_dir, f'trajectory_parameters_{case}.json')
            
            try:
                with open(old_file, 'r') as f:
                    params = json.load(f)
                
                # Convert lists back to numpy arrays for compatibility
                if 'phi' in params:
                    params['phi'] = np.array(params['phi'])
                if 'r0' in params:
                    params['r0'] = np.array(params['r0'])
                if 'l' in params:
                    params['l'] = np.array(params['l'])
                if 'phi_new' in params:
                    params['phi_new'] = np.array(params['phi_new'])
                if 'Flagb' in params:
                    params['Flagb'] = np.array(params['Flagb'])
                
                return params
            
            except FileNotFoundError:
                print(f"Warning: No parameters files found for case {case}")
                return None

def load_complete_trajectory_parameters(case):
    """
    Load the complete trajectory parameters for all robots.
    
    Args:
        case: Case name used in file naming
    
    Returns:
        dict: Complete parameters dictionary
    """
    # Create directory for case
    save_dir = os.path.join(config.data_path, case)
    complete_file = os.path.join(save_dir, f'complete_trajectory_parameters_{case}.json')
    
    try:
        with open(complete_file, 'r') as f:
            params = json.load(f)
        
        # Convert lists back to numpy arrays for compatibility
        if 'phi' in params:
            params['phi'] = np.array(params['phi'])
        if 'r0' in params:
            params['r0'] = np.array(params['r0'])
        if 'l' in params:
            params['l'] = np.array(params['l'])
        if 'phi_new' in params:
            params['phi_new'] = np.array(params['phi_new'])
        if 'Flagb' in params:
            params['Flagb'] = np.array(params['Flagb'])
        
        return params
    
    except FileNotFoundError:
        print(f"Warning: Complete parameters file not found for case {case}")
        return None

def plot_from_saved_trajectory(file_path, reeb_graph=None):
    """
    Create visualization from saved trajectory parameters.
    
    Args:
        file_path: Path to the saved trajectory parameters JSON file
        reeb_graph: Optional reeb graph object for waypoint positions
    
    Returns:
        dict: Trajectory parameters loaded from file
    """
    # Load parameters from file
    with open(file_path, 'r') as f:
        params = json.load(f)
    
    # Use reeb_graph from parameters if available, otherwise use provided one
    if reeb_graph is None and params['waypoint_positions'] is not None:
        print("Using waypoint positions from saved file")
        # Create a simple mock reeb_graph structure if needed
        class MockNode:
            def __init__(self, config):
                self.configuration = np.array(config)  # Use meter values as is - no conversion needed
        
        class MockReebGraph:
            def __init__(self, positions):
                self.nodes = {i: MockNode(pos) for i, pos in enumerate(positions)}
        
        reeb_graph = MockReebGraph(params['waypoint_positions'])
    
    from trajectory_visualization import plot_trajectory_with_time
    
    # Extract parameters from file
    waypoints = params['waypoints']
    phi = np.array(params['phi'])
    r0 = np.array(params['r0'])
    l = np.array(params['l'])
    phi_new = np.array(params['phi_new'])
    time_segments = params['time_segments']
    Flagb = np.array(params['Flagb'])
    
    case = params.get('case', 'default')
    figure_file = f'trajectory_visualization_{case}.png'
    
    # Create the visualization
    plot_trajectory_with_time(
        waypoints=waypoints,
        phi=phi,
        r0=r0,
        l=l,
        phi_new=phi_new,
        time_segments=time_segments,
        figure_file=figure_file,
        reeb_graph=reeb_graph,
        Flagb=Flagb,
        case=case,
        N=params.get('N', 1)
    )
    
    return params

def generate_spline_from_saved_trajectory(file_path, reeb_graph=None, dt=0.1, save_dir=None):
    """
    Generate spline trajectory from saved trajectory parameters.
    
    Args:
        file_path: Path to the saved trajectory parameters JSON file
        reeb_graph: Optional reeb graph object for waypoint positions
        dt: Time step for discretization
        save_dir: Directory to save the spline trajectory JSON file
    
    Returns:
        str: Path to the saved spline trajectory file
    """
    # Load parameters from file
    with open(file_path, 'r') as f:
        params = json.load(f)
    
    # Use reeb_graph from parameters if available, otherwise use provided one
    if reeb_graph is None and params['waypoint_positions'] is not None:
        print("Using waypoint positions from saved file")
        # Create a simple mock reeb_graph structure if needed
        class MockNode:
            def __init__(self, config):
                self.configuration = np.array(config)  # Use meter values as is - no conversion needed
        
        class MockReebGraph:
            def __init__(self, positions):
                self.nodes = {i: MockNode(pos) for i, pos in enumerate(positions)}
        
        reeb_graph = MockReebGraph(params['waypoint_positions'])
    
    from discretization import compare_discretization_with_spline
    
    # Extract parameters from file
    waypoints = params['waypoints']
    phi = np.array(params['phi'])
    r0 = np.array(params['r0'])
    l = np.array(params['l'])
    phi_new = np.array(params['phi_new'])
    time_segments = params['time_segments']
    Flagb = np.array(params['Flagb'])
    
    case = params.get('case', 'default')
    robot_id = params.get('robot_id', None)
    
    # Set default save directory if not provided
    if save_dir is None:
        if case:
            save_dir = os.path.join(config.data_path, case) + '/'
        else:
            save_dir = './'
    
    # Generate the spline trajectory
    saved_files = compare_discretization_with_spline(
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
        robot_id=robot_id
    )
    
    return saved_files
