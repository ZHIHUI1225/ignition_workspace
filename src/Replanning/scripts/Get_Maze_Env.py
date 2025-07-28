
import xml.etree.ElementTree as ET
import numpy as np
from Robot import Robot
from Polygon import Polygon, rectangle
from Environment import Environment

def extract_maze_obstacles_info(file_path):
    """
    Extracts obstacle information from the Gazebo world file.
    It looks for models named 'MAZE_*' and extracts their pose and size.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return []
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return []

    obstacles_info = []
    
    # First pass: get basic model info
    models_dict = {}
    for model in root.findall('.//model'): # Find all model elements
        model_name = model.get('name')
        if model_name and model_name.startswith('MAZE_'):
            pose_element = model.find('pose')
            pose_str = pose_element.text if pose_element is not None else '0 0 0 0 0 0'
            
            size_str = None
            link_element = model.find('link') # Direct child 'link' of 'model'
            if link_element is not None:
                # Try to find size in collision geometry first
                collision_element = link_element.find('collision')
                if collision_element is not None:
                    geometry_element = collision_element.find('geometry')
                    if geometry_element is not None:
                        box_element = geometry_element.find('box')
                        if box_element is not None:
                            size_element = box_element.find('size')
                            if size_element is not None:
                                if size_element.text and size_element.text.strip():
                                    size_str = size_element.text
                
                # If not found in collision, try visual geometry
                if size_str is None:
                    visual_element = link_element.find('visual')
                    if visual_element is not None:
                        geometry_element = visual_element.find('geometry')
                        if geometry_element is not None:
                            box_element = geometry_element.find('box')
                            if box_element is not None:
                                size_element = box_element.find('size')
                                if size_element is not None:
                                    if size_element.text and size_element.text.strip():
                                        size_str = size_element.text
                                    
            models_dict[model_name] = {'pose': pose_str, 'size': size_str}
    
    # Second pass: check state section for updated poses
    for state in root.findall(".//state[@world_name='default']"):
        for model in state.findall('model'):
            model_name = model.get('name')
            if model_name and model_name.startswith('MAZE_') and model_name in models_dict:
                pose_element = model.find('pose')
                if pose_element is not None and pose_element.text:
                    models_dict[model_name]['pose'] = pose_element.text
    
    # Create final obstacles list
    for model_name, model_data in models_dict.items():
        pose_str = model_data['pose']
        size_str = model_data['size']
            
        if size_str and size_str.strip():
            obstacles_info.append({'name': model_name, 'pose': pose_str, 'size': size_str})
            print(f"Found {model_name}: pose={pose_str}, size={size_str}")
        else:
            # Skip MAZE models that don't have geometry defined
            print(f"Warning: Could not find size for MAZE model {model_name} in {file_path}, skipping this model")
                
    if not obstacles_info:
        print(f"Warning: No obstacles starting with 'MAZE_' found in {file_path}.")
    return obstacles_info

def extract_wall_info(file_path):
    """
    Extracts wall information from WALL models in the Gazebo world file.
    This function looks for both single WALL model with links and individual WALL_* models.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return {}
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return {}

    wall_info = {
        'wall_poses': {},
        'wall_sizes': {},
        'wall_links': []
    }

    # Method 1: Look for single WALL model with multiple links (original maze_5_world.world structure)
    wall_model = root.find(".//model[@name='WALL']")
    if wall_model is not None:
        print("Found single WALL model with links")
        
        # Get the main WALL model pose
        wall_model_pose = wall_model.find('pose')
        wall_model_pose_str = wall_model_pose.text if wall_model_pose is not None else '0 0 0 0 0 0'
        print(f"WALL model pose: {wall_model_pose_str}")
        
        # Extract information for each wall link
        for link in wall_model.findall('link'):
            link_name = link.get('name')
            if link_name:
                # Get link pose (relative to model)
                link_pose = link.find('pose')
                link_pose_str = link_pose.text if link_pose is not None else '0 0 0 0 0 0'
                
                # Get size from collision or visual geometry
                size_str = None
                
                # Try collision first
                for collision in link.findall('collision'):
                    geometry = collision.find('geometry')
                    if geometry is not None:
                        box = geometry.find('box')
                        if box is not None:
                            size_elem = box.find('size')
                            if size_elem is not None and size_elem.text:
                                size_str = size_elem.text
                                break
                
                # If not found in collision, try visual
                if size_str is None:
                    for visual in link.findall('visual'):
                        geometry = visual.find('geometry')
                        if geometry is not None:
                            box = geometry.find('box')
                            if box is not None:
                                size_elem = box.find('size')
                                if size_elem is not None and size_elem.text:
                                    size_str = size_elem.text
                                    break
                
                if size_str:
                    wall_info['wall_poses'][link_name] = link_pose_str
                    wall_info['wall_sizes'][link_name] = size_str
                    wall_info['wall_links'].append({
                        'name': link_name,
                        'pose': link_pose_str,
                        'size': size_str
                    })
                    print(f"Found wall {link_name}: pose={link_pose_str}, size={size_str}")
                else:
                    print(f"Warning: Could not find size for wall link {link_name}")
    
    # Method 2: Look for individual WALL_* models (alternative structure)
    wall_models = []
    for model in root.findall('.//model'):
        model_name = model.get('name', '')
        if model_name.startswith('WALL_') or model_name == 'WALL':
            wall_models.append(model)
    
    if wall_models and len(wall_info['wall_links']) == 0:  # Only use this method if Method 1 didn't find anything
        print(f"Found {len(wall_models)} individual WALL_* models")
        
        for wall_model in wall_models:
            model_name = wall_model.get('name')
            
            # Get model pose
            model_pose = wall_model.find('pose')
            model_pose_str = model_pose.text if model_pose is not None else '0 0 0 0 0 0'
            
            # Get size from the first link's geometry
            link = wall_model.find('link')
            if link is not None:
                size_str = None
                
                # Try collision first
                for collision in link.findall('collision'):
                    geometry = collision.find('geometry')
                    if geometry is not None:
                        box = geometry.find('box')
                        if box is not None:
                            size_elem = box.find('size')
                            if size_elem is not None and size_elem.text:
                                size_str = size_elem.text
                                break
                
                # If not found in collision, try visual
                if size_str is None:
                    for visual in link.findall('visual'):
                        geometry = visual.find('geometry')
                        if geometry is not None:
                            box = geometry.find('box')
                            if box is not None:
                                size_elem = box.find('size')
                                if size_elem is not None and size_elem.text:
                                    size_str = size_elem.text
                                    break
                
                if size_str:
                    wall_info['wall_poses'][model_name] = model_pose_str
                    wall_info['wall_sizes'][model_name] = size_str
                    wall_info['wall_links'].append({
                        'name': model_name,
                        'pose': model_pose_str,
                        'size': size_str
                    })
                    print(f"Found wall {model_name}: pose={model_pose_str}, size={size_str}")
                else:
                    print(f"Warning: Could not find size for wall model {model_name}")
    
    if len(wall_info['wall_links']) == 0:
        print("Warning: No wall information found in the world file")
    
    return wall_info

def get_maze_simulation_env(file_path):
    """
    Processes the extracted wall and obstacle data to create an Environment object,
    robot, start, and goal positions for the maze.
    """
    # Extract both obstacle and wall information
    obstacles_data = extract_maze_obstacles_info(file_path)
    wall_data = extract_wall_info(file_path)
    
    # Process MAZE obstacles
    obstacle_rects = []
    all_x_coords_cm = []
    all_y_coords_cm = []

    for obs_data in obstacles_data:
        try:
            pose = np.array(obs_data['pose'].split(), dtype=float)
            size = np.array(obs_data['size'].split(), dtype=float)
        except ValueError as e:
            print(f"Warning: Could not parse pose/size for {obs_data.get('name', 'unknown obstacle')}: {e}. Skipping.")
            continue

        # Gazebo pose: x, y, z, roll, pitch, yaw (meters)
        # Gazebo box size: width (x-dim), depth (y-dim), height (z-dim) (meters)
        center_x_cm = pose[0] * 100  # Convert m to cm
        center_y_cm = pose[1] * 100  # Convert m to cm
        width_cm = size[0] * 100     # Convert m to cm
        depth_cm = size[1] * 100     # Convert m to cm
        
        # Polygon.rectangle(x_min, x_max, y_max, y_min)
        x_min_obs_cm = center_x_cm - width_cm / 2
        x_max_obs_cm = center_x_cm + width_cm / 2
        y_min_obs_cm = center_y_cm - depth_cm / 2
        y_max_obs_cm = center_y_cm + depth_cm / 2
        
        obstacle_rects.append(rectangle(x_min_obs_cm, x_max_obs_cm, y_max_obs_cm, y_min_obs_cm))
        
        all_x_coords_cm.extend([x_min_obs_cm, x_max_obs_cm])
        all_y_coords_cm.extend([y_min_obs_cm, y_max_obs_cm])

    # Process WALL information
    wall_rects = []
    wall_count = 0
    
    for wall_link in wall_data.get('wall_links', []):
        try:
            pose = np.array(wall_link['pose'].split(), dtype=float)
            size = np.array(wall_link['size'].split(), dtype=float)
            
            # Extract pose components: x, y, z, roll, pitch, yaw
            center_x_m = pose[0]
            center_y_m = pose[1] 
            center_z_m = pose[2]
            roll = pose[3]
            pitch = pose[4]
            yaw = pose[5]  # This is the rotation around Z-axis
            
            # Convert to cm
            center_x_cm = center_x_m * 100
            center_y_cm = center_y_m * 100
            
            # Extract size components: width, depth, height (in Gazebo coordinate system)
            width_m = size[0]   # X-dimension
            depth_m = size[1]   # Y-dimension  
            height_m = size[2]  # Z-dimension
            
            # Convert to cm
            width_cm = width_m * 100
            depth_cm = depth_m * 100
            height_cm = height_m * 100
            
            # Account for rotation: when yaw != 0, width and depth may be swapped in final orientation
            # For walls, we need to consider the actual oriented bounding box
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # Calculate the oriented bounding box corners before rotation
            half_width = width_cm / 2
            half_depth = depth_cm / 2
            
            # Local corners (relative to wall center, before rotation)
            local_corners = np.array([
                [-half_width, -half_depth],  # Bottom-left
                [half_width, -half_depth],   # Bottom-right
                [half_width, half_depth],    # Top-right
                [-half_width, half_depth]    # Top-left
            ])
            
            # Rotate corners by yaw angle
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            
            rotated_corners = local_corners @ rotation_matrix.T
            
            # Translate to world coordinates
            world_corners = rotated_corners + np.array([center_x_cm, center_y_cm])
            
            # Find axis-aligned bounding box of the rotated wall
            x_coords = world_corners[:, 0]
            y_coords = world_corners[:, 1]
            
            x_min_wall_cm = np.min(x_coords)
            x_max_wall_cm = np.max(x_coords)
            y_min_wall_cm = np.min(y_coords)
            y_max_wall_cm = np.max(y_coords)
            
            wall_rects.append(rectangle(x_min_wall_cm, x_max_wall_cm, y_max_wall_cm, y_min_wall_cm))
            wall_count += 1
            
            print(f"Processed wall {wall_link['name']}: center=({center_x_cm:.1f}, {center_y_cm:.1f}), " +
                  f"size=({width_cm:.1f}x{depth_cm:.1f}), yaw={yaw:.3f}rad ({np.degrees(yaw):.1f}°)")
            print(f"  Oriented bounding box: x=[{x_min_wall_cm:.1f}, {x_max_wall_cm:.1f}], " +
                  f"y=[{y_min_wall_cm:.1f}, {y_max_wall_cm:.1f}]")
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse wall {wall_link['name']}: {e}. Skipping.")
            continue
    
    # Combine walls and obstacles
    all_polygons = wall_rects + obstacle_rects
    
    # Create environment and add metadata
    barriers_environment = Environment(all_polygons)
    
    # Add metadata to identify which polygons should not be expanded
    # The first 'wall_count' polygons are walls and should not be expanded
    barriers_environment.wall_indices = list(range(wall_count))  # First wall_count polygons are walls
    barriers_environment.no_expand_indices = list(range(wall_count))   # Walls should not be expanded
    
    print(f"Environment created with {wall_count} walls and {len(obstacle_rects)} obstacles")
    print(f"Wall indices (not expanded): {barriers_environment.wall_indices}")
    
    # Robot geometry (from Get_Env.py, assumed to be in cm)
    robot_geometry = Polygon([(-33, 63), (-33, -63), (33, -63), (33, -42), (-11, -42),
                              (-11, 42), (33, 42), (33, 63), (-33, 63)])
    robot = Robot(robot_geometry)

    return barriers_environment, robot

# if __name__ == '__main__':
#     # Example usage for testing Get_Maze_Env.py directly
#     # Create a dummy maze_world.world or use the actual one if available
#     # For example, if maze_world.world is in the same directory:
#     # test_file_path = 'maze_world.world'
#     # If it's in the parent directory (typical for running from a subfolder):
#     # import os
#     # test_file_path = os.path.join(os.path.dirname(__file__), '..', 'maze_world.world')
#     case="simple_maze"
#     # This path assumes the script is in Relay_task-1 and maze_world.world is also there.
#     test_file_path = case+'_world.world' 

#     print(f"Testing with world file: {test_file_path}")
    
#     # Check if file exists before parsing
#     import os
#     if not os.path.exists(test_file_path):
#         print(f"Test world file {test_file_path} not found. Skipping direct test.")
#     else:
#         environment, robot, start, goal = get_maze_simulation_env(test_file_path)
#         print("\n--- Test Results ---")
#         print(f"Environment: {len(environment.polygons)} polygons")
#         print(f"Robot Geometry: {robot.geometry.vertices}")
#         print(f"Start Pose (cm): {start}")
#         print(f"Goal Pose (cm): {goal}")
#         print("--------------------\\n")

        # To visualize, you might need matplotlib and environment drawing methods
        # environment.draw('blue')
        # import matplotlib.pyplot as plt
        # plt.plot(start[0], start[1], 'go', label='Start')
        # plt.plot(goal[0], goal[1], 'ro', label='Goal')
        # robot.geometry.draw_patch('green',np.array([start[0],start[1],start[2]]))\n        # plt.legend()
        # plt.axis('equal')
        # plt.show()
