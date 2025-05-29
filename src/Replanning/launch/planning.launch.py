from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
import os

def generate_launch_description():
    """Generate launch description for multiple planning nodes - one per robot."""
    
    # Launch arguments
    case_arg = DeclareLaunchArgument(
        'case',
        default_value='simple_maze',
        description='Case name for planning (e.g., "simulation", "simple_maze")'
    )
    
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots in the system'
    )
    
    n_arg = DeclareLaunchArgument(
        'n',
        default_value='5',
        description='N parameter for planning'
    )
    
    # Add target_time argument
    target_time_arg = DeclareLaunchArgument(
        'target_time',
        default_value='30.0',
        description='Target completion time for robot trajectories (seconds)'
    )
    
    # Create a list to hold all our launch actions
    launch_actions = [case_arg, num_robots_arg, n_arg, target_time_arg]
    
    # Create a planning node for each robot
    num_robots = int(os.environ.get('NUM_ROBOTS', '3'))  # Default to 3 if not set
    
    planning_nodes = []
    for i in range(num_robots):
        robot_namespace = f'tb{i}'
        
        # Planning node for this robot
        planning_node = Node(
            package='Replanning',
            executable='planning',
            name=f'planning_node_{robot_namespace}',
            namespace=robot_namespace,
            output='screen',
            parameters=[{
                'case': LaunchConfiguration('case'),
                'n': LaunchConfiguration('n'),
                'namespace': robot_namespace,
                'robot_id': i,
                'target_time': LaunchConfiguration('target_time')
            }]
        )
        planning_nodes.append(planning_node)
    
    # Add all planning nodes to the launch actions
    launch_actions.extend(planning_nodes)
    
    return LaunchDescription(launch_actions)
