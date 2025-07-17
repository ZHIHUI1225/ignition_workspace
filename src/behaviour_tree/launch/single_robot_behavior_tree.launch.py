#!/usr/bin/env python3
"""
Launch file for a configurable single robot behavior tree.

This launch file allows running a behavior tree for any robot by specifying:
- robot_id: The ID of the robot (0-4) - determines the robot namespace (turtlebot<ID>)
- case: The simulation case (e.g., simple_maze, maze_5)

Example usage:
    ros2 launch behaviour_tree single_robot_behavior_tree.launch.py robot_id:=1 case:=maze_5
    
The robot namespace is automatically set to 'turtlebot<ID>' based on the robot_id.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch a single behavior tree instance for a specific robot.
    
    This function creates a launch description for running a modular behavior tree
    for a specific robot with configurable ID, namespace, and simulation case.
    """
    
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_id = LaunchConfiguration('robot_id', default='0')
    case = LaunchConfiguration('case', default='simple_maze')
    
    # Launch description
    ld = LaunchDescription()
    
    # Launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'))
    
    ld.add_action(DeclareLaunchArgument(
        'robot_id',
        default_value='0',
        description='Robot ID (0-4)'))
        
    ld.add_action(DeclareLaunchArgument(
        'case',
        default_value='simple_maze',
        description='Case name (e.g., simple_maze, maze_5, etc.)'))
    
    # Define the launch configuration callback
    def launch_setup(context):
        # Evaluate LaunchConfiguration objects to actual values
        robot_id_value = context.perform_substitution(robot_id)
        robot_id_int = int(robot_id_value)
        
        # Generate derived values for this robot
        robot_namespace = f'turtlebot{robot_id_value}'
        tree_node_name = f'tree_{robot_id_value}'
        
        # Behavior tree node
        behavior_tree_node = Node(
            package='behaviour_tree',
            executable='my_behaviour_tree_modular',
            name=tree_node_name,
            namespace=robot_namespace,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_id': robot_id_int,  # Pass as integer
                'robot_namespace': robot_namespace,
                'tree_name': f'BehaviorTree_{robot_id_value}',
                'case': case
            }],
            remappings=[
                # Robot-specific topic remappings
                ('Ready_flag', f'/{robot_namespace}/Ready_flag'),
                ('Pushing_flag', f'/{robot_namespace}/Pushing_flag'),
                ('Pickup_flag', f'/{robot_namespace}/Pickup_flag'),
                ('PickUpDone', f'/{robot_namespace}/PickUpDone'),
                ('cmd_vel', f'/{robot_namespace}/cmd_vel'),
                ('odom', f'/{robot_namespace}/odom'),
                # Snapshot stream topics with robot-specific namespace
                ('tree_log', f'/{robot_namespace}/tree_log'),
                ('tree_snapshot', f'/{robot_namespace}/tree_snapshot'),
                ('tree_updates', f'/{robot_namespace}/tree_updates'),
                ('snapshot_streams', f'/{robot_namespace}/snapshot_streams')
            ]
        )
        
        return [behavior_tree_node]
    
    # Add the launch setup via OpaqueFunction
    ld.add_action(OpaqueFunction(function=launch_setup))
    
    return ld
