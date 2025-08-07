#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch a single behavior tree instance for a specific robot."""
    
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_id = LaunchConfiguration('robot_id', default='0')
    robot_namespace = LaunchConfiguration('robot_namespace', default='robot0')
    
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
        'robot_namespace',
        default_value='robot0',
        description='Robot namespace (e.g., robot0, robot1, etc.)'))
    
    # Behavior tree node
    behavior_tree_node = Node(
        package='behaviour_tree',
        executable='my_behaviour_tree',
        name='tree',  # Keep consistent with snapshot stream namespace
        namespace='robot0',  # Use fixed namespace for simplicity
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_id': '0',
            'robot_namespace': 'robot0',
        }]
    )
    ld.add_action(behavior_tree_node)
    
    return ld
