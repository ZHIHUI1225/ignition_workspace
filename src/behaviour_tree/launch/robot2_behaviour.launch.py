#!/usr/bin/env python3
"""
Launch file for robot 1 behavior tree.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch behavior tree for robot 1."""
    
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Launch description
    ld = LaunchDescription()
    
    # Common launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'))
    
    # Robot 1 configuration
    robot_id = 2
    robot_namespace = f'robot{robot_id}'
    tree_node_name = f'tree_{robot_id}'
    
    # Modular behavior tree node for robot 1
    behavior_tree_node = Node(
        package='behaviour_tree',
        executable='my_behaviour_tree_modular',
        name=tree_node_name,
        namespace=robot_namespace,
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_id': robot_id,
            'robot_namespace': robot_namespace,
            'tree_name': f'BehaviorTree_{robot_id}',
            'case': 'simple_maze'
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
    
    ld.add_action(behavior_tree_node)
    
    return ld
