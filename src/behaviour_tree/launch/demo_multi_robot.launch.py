#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    """Demo launch file for testing multi-robot behavior trees with 2 robots."""
    
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Launch description
    ld = LaunchDescription()
    
    # Launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'))
    
    # Launch behavior trees for 2 robots only (for demo)
    robot_configs = [
        {'id': 0, 'namespace': 'turtlebot0'},
        {'id': 1, 'namespace': 'turtlebot1'}
    ]
    
    for robot in robot_configs:
        robot_group = GroupAction([
            PushRosNamespace(robot['namespace']),
            
            # Behavior tree node for this robot
            Node(
                package='behaviour_tree',
                executable='my_behaviour_tree',
                name='tree',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'robot_id': robot['id'],
                    'robot_namespace': robot['namespace'],
                }]
            )
        ])
        
        ld.add_action(robot_group)
    
    return ld
