#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    """Launch multiple behavior tree instances for multi-robot coordination."""
    
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    num_robots = LaunchConfiguration('num_robots', default='5')
    
    # Launch description
    ld = LaunchDescription()
    
    # Common launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'))
    
    ld.add_action(DeclareLaunchArgument(
        'num_robots',
        default_value='5',
        description='Number of robots to launch behavior trees for'))
    
    # Launch separate behavior tree instances for each robot
    for i in range(5):  # tb0 to tb4
        robot_namespace = f'turtlebot{i}'
        tree_node_name = f'tree_{i}'
        
        # Create a group for each robot's behavior tree
        robot_group = GroupAction([
            PushRosNamespace(robot_namespace),
            
            # Behavior tree node for this robot
            Node(
                package='behaviour_tree',
                executable='my_behaviour_tree',
                name=tree_node_name,
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'robot_id': i,
                    'robot_namespace': robot_namespace,
                    'tree_name': f'BehaviorTree_{i}'
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
        ])
        
        ld.add_action(robot_group)
    
    return ld
