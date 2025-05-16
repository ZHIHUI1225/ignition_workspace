#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch behavior tree controller and robot controllers."""
    
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Environment variables
    behavior_tree_pkg = get_package_share_directory('behaviour_tree')
    
    # Launch description
    ld = LaunchDescription()
    
    # Common launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'))
    
    # Launch the behavior tree controller
    behavior_tree_node = Node(
        package='behaviour_tree',
        executable='Behaviour_tree.py',
        name='behavior_tree_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    ld.add_action(behavior_tree_node)
    
    # Launch Follow controllers for each robot
    for i in range(5):  # tb0 to tb4
        namespace = f'turtlebot{i}'
        robot_name = f'turtlebot{i}'
        
        # Follow controller for this robot
        follow_controller = Node(
            package='behaviour_tree',
            executable='Follow_controller.py',
            name=f'{robot_name}_follow_controller',
            namespace=namespace,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'namespace': robot_name
            }],
            remappings=[
                # Map the generic topics to robot-specific topics
                ('Ready_flag', f'/{namespace}/Ready_flag'),
                ('Pushing_flag', f'/{namespace}/Pushing_flag'),
                ('cmd_vel', f'/{namespace}/cmd_vel'),
                ('odom', f'/{namespace}/odom')
            ]
        )
        ld.add_action(follow_controller)
        
        # Pickup controller for this robot
        pickup_controller = Node(
            package='behaviour_tree',
            executable='Pickup_controller.py',
            name=f'{robot_name}_pickup_controller',
            namespace=namespace,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'namespace': robot_name
            }],
            remappings=[
                # Map the generic topics to robot-specific topics
                ('Pickup_flag', f'/{namespace}/Pickup_flag'),
                ('PickUpDone', f'/{namespace}/PickUpDone'),
                ('cmd_vel', f'/{namespace}/cmd_vel'),
                ('odom', f'/{namespace}/odom')
            ]
        )
        ld.add_action(pickup_controller)
    
    # Add parcel pose bridge - share the parcel pose with all controllers
    parcel_pose_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='parcel_pose_static_publisher',
        arguments=['--frame-id', 'map', '--child-frame-id', 'parcel']
    )
    ld.add_action(parcel_pose_bridge)
    
    return ld