#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    """Launch three e-puck robots with different movement behaviors."""
    
    # Package directory
    pkg_dir = get_package_share_directory('epuck_driver_ros2')
    
    # Robot 0 - Forward movement
    robot0_driver = Node(
        package='epuck_driver_ros2',
        executable='epuck_driver_ros2.py',
        name='epuck_driver_robot0',
        namespace='robot0',
        output='screen',
        parameters=[os.path.join(pkg_dir, 'config', 'robot0_ns.yaml')]
    )
    
    # Robot 1 - Backward movement  
    robot1_driver = Node(
        package='epuck_driver_ros2',
        executable='epuck_driver_ros2.py',
        name='epuck_driver_robot1',
        namespace='robot1', 
        output='screen',
        parameters=[os.path.join(pkg_dir, 'config', 'robot1_ns.yaml')]
    )
    
    # Robot 2 - Rotation movement
    robot2_driver = Node(
        package='epuck_driver_ros2',
        executable='epuck_driver_ros2.py',
        name='epuck_driver_robot2',
        namespace='robot2',
        output='screen', 
        parameters=[os.path.join(pkg_dir, 'config', 'robot2_ns.yaml')]
    )
    
    # Movement command publishers using Simple CMD Publisher nodes with keyboard control


    return LaunchDescription([
        # Robot drivers
        robot0_driver,
        robot1_driver, 
        robot2_driver,
    
    ])
