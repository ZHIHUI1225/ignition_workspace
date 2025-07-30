#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch e-puck driver with configurable parameters."""
    
    # Package directory
    pkg_dir = get_package_share_directory('epuck_driver_ros2')
    
    # Parameter file argument only (load all parameters from YAML)
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'epuck_driver.yaml'),
        description='Path to YAML parameter file for e-puck driver')

    # e-puck driver node
    epuck_driver_node = Node(
        package='epuck_driver_ros2',
        executable='epuck_driver_ros2.py',
        name='epuck_driver_ros2',  # Match the name in YAML file
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
    )
    
    return LaunchDescription([
        params_file_arg,
        epuck_driver_node
    ])
