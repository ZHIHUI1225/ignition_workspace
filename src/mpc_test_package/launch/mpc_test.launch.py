#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mpc_test_package',
            executable='mpc_test_node',
            name='mpc_test_node',
            output='screen',
            parameters=[{
                'use_sim_time': False
            }]
        )
    ])
