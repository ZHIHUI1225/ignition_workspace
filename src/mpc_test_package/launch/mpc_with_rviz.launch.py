#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directory
    package_dir = get_package_share_directory('mpc_test_package')
    
    # RViz config file path
    rviz_config_file = os.path.join(package_dir, 'rviz', 'mpc_visualization.rviz')
    
    return LaunchDescription([
        # MPC Test Node
        Node(
            package='mpc_test_package',
            executable='mpc_test_node',
            name='mpc_test_node',
            output='screen',
            parameters=[{
                'use_sim_time': False
            }]
        ),
        
        # Robot State Publisher (for transforms)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': '''
                    <?xml version="1.0"?>
                    <robot name="simple_robot">
                        <link name="base_link">
                            <visual>
                                <geometry>
                                    <cylinder length="0.1" radius="0.05"/>
                                </geometry>
                                <material name="blue">
                                    <color rgba="0 0 1 1"/>
                                </material>
                            </visual>
                        </link>
                    </robot>
                ''',
                'use_sim_time': False
            }]
        ),
        
        # Static transform from world to camera_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_camera_transform',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'camera_frame'],
            output='screen'
        ),
        
        # RViz Node
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else [],
            parameters=[{
                'use_sim_time': False
            }]
        )
    ])
