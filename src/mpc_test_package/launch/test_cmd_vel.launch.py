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
    rviz_config_file = os.path.join(package_dir, 'rviz', 'cmd_vel_test.rviz')
    
    return LaunchDescription([
        # Robot State Publisher (for transforms)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': '''
                    <?xml version="1.0"?>
                    <robot name="test_robot">
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
                        
                        <!-- Add a forward-pointing arrow to show robot orientation -->
                        <link name="arrow_link">
                            <visual>
                                <origin xyz="0.08 0 0.05" rpy="0 0 0"/>
                                <geometry>
                                    <cylinder length="0.02" radius="0.01"/>
                                </geometry>
                                <material name="red">
                                    <color rgba="1 0 0 1"/>
                                </material>
                            </visual>
                        </link>
                        
                        <joint name="base_to_arrow" type="fixed">
                            <parent link="base_link"/>
                            <child link="arrow_link"/>
                        </joint>
                    </robot>
                ''',
                'use_sim_time': False
            }]
        ),
        
        # Static transform from world to base_link (robot starting position)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_base_link_transform',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
            output='screen'
        ),
        
        # Static transform from world to camera_frame (overhead camera)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_camera_transform',
            arguments=['0', '0', '2.0', '0', '1.5708', '0', 'world', 'camera_frame'],
            output='screen'
        ),
        
        # CMD_VEL Publisher Node - publishes constant velocity
        Node(
            package='mpc_test_package',
            executable='cmd_vel_publisher',
            name='cmd_vel_test_publisher',
            output='screen',
            parameters=[{
                'linear_x': 0.1,      # 0.05 m/s forward
                'linear_y': 0.0,       # No sideways movement
                'angular_z': 0.2,      # No rotation
                'publish_rate': 10.0,  # 10 Hz
                'use_sim_time': False
            }]
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
