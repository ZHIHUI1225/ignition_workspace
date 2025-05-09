#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joep Tool

import os
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument, TimerAction

def generate_launch_description():
    ld = LaunchDescription()
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    ld.add_action(
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='If true, use simulated clock')
    )

    world_name='turtlebot3_maze'
    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'empty_world.world'
    )
    model_folder = 'epuck'
    urdf = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        model_folder,
        'model.sdf'
    )
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')

    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            os.path.join(get_package_share_directory('turtlebot3_gazebo'), "models"),
            ":" +  # Linux路径分隔符
            os.path.join("/opt/ros/humble", "share")])
    ld.add_action(ign_resource_path)

    # Launch Gazebo first
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('ros_gz_sim'),
                          'launch', 'gz_sim.launch.py')]),
        launch_arguments=[('ign_args', ' -r -v 3 ' + world)]
    )
    ld.add_action(gz_sim)

    # Define robots and their spawn positions
    robots = [
        {'name': 'epuck1', 'x': '0.0', 'y': '0.0'},
        {'name': 'epuck2', 'x': '1.0', 'y': '0.0'},
        # Add more robots as needed
    ]

    bridge_args = []

    for robot in robots:
        spawn_node = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-entity", robot['name'],
                "-file", urdf,
                "-name", robot['name'],
                "-x", robot['x'],
                "-y", robot['y'],
                "-z", "0.01",
                "-Y", "0.0"
            ],
            namespace=robot['name'],
            output="screen",
        )
        ld.add_action(spawn_node
        )
        bridge_args.append(f"/model/{robot['name']}/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist")

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        output='screen',
        arguments=bridge_args
    )
    ld.add_action(bridge_node)

    world_frame_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'world', 'odom'],
        name='static_tf_world_to_odom',
        output='both'
    )
    # ld.add_action(world_frame_publisher)

    return ld
