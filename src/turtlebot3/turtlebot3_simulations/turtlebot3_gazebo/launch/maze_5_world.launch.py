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
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    TURTLEBOT3_MODEL = "burger"
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Declare use_sim_time argument
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='If true, use simulated clock'
    )
    ld.add_action(declare_use_sim_time)

    turtlebot3_gazebo_pkg = get_package_share_directory("turtlebot3_gazebo")

    world = os.path.join(
        turtlebot3_gazebo_pkg,
        'worlds',
        'maze_5_world.world'
    )

    urdf_file_name = "turtlebot3_" + TURTLEBOT3_MODEL + ".urdf"
    print("urdf_file_name : {}".format(urdf_file_name))
    model_folder = 'turtlebot3_burger_fork'
    urdf = os.path.join(
        turtlebot3_gazebo_pkg,
        'models',
        model_folder,
        'model.sdf'
    )

    if not os.path.exists(world):
        raise FileNotFoundError(f"World file not found: {world}")
    if not os.path.exists(urdf):
        raise FileNotFoundError(f"URDF file not found: {urdf}")

    # Set Ignition Gazebo resource path
    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            os.path.join(turtlebot3_gazebo_pkg, "models"),
            ":" + os.path.join("/opt/ros/humble", "share")
        ]
    )
    ld.add_action(ign_resource_path)

    # Disable shared memory transport for Fast RTPS
    disable_shm_transport = SetEnvironmentVariable(
        name='RMW_FASTRTPS_USE_SHM',
        value='0'
    )
    ld.add_action(disable_shm_transport)

    # Camera setup after Gazebo starts (delayed)
    camera_setup = TimerAction(
        period=5.0,  # Wait 5 seconds for Gazebo to start
        actions=[
            ExecuteProcess(
                cmd=['python3', '/root/workspace/camera_setup.py'],
                name='camera_setup',
                output='screen'
            )
        ]
    )
    ld.add_action(camera_setup)

    # Launch Ignition Gazebo
    ign_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments=[('gz_args', f' -r -v 3 {world}')]
    )
    ld.add_action(ign_gazebo)

    # TF Publisher - Map to World
    map_to_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'map', '--child-frame-id', 'world'
        ],
        output='screen'
    )
    ld.add_action(map_to_world)

    # Spawn TurtleBot3 at default position
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='-0.5')
    
    # Declare pose arguments
    declare_x_pose = DeclareLaunchArgument(
        'x_pose',
        default_value='-2.0',
        description='Initial X position of the robot'
    )
    ld.add_action(declare_x_pose)
    
    declare_y_pose = DeclareLaunchArgument(
        'y_pose',
        default_value='-0.5',
        description='Initial Y position of the robot'
    )
    ld.add_action(declare_y_pose)

    spawn_turtlebot3 = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file", urdf,
            "-name", "turtlebot3",
            "-x", x_pose,
            "-y", y_pose,
            "-z", "0.01",
            "-Y", "0.0"
        ],
        output="screen",
    )
    # ld.add_action(spawn_turtlebot3)

    # ROS-Ignition bridge for robot topics
    config_file = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'config',
        'relay_bridge.yaml'
    )
    
    # Add basic bridge if config file doesn't exist
    if os.path.exists(config_file):
        bridge_node = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='parameter_bridge',
            output='screen',
            parameters=[{
                'config_file': config_file
            }]
        )
    else:
        # Basic bridge without config file
        bridge_node = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='parameter_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
            ],
            output='screen'
        )
    ld.add_action(bridge_node)

    # World frame publisher
    world_frame_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0', 
                  '--roll', '0', '--pitch', '0', '--yaw', '0', 
                  '--frame-id', 'world', '--child-frame-id', 'odom'],
        name='static_tf_world_to_odom',
        output='screen'
    )
    ld.add_action(world_frame_publisher)

    return ld
