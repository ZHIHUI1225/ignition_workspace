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
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    model_folder = 'epuck'
    urdf = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        model_folder,
        'test.sdf'
    )
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.5')
    y_pose = LaunchConfiguration('y_pose', default='0.65')
    yaw= LaunchConfiguration('yaw', default='1.57079')
    x_pose_parcel = LaunchConfiguration('x_pose_parcel', default='0.5')
    y_pose_parcel = LaunchConfiguration('y_pose_parcel', default='0.8')
    yaw_parcel= LaunchConfiguration('yaw_parcel', default='0')
    x_pose_RP = LaunchConfiguration('x_pose_RP', default='0.5')
    y_pose_RP = LaunchConfiguration('y_pose_RP', default='3')
    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'turtlebot3_maze.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # for i in range(4):
    #     name = "turtlebot" + str(i)
    #     namespace = "/tb" + str(i) 
    #     x_pose = 0.5 + 0.2 * i
        
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # spawn_turtlebot_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
    #     ),
    #     launch_arguments={
    #         'x_pose': x_pose,
    #         'y_pose': y_pose,
    #         'theta': yaw
    #     }.items()
    # )
    x_pose = 0.5
    y_pose = 0.65
    yaw = 1.57079
    spawn_epuck2 = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-file", urdf,
            "-entity", "epuck2",
            "-robot_namespace", "epuck",
            "-x", str(x_pose),
            "-y", str(y_pose),
            "-z", "0.05",
            "-Y", str(yaw),
            "-unpause",
        ],
        output="screen",
    )
    

    #load the parcel model
    spawn_parcel_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_parcel.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose_parcel,
            'y_pose': y_pose_parcel,
            'theta': yaw_parcel
        }.items()
    )
     #load the relay points model
    spawn_relaypoints_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_relaypoints.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose_RP,
            'y_pose': y_pose_RP
        }.items()
    )
    world_frame_publisher=Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            name='static_tf_world_to_odom',
            output='screen'
        )
   
    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(world_frame_publisher)
    ld.add_action(spawn_epuck2 )
    ld.add_action(spawn_parcel_cmd)
    ld.add_action(spawn_relaypoints_cmd)

    return ld
