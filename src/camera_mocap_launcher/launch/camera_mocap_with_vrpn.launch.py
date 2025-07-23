from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os


def generate_launch_description():
    # Path to the vrpn_mocap launch file (update if needed)
    ws_path = '/root/workspace/install/share/vrpn_mocap/launch/client.launch.yaml'
    sys_path = '/opt/ros/humble/share/vrpn_mocap/launch/client.launch.yaml'
    vrpn_launch_file = ws_path if os.path.exists(ws_path) else sys_path

    return LaunchDescription([
        Node(
            package='camera_mocap_launcher',
            executable='camera_node.py',
            name='camera_node',
            output='screen',
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource([vrpn_launch_file]),
            launch_arguments={
                'server': '192.168.0.185',
                'port': '3883',
            }.items(),
        ),
    ])
