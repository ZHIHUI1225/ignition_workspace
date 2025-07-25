#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    # Get the workspace directory
    workspace_dir = os.path.expanduser('/root/workspace')
    
    return LaunchDescription([
        # Camera node that publishes processed video
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                f'source {workspace_dir}/install/setup.bash && ros2 run camera_mocap_launcher camera_node'
            ],
            name='camera_node',
            output='screen',
        ),
        
        # Rectangle detector node that subscribes to processed video
        ExecuteProcess(
            cmd=[
                'bash', '-c', 
                f'source {workspace_dir}/install/setup.bash && ros2 run camera_mocap_launcher rectangle_detector'
            ],
            name='rectangle_detector',
            output='screen',
        ),
    ])
