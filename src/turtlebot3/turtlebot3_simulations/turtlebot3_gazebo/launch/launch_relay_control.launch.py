import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
from rclpy.time_source import USE_SIM_TIME_NAME
from ament_index_python import get_package_share_directory
import json 
import math


def generate_launch_description():
    ld = LaunchDescription()
    goals = []
    # Create multiple controllers in different namespaces
    for i in range(5):
        with open(f'/root/workspace/data/tb{i}_DiscreteTrajectory.json', 'r') as f:
            data = json.load(f)
        trajectory_data = data["Trajectory"]
        goal = trajectory_data[-1]
        goals.append({'x': goal[0] - 0.07 * math.tan(goal[2]), 'y': goal[1] - 0.07 * math.tan(goal[2]), 'theta': goal[2]})
        namespace = f"tb{i}"

        # Controller node
        controller_node = Node(
                package='turtlebot3_gazebo',
                executable='Follow_controller.py',
                name=f'cmd_vel_publisher_{i}',
                namespace=namespace,
                output='screen',
                parameters=[
                    {'use_sim_time': True},  # Ensure use_sim_time is passed
                    {'namespace': namespace}  # Pass namespace as parameter
                ],
                remappings=[
                (f'/{namespace}/pushing/robot_pose', f'/turtlebot{i}/odom_map'),
                (f'/{namespace}/pushing/object_pose', f'/parcel/pose'),
                (f'/{namespace}/pushing/cmd_vel', f'/turtlebot{i}/cmd_vel'),
                ]
            )
        ld.add_action(controller_node)  # Ensure this is added only once
        nub_relay=i+1
        namespace_r= f"tb{nub_relay}"
        if i==0:
            last_num=0
        else:
            last_num=int(i-1)
        namespace_last= f"tb{last_num}"
        Flag_node=Node(
            package='turtlebot3_gazebo',
            executable='State_switch.py',
            name=f'flag_publisher_{i}',
            namespace=namespace,
            output='screen',
            parameters=[
                {'use_sim_time': True},  # Ensure simulation time is used
                {'namespace': namespace}
            ],
            remappings=[
                (f'/{namespace}/Target/pose', f'/Relaypoint{nub_relay}/pose'),
                (f'/{namespace}/Start/pose', f'/Relaypoint{i}/pose'),
                (f'/{namespace}/parcel/pose', f'/parcel/pose'),
                (f'/{namespace}/Robot/pose', f'/turtlebot{i}/odom_map'),
                (f'/{namespace}/Last/pose', f'/turtlebot{last_num}/odom_map'),
                (f'/{namespace}/Last_flag', f'/{namespace_last}/Pushing_flag'),
                (f'/{namespace}/cmd_vel', f'/turtlebot{i}/cmd_vel'),
                 ]
                 )
        ld.add_action(Flag_node)

        # ld.add_action(Error_node)


        Error_node=Node(
            package='turtlebot3_gazebo',
            executable='Error_plot.py',
            name=f'error_plot_{namespace}',
            output='screen',
            parameters=[{'namespace': namespace}]
        )
        # ld.add_action(Error_node)


     # Add RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
        '-d',
        # Use absolute path with $(HOME) or package-relative path
        os.path.join(
            get_package_share_directory('pushing_controller'),  # Your package name
            'rviz',
            'test.rviz'
            )
        ],
        parameters=[{'use_sim_time': True}]  # Ensure simulation time is used
    )
    ld.add_action(rviz_node)
    return ld