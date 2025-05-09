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
    
    # Define different goals for each robot
    goals = []
    
    # Create multiple controllers in different namespaces
    for i in range(5):
        
        with open('/home/zhihui/data/tb'+str(i)+'_DiscreteTrajectory.json', 'r') as f:
            data = json.load(f)
            trajectory_data = data["Trajectory"]
        goal=trajectory_data[-1]
        goals.append({'x': goal[0]-0.14*math.cos(goal[2]), 'y': goal[1]-0.14*math.sin(goal[2]), 'theta': goal[2]})
        namespace = f"tb{i}"
        
        controller_node = Node(
            package='pushing_controller',
            executable='pushing_controller_node',
            name=f'pushing_controller_{i}',
            namespace=namespace,
            output='screen',
            parameters=[{USE_SIM_TIME_NAME:True}],
            remappings=[
                (f'/{namespace}/pushing/robot_pose', f'/{namespace}/turtlebot{i}/pose'),
                (f'/{namespace}/pushing/object_pose', f'/{namespace}/parcel{i}/pose'),
                (f'/{namespace}/pushing/cmd_vel', f'/{namespace}/cmd_vel'),
            ]
        )
        ld.add_action(controller_node)
        
        # Create a client for each controller with different goals
        client_node = Node(
            package='pushing_controller',
            executable='Robot_trajectory_follow.py',
            name=f'move_robot_to_client_{i}',
            namespace=namespace,
            output='screen',
            parameters=[
                {USE_SIM_TIME_NAME:True},
                {'goal_x': goals[i]['x']},
                {'goal_y': goals[i]['y']},
                {'goal_theta': goals[i]['theta']}
            ]
        )
        ld.add_action(client_node)
    
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
        parameters=[{'use_sim_time': True}]
    )
    ld.add_action(rviz_node)
    
    return ld