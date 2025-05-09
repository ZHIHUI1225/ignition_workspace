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
    for i in range(1):
        
        with open('/root/workspace/data/tb'+str(i)+'_DiscreteTrajectory.json', 'r') as f:
            data = json.load(f)
            trajectory_data = data["Trajectory"]
        goal=trajectory_data[-1]
        goals.append({'x': goal[0]-0.07*math.tan(goal[2]), 'y': goal[1]-0.07*math.tan(goal[2]), 'theta': goal[2]})
        namespace = f"tb{i}"
        
        controller_node = Node(
            package='pushing_controller',
            executable='MPCcontroller.py',
            name=f'pushing_controller_{i}',
            namespace=namespace,
            output='screen',
            parameters=[{USE_SIM_TIME_NAME:True}],
            remappings=[
                (f'/{namespace}/pushing/robot_pose', f'/turtlebot{i}/odom_map'),
                (f'/{namespace}/pushing/object_pose', f'/parcel{i}/pose'),
                (f'/{namespace}/pushing/cmd_vel', f'/turtlebot{i}/cmd_vel'),
            ]
        )
        ld.add_action(controller_node)
        
        # Create a client for each controller with different goals
        client_node = Node(
            package='pushing_controller',
            executable='Pushing_trajectory_define.py',
            name=f'pushing_trajectory_client_{i}',
            namespace=namespace,
            output='screen'
        )
        ld.add_action(client_node)

        # Error_node=Node(
        #     package='turtlebot3_gazebo',
        #     executable='Error_plot.py',
        #     name=f'error_plot_{namespace}',
        #     output='screen',
        #     parameters=[{'namespace': namespace}]
        # )
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
        parameters=[{'use_sim_time': True}]
    )
    ld.add_action(rviz_node)
    return ld