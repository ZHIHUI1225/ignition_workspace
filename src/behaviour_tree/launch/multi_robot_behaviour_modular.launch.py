#!/usr/bin/env python3
"""
Launch file for multi-robot behavior trees using the modular structure.
This launches separate behavior tree instances for each robot using the new modular behaviors.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    """Launch multiple modular behavior tree instances for multi-robot coordination."""
    
    # Launch Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    num_robots_config = LaunchConfiguration('num_robots', default='3')
    
    # Launch description
    ld = LaunchDescription()
    
    # Common launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'))
    
    ld.add_action(DeclareLaunchArgument(
        'num_robots',
        default_value='1',
        description='Number of robots to launch behavior trees for'))
    
    # Use OpaqueFunction to handle dynamic robot count properly
    def create_robot_nodes(context):
        """Create robot nodes based on the num_robots parameter."""
        num_robots = int(context.launch_configurations['num_robots'])
        actions = []
        
        for i in range(num_robots):
            robot_namespace = f'turtlebot{i}'
            tree_node_name = f'tree_{i}'
            
            # Create a group for each robot's behavior tree
            robot_group = GroupAction([
                PushRosNamespace(robot_namespace),
                
                # Modular behavior tree node for this robot
                Node(
                    package='behaviour_tree',
                    executable='my_behaviour_tree_modular',
                    name=tree_node_name,
                    output='screen',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'robot_id': i,
                        'robot_namespace': robot_namespace,
                        'tree_name': f'BehaviorTree_{i}'
                    }],
                    remappings=[
                        # Robot-specific topic remappings
                        ('Ready_flag', f'/{robot_namespace}/Ready_flag'),
                        ('Pushing_flag', f'/{robot_namespace}/Pushing_flag'),
                        ('Pickup_flag', f'/{robot_namespace}/Pickup_flag'),
                        ('PickUpDone', f'/{robot_namespace}/PickUpDone'),
                        ('cmd_vel', f'/{robot_namespace}/cmd_vel'),
                        ('odom', f'/{robot_namespace}/odom'),
                        # Snapshot stream topics with robot-specific namespace
                        ('tree_log', f'/{robot_namespace}/tree_log'),
                        ('tree_snapshot', f'/{robot_namespace}/tree_snapshot'),
                        ('tree_updates', f'/{robot_namespace}/tree_updates'),
                        ('snapshot_streams', f'/{robot_namespace}/snapshot_streams')
                    ]
                )
            ])
            
            actions.append(robot_group)
        
        return actions
    
    # Import OpaqueFunction for dynamic launch configuration handling
    from launch.actions import OpaqueFunction
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
    # Add the dynamic robot node creation
    ld.add_action(OpaqueFunction(function=create_robot_nodes))
    return ld
