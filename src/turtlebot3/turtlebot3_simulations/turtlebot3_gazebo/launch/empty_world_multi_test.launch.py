#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, RegisterEventHandler, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
import xacro

def generate_launch_description():
    ld = LaunchDescription()

    # 机器人配置
    robots = [
        {'name': 'tb1', 'x_pose': '-1.5', 'y_pose': '-0.5', 'z_pose': '0.01'},
        {'name': 'tb2', 'x_pose': '-1.5', 'y_pose': '0.5', 'z_pose': '0.01'},
    ]

    # 基础环境配置
    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            os.path.join("/opt/ros/humble", "share"),
            ":" +
            os.path.join(get_package_share_directory('turtlebot3_gazebo'), "models")
        ])

    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time', 
        default_value='true',
        description='Use simulator time'
    )

    # Gazebo启动配置
    world_file = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        "worlds", 'empty_world.world')
    
    ign_gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'),
                        'launch', 'gz_sim.launch.py')]),
        launch_arguments=[('ign_args', [' -r -v1 ' + world_file])]
    )

    # 时钟桥接
    clock_bridge = Node(
        package='ros_gz_bridge', 
        executable='parameter_bridge',
        name='clock_bridge',
        output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    )

    ld.add_action(declare_use_sim_time)
    ld.add_action(ign_resource_path)
    ld.add_action(ign_gz)
    ld.add_action(clock_bridge)

    # 机器人生成逻辑
    last_action = None
    for robot in robots:
        namespace = robot['name']  # Remove leading slash
        sdf_xacro_path = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'models', 'epuck', 'model.sdf.xacro')
        robot_name = namespace
        sdf_generated_path = f"/tmp/{namespace}.sdf"
        xacro_cmd = (
            f"xacro {sdf_xacro_path} namespace:={namespace} robot_name:={robot_name} > {sdf_generated_path}"
        )
        os.system(xacro_cmd)
        spawn_robot = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-file', sdf_generated_path,
                '-name', robot_name,
                '-x', robot['x_pose'],
                '-y', robot['y_pose'],
                '-z', robot['z_pose'],
                '-robot_namespace', namespace
            ],
            namespace=namespace,
            output='screen'
        )
        # 通信桥接
        bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            namespace=namespace,
            arguments=[
                f"/model/{namespace}/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
                f"/model/{namespace}/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            ],
            output='screen'
        )

        # 顺序控制
        if last_action:
            spawn_robot_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_action,
                    on_exit=[spawn_robot, bridge]
                )
            )
            ld.add_action(spawn_robot_event)
        else:
            ld.add_action(spawn_robot)
            ld.add_action(bridge)

        last_action = spawn_robot

    return ld