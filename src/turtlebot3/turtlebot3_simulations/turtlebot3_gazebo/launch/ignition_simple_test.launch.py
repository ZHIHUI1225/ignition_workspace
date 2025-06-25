import os
import json
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, IncludeLaunchDescription, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.actions import ExecuteProcess, RegisterEventHandler
from std_msgs.msg import Bool


def generate_launch_description():
    ld = LaunchDescription()
    case='simple_maze'
    json_file_path = '/root/workspace/data/'+case+'/Waypoints.json'
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
        case+'_world.world'
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
    urdf_parcel= os.path.join(
        turtlebot3_gazebo_pkg,
        'models',
        "parcel",
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

    # Launch Ignition Gazebo
    ign_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments=[('gz_args', ' -r -v 3 ' + world)]
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


    
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        relay_points = data['RelayPoints']
        Waypoints = data['Waypoints']
    N_w=len(Waypoints)
    N=len(relay_points)
    x_parcel = relay_points[0]['Position'][0] / 100
    y_parcel = relay_points[0]['Position'][1] / 100
    theta = relay_points[0]['Orientation']

    # Spawn parcel for each robot (initial, if you ever enable it)
    spawn_parcel_cmd = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file", urdf_parcel,
            "-name", "parcel0",
            "-x", str(x_parcel),
            "-y", str(y_parcel),
            "-z", "0.05",
            "-Y", str(theta),
            "-allow_renaming", "true"
        ],
        output="screen",
    )
    ld.add_action(spawn_parcel_cmd)

    # --- Add subscriber handler for '/spawn_next_parcel' ---

  
    listener_node = Node(
        package='turtlebot3_gazebo',
        executable='spawn_listener.py',
        name='parcel_spawner',
        output='screen',
        parameters=[{
            'urdf_path': urdf_parcel, # 从launch文件传递URDF路径
            'json_file_path': json_file_path,  # Pass the JSON file path as a parameter
        }]
    )
    ld.add_action(listener_node)
    # --- end subscriber setup ---

    # Spawn relay points, robots, and parcels
    last_action = None
    for i, relay_point in enumerate(relay_points):
        name = "turtlebot" + str(i)
        namespace = name  # Use robot name as namespace for consistency
        theta = relay_point['Orientation']
        x = relay_point['Position'][0] / 100
        y = relay_point['Position'][1] / 100

        # Spawn relay point
        spawn_relaypoint = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-file", os.path.join(turtlebot3_gazebo_pkg, "models", "RelayPoint", "model.sdf"),
                "-name", "Relaypoint" + str(i),
                "-x", str(x),
                "-y", str(y),
                "-z", "0.001"
            ],
            output="screen",
        )
        ld.add_action(spawn_relaypoint)

        # Spawn TurtleBot3
        x_bot = x - 0.3 * math.cos(theta)
        y_bot = y - 0.3 * math.sin(theta)
        spawn_turtlebot3_burger = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-file", urdf,
                "-name", name,
                "-x", str(x_bot),
                "-y", str(y_bot),
                "-z", "0.01",
                "-Y", str(theta),
                "--ros-args", "--remap", f"name:={namespace}"
            ],
            output="screen",
        )

        # Add TurtleBot3 spawn action
        if last_action is None:
            ld.add_action(spawn_turtlebot3_burger)
        else:
            spawn_turtlebot3_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_action,
                    on_exit=[spawn_turtlebot3_burger],
                )
            )
            ld.add_action(spawn_turtlebot3_event)

        last_action = spawn_turtlebot3_burger

        # Add TF publisher for robot
        ld.add_action(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                '--x', str(x_bot),
                '--y', str(y_bot),
                '--z', '0',
                '--yaw', str(theta),
                '--frame-id', 'map',
                '--child-frame-id', f'{name}/odom'
            ],
            name=f'init_{name}_odom',
            output='screen'
        ))

        # Add pose publisher
        ld.add_action(Node(
            package='turtlebot3_gazebo',
            executable='tb_pose_publish.py',
            name=f'{name}_pose_pub',
            namespace=namespace,
            parameters=[{
                'namespace': namespace
            }],
            output='screen'
        ))

        # Add trajectory visualizer for each robot
        trajectory_file_path = f'/root/workspace/data/{case}/tb{i}_Trajectory.json'
        ld.add_action(Node(
            package='turtlebot3_gazebo',
            executable='trajectory_visualizer.py',
            name=f'{name}_trajectory_visualizer',
            namespace=namespace,
            parameters=[{
                'robot_namespace': namespace,
                'trajectory_file': trajectory_file_path,
                'robot_id': i
            }],
            output='screen'
        ))

    # Add model state publisher
    parcel_publisher = Node(
        package="turtlebot3_gazebo",
        executable="model_state_publisher.py",
        name="model_state_publisher",
        output="screen",
    )
    ld.add_action(parcel_publisher)

    # Spawn final goal
    i = i + 1
    goals_x = Waypoints[N_w-1]['Position'][0]/100
    goals_y = Waypoints[N_w-1]['Position'][1]/100
    spawn_goal = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file", os.path.join(turtlebot3_gazebo_pkg, "models", "RelayPoint", "model.sdf"),
            "-name", f"Relaypoint{i}",
            "-x", str(goals_x),
            "-y", str(goals_y),
            "-z", "0.05",
        ],
        output="screen",
    )
    ld.add_action(spawn_goal)

    # ROS-Ignition bridge with service forwarding
    config_file = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'config',
        'relay_bridge.yaml'
    )
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        output='screen',
        parameters=[{
            'config_file': config_file,
            'frequency': 20.0  # Set parcel pose publishing frequency to 10 Hz
        }]
    )
    ld.add_action(bridge_node)

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

    # Launch RViz2 for trajectory visualization
    rviz_config_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'rviz',
        'trajectory_visualization.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path] if os.path.exists(rviz_config_path) else [],
        output='screen'
    )
    # ld.add_action(rviz_node)

    return ld