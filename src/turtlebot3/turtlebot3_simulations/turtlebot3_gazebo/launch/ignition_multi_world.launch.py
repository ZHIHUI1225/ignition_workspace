import os
import json
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
# from launch.conditions import IfCondition # Removed IfCondition as enable_drive is removed

def generate_launch_description():
    ld = LaunchDescription()

    TURTLEBOT3_MODEL = "burger" # Keep using burger model specified here
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')


    turtlebot3_gazebo_pkg = get_package_share_directory("turtlebot3_gazebo")
    # turtlebot3_common_pkg = get_package_share_directory("turtlebot3_common") # Get common package path

    # launch_file_dir = os.path.join(turtlebot3_gazebo_pkg, "launch") # Not used

    world_name = 'turtlebot3_maze' # Define world name
    world = os.path.join(
        turtlebot3_gazebo_pkg,
        'worlds',
        f'{world_name}.world'
    )

    # Corrected model path to use the SDF directly as updated previously
    model_folder = 'turtlebot3_burger_fork' # Assuming the updated SDF is in the standard burger folder
    urdf = os.path.join(
        turtlebot3_gazebo_pkg,
        'models',
        model_folder,
        'model.sdf' # Use the SDF file directly
    )
    urdf_parcel= os.path.join(
        turtlebot3_gazebo_pkg,
        'models',
        "parcel", # Assuming parcel model exists
        'model.sdf'
    )
    if not os.path.exists(world):
        raise FileNotFoundError(f"World file not found: {world}")
    if not os.path.exists(urdf):
        raise FileNotFoundError(f"Robot SDF file not found: {urdf}")
    if not os.path.exists(urdf_parcel):
        raise FileNotFoundError(f"Parcel SDF file not found: {urdf_parcel}")

    # Set Ignition Gazebo resource path
    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            os.path.join(get_package_share_directory('turtlebot3_gazebo'), "models"),
            ":" +
            os.path.join("/opt/ros/humble", "share")])
    ld.add_action(ign_resource_path)

    # Disable shared memory transport for Fast RTPS
    disable_shm_transport = SetEnvironmentVariable(
        name='RMW_FASTRTPS_USE_SHM',
        value='0'
    )
    ld.add_action(disable_shm_transport)

    # Declare use_sim_time argument
    ld.add_action(
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='If true, use simulated clock')
    )

    # Include Ignition Gazebo launch file
    ld.add_action(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(get_package_share_directory('ros_gz_sim'),
                              'launch', 'gz_sim.launch.py')]),
            # Pass world file and other arguments to ign gazebo
            launch_arguments=[('gz_args', ' -r -v 3 ' + world)]
        )
    )

    # TF Publisher - Update to modern argument style
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

    num_robots = 4 # Keep this logic if needed, though not directly used below

    # Load relay points from JSON
    json_file_path = '/root/workspace/data/Trajectory_simulation.json' # Corrected path
    if not os.path.exists(json_file_path):
         raise FileNotFoundError(f"Trajectory file not found: {json_file_path}")

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        relay_points = data['RelayPoints']
        Waypoints = data['Waypoints']

    # Spawn relay points, robots, and parcels
    last_spawn_action = None
    for i, relay_point in enumerate(relay_points):
        robot_name = "turtlebot" + str(i)
        # Use robot_name as the namespace for consistency
        robot_namespace = robot_name
        relay_name = "Relaypoint" + str(i)
        parcel_name = "parcel" + str(i)

        theta = relay_point['Orientation']
        x = relay_point['Position'][0] / 100
        y = relay_point['Position'][1] / 100

        # Spawn relay point using ros_gz_sim create
        spawn_relaypoint = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-file", os.path.join(turtlebot3_gazebo_pkg, "models", "RelayPoint", "model.sdf"),
                "-name", relay_name,
                "-x", str(x),
                "-y", str(y),
                "-z", "0.01",
                "-Y", str(theta) # Assuming RelayPoint model is oriented along X
            ],
            output="screen",
        )
        ld.add_action(spawn_relaypoint) # Spawn relay points immediately

        # Spawn TurtleBot3 using ros_gz_sim create
        x_bot = x - 0.12 * math.cos(theta) # Adjust offset as needed
        y_bot = y - 0.12 * math.sin(theta)
        spawn_turtlebot3_burger = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-file", urdf,
                "-name", robot_name,
                "-x", str(x_bot),
                "-y", str(y_bot),
                "-z", "0.01",
                "-Y", str(theta),
                # Place remap parameters at the end, separated from other args
                "--ros-args", "--remap", f"name:={robot_namespace}"
            ],
            output="screen",
        )

        # Spawn parcel using ros_gz_sim create
        x_parcel = x # Adjust parcel position relative to relay point if needed
        y_parcel = y
        spawn_parcel_cmd = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-file", urdf_parcel,
                "-name", parcel_name,
                "-x", str(x_parcel),
                "-y", str(y_parcel),
                "-z", "0.05",
                "-Y", str(theta), # Assuming parcel model is oriented along X
                "-allow_renaming", "true" # Useful if names might clash
            ],
            output="screen",
        )
        ld.add_action(spawn_parcel_cmd) # Spawn parcels immediately

        if last_spawn_action:
            ld.add_action(RegisterEventHandler(
                OnProcessExit(
                    target_action=last_spawn_action,
                    on_exit=[ spawn_turtlebot3_burger]
                )
            ))
        else:
            ld.add_action( spawn_turtlebot3_burger)
        last_spawn_action =spawn_turtlebot3_burger

        ld.add_action(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                # Use new-style named arguments
                '--x', str(x_bot),
                '--y', str(y_bot),
                '--z', '0',
                '--yaw', str(theta),
                # No roll/pitch, defaults to 0
                '--frame-id', 'map',
                '--child-frame-id', f'{robot_name}/odom'
            ],
            name=f'init_{robot_name}_odom',
            output='screen'
        ))

        ld.add_action(Node(
            package='turtlebot3_gazebo',
            executable='tb_pose_publish.py',
            name=f'{robot_name}_pose_pub',
            namespace=robot_namespace,
            parameters=[{
                'namespace': robot_namespace # Ensure namespace is passed correctly
            }],
            output='screen'
        ))
        
        # navigation_nodes = [
        #     # Robot State Publisher
        #     Node(
        #         package='robot_state_publisher',
        #         executable='robot_state_publisher',
        #         namespace=robot_name,
        #         parameters=[{
        #             'use_sim_time': use_sim_time,
        #             'robot_description': f"<robot name='{robot_name}'"
        #         }],
        #         output='screen'
        #     ),
            # Node(
            #     package='nav2_amcl',
            #     executable='amcl',
            #     namespace=robot_name,
            #     parameters=[{
            #         'use_sim_time': use_sim_time,
            #         'odom_frame_id': f'{robot_name}/odom',
            #         'base_frame_id': f'{robot_name}/base_footprint',
            #         'global_frame_id': 'map',  # 关键：统一全局坐标系
            #         'initial_pose': {
            #             'x': x_bot,
            #             'y': y_bot,
            #             'yaw': theta
            #         },
            #         'set_initial_pose': True
            #     }],
            #     output='screen'
            # ),
            # 控制器节点
        #     Node(
        #         package='nav2_controller',
        #         executable='controller_server',
        #         namespace=robot_name,
        #         parameters=[{
        #             'use_sim_time': use_sim_time,
        #             'controller_frequency': 20.0
        #         }],
        #         output='screen'
        #     )
        # ]
        # for node in navigation_nodes:
        #     ld.add_action(node)


    # Spawn final goal point
    final_goal_index = i + 1 # Index for the final goal
    final_goal_name = f"Relaypoint{final_goal_index}"
    goals_x = Waypoints[8]['Position'][0]/100 # Assuming Waypoint 8 is the final goal
    goals_y = Waypoints[8]['Position'][1]/100
    spawn_goal = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file", os.path.join(turtlebot3_gazebo_pkg, "models", "RelayPoint", "model.sdf"),
            "-name", final_goal_name,
            "-x", str(goals_x),
            "-y", str(goals_y),
            "-z", "0.05",
        ],
        output="screen",
    )
    # Ensure final goal spawns after the last robot
    spawn_final_goal_event = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_parcel_cmd
,
            on_exit=[spawn_goal],
        )
    )
    ld.add_action(spawn_final_goal_event)



    config_file = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),  # Replace with your package name
        'config',  # Folder where `odom_bridge.yaml` is stored
        'odom_bridge.yaml'
        )
    # config_file = '/root/workspace/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/config/odom_bridge.yaml'
    # Replace the existing bridge node with a properly configured one
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        output='screen',
        parameters=[{
            'config_file':config_file
        }]
    )
    ld.add_action(bridge_node)
    return ld