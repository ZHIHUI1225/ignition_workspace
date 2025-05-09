import os
import json
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    ld = LaunchDescription()
    TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']   # waffle
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name='turtlebot3_maze'
    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'turtlebot3_maze.world'
    )

    model_folder = 'epuck'
    urdf = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        model_folder,
        'model.sdf'
    )
    urdf_parcel = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        "cube",
        'model.sdf'
    )
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')

    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            os.path.join(get_package_share_directory('turtlebot3_gazebo'), "models"),
            ":" +  # Linux路径分隔符
            os.path.join("/opt/ros/humble", "share")])
    ld.add_action(ign_resource_path)

    num_robots = 4
  
    
        # Path to the JSON file
    data_dir = '/root/workspace/data'
    os.makedirs(data_dir, exist_ok=True)
    json_file_path = os.path.join(data_dir, 'Trajectory_simulation.json')
    
    # Read the data from the file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        relay_points = data['RelayPoints']
        Waypoints = data['Waypoints']

    last_action = None
    for i, relay_point in enumerate(relay_points):
        name = "turtlebot" + str(i)
        namespace = "/tb" + str(i)
        theta = relay_point['Orientation']
        x = relay_point['Position'][0] / 100
        y = relay_point['Position'][1] / 100
        # Spawn relay point
        spawn_relaypoint = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                '-entity', "Relaypoint" + str(i),
                "-file", os.path.join(get_package_share_directory('turtlebot3_gazebo'), "models", "RelayPoint", "model.sdf"),
                "-name", "Relaypoint" + str(i),
                "-x", str(x),
                "-y", str(y),
                "-z", "0.001",
                "-Y", str(theta),
                "--wait", "1" 
            ],
            output="screen",
        )
        ld.add_action(spawn_relaypoint)
        
        # Spawn TurtleBot3
        x_bot = x - 0.14 * math.cos(theta)
        y_bot = y - 0.14 * math.sin(theta)
        spawn_turtlebot3_burger = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-entity", name,
                "-file", urdf,
                "-name", name,
                "-x", str(x_bot),
                "-y", str(y_bot),
                "-z", "0.01",
                "-Y", str(theta)
            ],
            output="screen",
        )

        # Spawn parcel
        x_parcel = x - 0.07 * math.cos(theta)
        y_parcel = y - 0.07 * math.sin(theta)
        spawn_parcel_cmd = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-entity", "parcel" + str(i),
                "-file", urdf_parcel,
                "-name", "parcel" + str(i),
                "-x", str(x_parcel),
                "-y", str(y_parcel),
                "-z", "0.05",
                "-Y", str(theta),
                "-static", "false" 
            ],
            output="screen",
        )
        ld.add_action(spawn_parcel_cmd)

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

    # Spawn final goal
    i += 1
    goals_x = Waypoints[8]['Position'][0]/100
    goals_y = Waypoints[8]['Position'][1]/100
    spawn_goal = Node(
        package="ros_gz_sim",  # Updated from ros_ign_gazebo to ros_gz_sim for Fortress
        executable="create",
        arguments=[
            "-entity",  f"Relaypoint"+str(i),
            "-file", os.path.join(get_package_share_directory('turtlebot3_gazebo'), "models", "RelayPoint", "model.sdf"),
            "-name", f"Relaypoint"+str(i),
            "-x", str(goals_x),
            "-y", str(goals_y),
            "-z", "0.05"
        ],
        output="screen",
    )
    ld.add_action(spawn_goal)

    # TF Publisher
    world_frame_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'world', 'odom'],
        name='static_tf_world_to_odom',
        output='both'
    )
    ld.add_action(world_frame_publisher)
    ld.add_action(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(get_package_share_directory('ros_gz_sim'),
                              'launch', 'gz_sim.launch.py')]),
            launch_arguments=[('ign_args', ' -r -v 3 ' + world)]
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='If true, use simulated clock')
    )

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