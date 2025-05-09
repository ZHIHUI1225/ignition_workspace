import os
import json
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition


def generate_launch_description():
    ld = LaunchDescription()

    TURTLEBOT3_MODEL = "burger"

    enable_drive = LaunchConfiguration("enable_drive", default="true")
    declare_enable_drive = DeclareLaunchArgument(
        name="enable_drive", default_value="true", description="Enable robot drive node"
    )

    turtlebot3_multi_robot = get_package_share_directory("turtlebot3_gazebo")
    launch_file_dir = os.path.join(turtlebot3_multi_robot, "launch")

    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'turtlebot3_maze.world'
    )

    urdf_file_name = "turtlebot3_" + TURTLEBOT3_MODEL + ".urdf"
    print("urdf_file_name : {}".format(urdf_file_name))
    model_folder = 'turtlebot3_burger_fork'
    urdf = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        model_folder,
        'model.sdf'
    )
    urdf_parcel= os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        "parcel",
        'model.sdf'
    )
    if not os.path.exists(world):
        raise FileNotFoundError(f"World file not found: {world}")
    if not os.path.exists(urdf):
        raise FileNotFoundError(f"URDF file not found: {urdf}")

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gzclient.launch.py")
        ),
    )

    ld.add_action(declare_enable_drive)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    num_robots = 4

    x = 0.5
    y = 1.2
    theta=1.256

    json_file_path = '/home/zhihui/data/Trajectory_simulation.json'
    
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        relay_points = data['RelayPoints']
        Waypoints = data['Waypoints']

    x_parcel = relay_points[0]['Position'][0] / 100
    y_parcel = relay_points[0]['Position'][1] / 100
    # Spawn parcel for each robot
    spawn_parcel_cmd = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-file", urdf_parcel,
            "-entity", "parcel",
            "-x", str(x_parcel ),  # Adjust parcel position relative to relay point
            "-y", str(y_parcel ),
            "-z", "0.05",
            "-Y", str(theta),
            "-unpause",
        ],
        output="screen",
    )
    ld.add_action(spawn_parcel_cmd)
    # remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
# Spawn relay points, robots, and parcels
    last_action = None
    for i, relay_point in enumerate(relay_points):
        name = "turtlebot" + str(i)
        namespace = "/tb" + str(i)
        theta = relay_point['Orientation']
        x = relay_point['Position'][0] / 100
        y = relay_point['Position'][1] / 100

        # Spawn relay point
        spawn_relaypoint = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-file",
                os.path.join(turtlebot3_multi_robot, "models", "RelayPoint", "model.sdf"),
                "-entity",
                "Relaypoint" + str(i),
                "-x", str(x),
                "-y", str(y),
                "-z", "0.05",
                "-unpause",
            ],
            output="screen",
        )
        ld.add_action(spawn_relaypoint)

        # Spawn TurtleBot3
        x_bot = x - 0.3 * math.cos(theta)
        y_bot = y - 0.3* math.sin(theta)
        spawn_turtlebot3_burger = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-file", urdf,
                "-entity", name,
                "-robot_namespace", namespace,
                "-x", str(x_bot),
                "-y", str(y_bot),
                "-z", "0.01",
                "-Y", str(theta),
                "-unpause",
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
    parcel_publisher = Node(
        package="turtlebot3_gazebo",
        executable="model_state_publisher.py",
        name="model_state_publisher",
        output="screen",
    )
    ld.add_action(parcel_publisher)
    i=i+1
    goals_x=Waypoints[8]['Position'][0]/100
    goals_y=Waypoints[8]['Position'][1]/100
    spawn_goal=Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-file",
            os.path.join(turtlebot3_multi_robot, "models", "RelayPoint", "model.sdf"),
            "-entity",
            f"Relaypoint"+str(i),
            "-x", str(goals_x),
            "-y", str(goals_y),
            "-z", "0.05",
            "-unpause",
        ],
        output="screen",
    )
    ld.add_action(spawn_goal)
    world_frame_publisher=Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
            name='static_tf_world_to_odom',
            output='screen'
        )
    ld.add_action(world_frame_publisher)
    return ld