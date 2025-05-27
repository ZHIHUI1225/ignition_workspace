#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger # Changed from Bool
import json
import os
# spawn_listener.py
import rclpy
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from rclpy.node import Node
import subprocess

class SpawnListener(Node):
    def __init__(self):
        super().__init__('spawn_listener')
        # Changed from subscription to service server
        self.srv = self.create_service(
            Trigger,
            '/spawn_next_parcel_service',  # Service name
            self.service_callback  # New service callback method
        )
        self.counter = 0
        
        # Declare and get parameters
        self.declare_parameter('json_file_path', '')  # Declare the parameter with a default empty string
        json_file_path = self.get_parameter('json_file_path').get_parameter_value().string_value
        
        self.get_logger().info(f'Using JSON file path: {json_file_path}')
        
        # Load trajectory data
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            relay_points = data['RelayPoints']
            Waypoints = data['Waypoints']

        self.x_parcel = relay_points[0]['Position'][0] / 100
        self.y_parcel = relay_points[0]['Position'][1] / 100
        self.theta = relay_points[0]['Orientation']
       # Get path to parcel model
        try:
            turtlebot3_gazebo_pkg = get_package_share_directory("turtlebot3_gazebo")
        except PackageNotFoundError:
            # Fallback to local package directory if not in overlay
            script_dir = os.path.dirname(__file__)
            turtlebot3_gazebo_pkg = os.path.abspath(os.path.join(script_dir, '..'))
        self.parcel_model_path = os.path.join(
            turtlebot3_gazebo_pkg,
            'models',
            "parcel",
            'model.sdf'
        )

    def service_callback(self, request, response): # Changed method signature for service
        # Logic from old callback, msg.data is not used anymore as Trigger has no data
        self.counter += 1
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', self.parcel_model_path,
            '-name', f'parcel{self.counter}',
            '-x', str(self.x_parcel), '-y', str(self.y_parcel), '-z', '0.05',
            "-Y", str(self.theta),
            "-allow_renaming", "true"
        ]
        subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self.get_logger().info(f"生成第{self.counter}个包裹节点 (via service)")
        response.success = True  # Indicate service call was successful
        response.message = f"Spawned parcel{self.counter}"
        return response

def main(args=None):
    rclpy.init(args=args)
    spawn_listener = SpawnListener()
    try:
        rclpy.spin(spawn_listener)
    except KeyboardInterrupt:
        pass
    finally:
        spawn_listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()