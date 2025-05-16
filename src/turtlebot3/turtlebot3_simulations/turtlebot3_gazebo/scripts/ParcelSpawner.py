#!/usr/bin/env python3
import os
import rclpy
import json
import math
from rclpy.node import Node
from std_msgs.msg import String, Header # String might not be needed anymore
from geometry_msgs.msg import PoseStamped
import tf_transformations as tf
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from ros_gz_interfaces.srv import SpawnEntity
from std_srvs.srv import Trigger # Import Trigger service

class ParcelSpawner(Node):
    def __init__(self):
        super().__init__('parcel_spawner')
        
        # Counter for parcel naming
        self.parcel_counter = 1  # Start from 1 since the initial parcel is named "parcel"
        
        # ADD Service server for spawning the next parcel
        self.spawn_service = self.create_service(
            Trigger,
            '/spawn_next_parcel_service',
            self.handle_spawn_request_service
        )
        self.get_logger().info('Spawn next parcel service is ready.')
        
        # Create a client for the spawn service
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        # Wait for the spawn_entity service to be available
        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('spawn_entity service not available, parcel spawning will fail')
        
        # Store the position of Relaypoint0
        self.relay_point_pose = None
        
        # Load trajectory data
        self.json_file_path = '/root/workspace/data/Trajectory_simulation.json'
        try:
            with open(self.json_file_path, 'r') as json_file:
                data = json.load(json_file)
                self.relay_points = data['RelayPoints']
        except Exception as e:
            self.get_logger().error(f'Error loading trajectory data: {str(e)}')
            self.relay_points = None
            
        # Log initialization
        self.get_logger().info('Parcel spawner initialized')

    def relaypoint_callback(self, msg):
        """Store the position of Relaypoint0"""
        self.relay_point_pose = msg.pose
        self.get_logger().debug('Updated Relaypoint0 position')

    def get_relay_point_position(self):
        """Get the position of Relaypoint0 either from subscription or JSON file"""
        if self.relay_point_pose is not None:
            # Use live data from subscription
            x = self.relay_point_pose.position.x
            y = self.relay_point_pose.position.y
            
            # Extract orientation - a simple approximation since we don't have direct quaternion access
            # In a real implementation you'd use proper quaternion conversion
            theta = 0.0  # Default orientation
            
            return x, y, theta
        elif self.relay_points is not None and len(self.relay_points) > 0:
            # Fallback to JSON data
            relay_point = self.relay_points[0]  # First relay point
            x = relay_point['Position'][0] / 100  # Convert to meters
            y = relay_point['Position'][1] / 100  # Convert to meters
            theta = relay_point['Orientation']
            
            return x, y, theta
        else:
            # Default position if all else fails
            self.get_logger().warn('Using default position for new parcel - no relay point data found')
            return 0.0, 0.0, 0.0
            
    def spawn_new_parcel(self):
        """Spawn a new parcel at Relaypoint0"""
        # Get the position of Relaypoint0
        # x, y, theta = self.get_relay_point_position()
        json_file_path = '/root/workspace/data/Trajectory_simulation.json'
    
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            relay_points = data['RelayPoints']

        x_parcel = relay_points[0]['Position'][0] / 100
        y_parcel = relay_points[0]['Position'][1] / 100
        theta = relay_points[0]['Orientation']

        # Get path to parcel model
        try:
            turtlebot3_gazebo_pkg = get_package_share_directory("turtlebot3_gazebo")
        except PackageNotFoundError:
            # Fallback to local package directory if not in overlay
            script_dir = os.path.dirname(__file__)
            turtlebot3_gazebo_pkg = os.path.abspath(os.path.join(script_dir, '..'))
        parcel_model_path = os.path.join(
            turtlebot3_gazebo_pkg,
            'models',
            "parcel",
            'model.sdf'
        )
        
        # Read SDF file content
        try:
            with open(parcel_model_path, 'r') as file:
                sdf_data = file.read()
        except Exception as e:
            self.get_logger().error(f'Error reading parcel model SDF: {str(e)}')
            return

        # Create a new parcel name
        parcel_name = f"parcel{self.parcel_counter}"
        self.parcel_counter += 1
        # Use SpawnEntity service to spawn the parcel
        request = SpawnEntity.Request()
        request.entity_factory.name = parcel_name
        request.entity_factory.sdf = sdf_data
        request.entity_factory.pose.position.x = float(x_parcel)
        request.entity_factory.pose.position.y = float(y_parcel)
        request.entity_factory.pose.position.z = 0.05  # Small elevation from ground
        # Set orientation using theta
        quat = tf.quaternion_from_euler(0.0, 0.0, theta)
        request.entity_factory.pose.orientation.x = quat[0]
        request.entity_factory.pose.orientation.y = quat[1]
        request.entity_factory.pose.orientation.z = quat[2]
        request.entity_factory.pose.orientation.w = quat[3]
        # Ensure service is available before calling
        if not self.spawn_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('spawn_entity service unavailable at spawn time')
            return
        # Allow renaming to avoid conflicts
        request.entity_factory.allow_renaming = True
        # Log request details
        self.get_logger().debug(f'SpawnEntity request: name={parcel_name}, x={x_parcel:.2f}, y={y_parcel:.2f}, theta={theta:.2f}')
        # Call the service
        future = self.spawn_client.call_async(request)
        future.add_done_callback(self.spawn_callback)
        self.get_logger().info(f'Requested spawn of {parcel_name} at position ({x_parcel:.2f}, {y_parcel:.2f}) with orientation {theta:.2f}')
        
    def spawn_callback(self, future):
        """Handle response from spawn service"""
        # Process spawn service response
        if future.cancelled():
            self.get_logger().error('SpawnEntity request was cancelled')
            return
        if future.exception():
            self.get_logger().error(f'SpawnEntity service exception: {future.exception()}')
            return
        response = future.result()
        if response.success:
            self.get_logger().info('SpawnEntity succeeded: parcel spawned in Gazebo')
        else:
            self.get_logger().error('SpawnEntity failed: parcel not spawned')

    # ADD Service callback
    def handle_spawn_request_service(self, request, response):
        """Handle request to spawn a new parcel via service."""
        self.get_logger().info('Received request to spawn next parcel via /spawn_next_parcel_service.')
        try:
            self.spawn_new_parcel()
            response.success = True
            response.message = "Parcel spawning initiated."
        except Exception as e:
            self.get_logger().error(f"Failed to spawn parcel: {str(e)}")
            response.success = False
            response.message = f"Failed to spawn parcel: {str(e)}"
        return response

def main(args=None):
    rclpy.init(args=args)
    parcel_spawner = ParcelSpawner()
    rclpy.spin(parcel_spawner)
    parcel_spawner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()