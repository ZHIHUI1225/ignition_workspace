#!/usr/bin/env python3
"""
Gazebo Trajectory Visualizer for Ignition Gazebo
Creates visual trajectory models directly in Gazebo Ignition simulation
"""

import rclpy
from rclpy.node import Node
import json
import os
import math
from ros_gz_interfaces.srv import SpawnEntity
from ament_index_python.packages import get_package_share_directory


class GazeboTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('gazebo_trajectory_visualizer')
        
        # Parameters
        self.declare_parameter('robot_namespace', 'turtlebot0')
        self.declare_parameter('trajectory_file', '')
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('case', 'simple_maze')
        
        self.robot_namespace = self.get_parameter('robot_namespace').value
        self.trajectory_file = self.get_parameter('trajectory_file').value
        self.robot_id = self.get_parameter('robot_id').value
        self.case = self.get_parameter('case').value
        
        # Colors for different robots
        self.robot_colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
        ]
        
        # Client for spawning entities
        self.spawn_client = self.create_client(SpawnEntity, '/world/default/create')
        
        # Wait for service
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /world/default/create service...')
        
        # Load and visualize trajectory
        self.timer = self.create_timer(1.0, self.spawn_trajectory_once)
        self.trajectory_spawned = False
        
    def create_sphere_sdf(self, name, x, y, z, radius=0.02, color=(1.0, 0.0, 0.0)):
        """Create SDF content for a sphere marker"""
        r, g, b = color
        sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>{radius}</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1.0</ambient>
          <diffuse>{r} {g} {b} 1.0</diffuse>
          <specular>0.1 0.1 0.1 1.0</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        return sdf_content
    
    def create_cylinder_sdf(self, name, x, y, z, height=0.05, radius=0.01, color=(1.0, 0.0, 0.0)):
        """Create SDF content for a cylinder marker"""
        r, g, b = color
        sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1.0</ambient>
          <diffuse>{r} {g} {b} 1.0</diffuse>
          <specular>0.1 0.1 0.1 1.0</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        return sdf_content
    
    def spawn_trajectory_once(self):
        """Spawn trajectory visualization once"""
        if self.trajectory_spawned:
            return
            
        if not self.trajectory_file or not os.path.exists(self.trajectory_file):
            self.get_logger().error(f'Trajectory file not found: {self.trajectory_file}')
            return
            
        try:
            with open(self.trajectory_file, 'r') as f:
                trajectory_data = json.load(f)
                
            trajectory_points = trajectory_data.get('Trajectory', [])
            if not trajectory_points:
                self.get_logger().error('No trajectory points found in file')
                return
                
            self.get_logger().info(f'Spawning {len(trajectory_points)} trajectory markers for robot {self.robot_id}')
            
            # Get robot color
            color = self.robot_colors[self.robot_id % len(self.robot_colors)]
            
            # Spawn start point (larger green sphere)
            start_point = trajectory_points[0]
            start_x = start_point['Position'][0] / 100.0
            start_y = start_point['Position'][1] / 100.0
            start_z = 0.05
            
            self.spawn_marker(
                f'trajectory_start_tb{self.robot_id}',
                self.create_sphere_sdf(f'trajectory_start_tb{self.robot_id}', start_x, start_y, start_z, 0.04, (0.0, 1.0, 0.0))
            )
            
            # Spawn end point (larger red sphere)
            end_point = trajectory_points[-1]
            end_x = end_point['Position'][0] / 100.0
            end_y = end_point['Position'][1] / 100.0
            end_z = 0.05
            
            self.spawn_marker(
                f'trajectory_end_tb{self.robot_id}',
                self.create_sphere_sdf(f'trajectory_end_tb{self.robot_id}', end_x, end_y, end_z, 0.04, (1.0, 0.0, 0.0))
            )
            
            # Spawn trajectory path points (every 10th point to avoid cluttering)
            for i in range(0, len(trajectory_points), 10):
                point = trajectory_points[i]
                x = point['Position'][0] / 100.0
                y = point['Position'][1] / 100.0
                z = 0.02
                
                marker_name = f'trajectory_point_tb{self.robot_id}_{i}'
                self.spawn_marker(
                    marker_name,
                    self.create_cylinder_sdf(marker_name, x, y, z, 0.03, 0.008, color)
                )
            
            # Spawn waypoint markers (every 50th point with larger markers)
            for i in range(0, len(trajectory_points), 50):
                point = trajectory_points[i]
                x = point['Position'][0] / 100.0
                y = point['Position'][1] / 100.0
                z = 0.06
                
                marker_name = f'trajectory_waypoint_tb{self.robot_id}_{i}'
                self.spawn_marker(
                    marker_name,
                    self.create_sphere_sdf(marker_name, x, y, z, 0.025, color)
                )
                
            self.trajectory_spawned = True
            self.get_logger().info(f'Successfully spawned trajectory visualization for robot {self.robot_id}')
            
        except Exception as e:
            self.get_logger().error(f'Error spawning trajectory: {str(e)}')
    
    def spawn_marker(self, name, sdf_content):
        """Spawn a single marker in Gazebo"""
        request = SpawnEntity.Request()
        request.name = name
        request.xml = sdf_content
        request.robot_namespace = ''
        request.initial_pose.position.x = 0.0
        request.initial_pose.position.y = 0.0
        request.initial_pose.position.z = 0.0
        request.initial_pose.orientation.w = 1.0
        
        # Send request asynchronously
        future = self.spawn_client.call_async(request)
        

def main(args=None):
    rclpy.init(args=args)
    
    try:
        visualizer = GazeboTrajectoryVisualizer()
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
