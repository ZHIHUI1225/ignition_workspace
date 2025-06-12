#!/usr/bin/env python3
"""
Trajectory Visualizer Node
Loads trajectory points from JSON files and visualizes them in Gazebo Ignition
"""

import rclpy
from rclpy.node import Node
import json
import os
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math


class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer')
        
        # Parameters - support both individual robot mode and multi-robot mode
        self.declare_parameter('robot_namespace', '')
        self.declare_parameter('trajectory_file', '')
        self.declare_parameter('robot_id', 0)
        
        # Multi-robot mode parameters (only used when not in individual mode)
        self.declare_parameter('case', 'simple_maze')
        self.declare_parameter('robot_count', 1)
        self.declare_parameter('data_path', '/root/workspace/data')
        
        self.robot_namespace = self.get_parameter('robot_namespace').get_parameter_value().string_value
        self.trajectory_file = self.get_parameter('trajectory_file').get_parameter_value().string_value
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        
        # Check if running in individual robot mode
        self.individual_robot_mode = bool(self.robot_namespace and self.trajectory_file)
        
        if self.individual_robot_mode:
            self.get_logger().info(f"Trajectory Visualizer started for individual robot: {self.robot_namespace}")
        else:
            # Only get multi-robot parameters when needed
            self.case = self.get_parameter('case').get_parameter_value().string_value
            self.robot_count = self.get_parameter('robot_count').get_parameter_value().integer_value
            self.data_path = self.get_parameter('data_path').get_parameter_value().string_value
            self.get_logger().info(f"Trajectory Visualizer started for case: {self.case}")
        
        # Publishers for trajectories
        self.path_publishers = {}
        self.marker_publishers = {}
        
        # Colors for different robots
        self.colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow
            ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),  # Magenta
            ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),  # Cyan
        ]
        
        # Create publishers based on mode
        if self.individual_robot_mode:
            # Single robot mode - create publishers for this specific robot
            self.path_publishers[self.robot_id] = self.create_publisher(
                Path, f'/trajectory_path_tb{self.robot_id}', 10)
            
            self.marker_publishers[self.robot_id] = self.create_publisher(
                MarkerArray, f'/trajectory_markers_tb{self.robot_id}', 10)
        else:
            # Multi-robot mode - create publishers for each robot
            for i in range(self.robot_count):
                # Path publisher for rviz-style visualization
                self.path_publishers[i] = self.create_publisher(
                    Path, f'/trajectory_path_tb{i}', 10)
                
                # Marker publisher for Gazebo visualization
                self.marker_publishers[i] = self.create_publisher(
                    MarkerArray, f'/trajectory_markers_tb{i}', 10)
        
        # Load and publish trajectories
        self.load_and_publish_trajectories()
        
        # Timer to republish markers periodically (Gazebo sometimes needs this)
        self.timer = self.create_timer(5.0, self.republish_trajectories)
        
    def load_and_publish_trajectories(self):
        """Load trajectory data from JSON files and publish as paths and markers"""
        if self.individual_robot_mode:
            # Single robot mode - load the specific trajectory file
            if os.path.exists(self.trajectory_file):
                # self.get_logger().info(f"Loading trajectory for robot {self.robot_namespace} from {self.trajectory_file}")
                self.load_trajectory(self.robot_id, self.trajectory_file)
            else:
                self.get_logger().warn(f"Trajectory file not found: {self.trajectory_file}")
        else:
            # Multi-robot mode - load all robot trajectories
            for i in range(self.robot_count):
                trajectory_file = os.path.join(
                    self.data_path, self.case, f'tb{i}_Trajectory.json')
                
                if os.path.exists(trajectory_file):
                    # self.get_logger().info(f"Loading trajectory for robot tb{i}")
                    self.load_trajectory(i, trajectory_file)
                else:
                    self.get_logger().warn(f"Trajectory file not found: {trajectory_file}")
    
    def load_trajectory(self, robot_id, trajectory_file):
        """Load a single robot's trajectory and publish it"""
        try:
            with open(trajectory_file, 'r') as f:
                data = json.load(f)
            
            trajectory_points = data['Trajectory']
            self.get_logger().info(f"Loaded {len(trajectory_points)} trajectory points for tb{robot_id}")
            
            # Create Path message for RViz-style visualization
            path_msg = self.create_path_message(trajectory_points, robot_id)
            
            # Create Marker messages for Gazebo visualization
            marker_array = self.create_trajectory_markers(trajectory_points, robot_id)
            
            # Publish both
            self.path_publishers[robot_id].publish(path_msg)
            self.marker_publishers[robot_id].publish(marker_array)
            
            self.get_logger().info(f"Published trajectory visualization for tb{robot_id}")
            
        except Exception as e:
            self.get_logger().error(f"Error loading trajectory for tb{robot_id}: {str(e)}")
    
    def create_path_message(self, trajectory_points, robot_id):
        """Create a Path message from trajectory points"""
        path_msg = Path()
        path_msg.header.frame_id = 'world'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in trajectory_points:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'world'
            pose_stamped.header.stamp = path_msg.header.stamp
            
            # Extract position and orientation from trajectory point
            # Format: [x, y, theta, v, omega]
            x, y, theta = point[0], point[1], point[2]
            
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.05  # Slightly above ground
            
            # Convert yaw to quaternion
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = math.sin(theta / 2.0)
            pose_stamped.pose.orientation.w = math.cos(theta / 2.0)
            
            path_msg.poses.append(pose_stamped)
        
        return path_msg
    
    def create_trajectory_markers(self, trajectory_points, robot_id):
        """Create marker array for Gazebo visualization"""
        marker_array = MarkerArray()
        
        # Use larger, more visible markers for Gazebo Ignition
        # Line strip marker for the trajectory path
        line_marker = Marker()
        line_marker.header.frame_id = 'world'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = f'trajectory_tb{robot_id}'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # Increased line width for better visibility
        line_marker.color = self.colors[robot_id % len(self.colors)]
        line_marker.pose.orientation.w = 1.0
        line_marker.lifetime.sec = 0  # Persistent markers
        
        # Add all trajectory points to the line strip using Point messages
        for point in trajectory_points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1]) 
            p.z = 0.1  # Higher for better visibility
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # Add start and end point markers with better visibility
        if trajectory_points:
            # Start point marker (large green sphere)
            start_marker = Marker()
            start_marker.header.frame_id = 'world'
            start_marker.header.stamp = self.get_clock().now().to_msg()
            start_marker.ns = f'trajectory_tb{robot_id}'
            start_marker.id = 1
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD
            start_marker.pose.position.x = float(trajectory_points[0][0])
            start_marker.pose.position.y = float(trajectory_points[0][1])
            start_marker.pose.position.z = 0.15
            start_marker.pose.orientation.w = 1.0
            start_marker.scale.x = 0.2
            start_marker.scale.y = 0.2
            start_marker.scale.z = 0.2
            start_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Bright green for start
            start_marker.lifetime.sec = 0  # Persistent
            
            marker_array.markers.append(start_marker)
            
            # End point marker (large red cube)
            end_marker = Marker()
            end_marker.header.frame_id = 'world'
            end_marker.header.stamp = self.get_clock().now().to_msg()
            end_marker.ns = f'trajectory_tb{robot_id}'
            end_marker.id = 2
            end_marker.type = Marker.CUBE
            end_marker.action = Marker.ADD
            end_marker.pose.position.x = float(trajectory_points[-1][0])
            end_marker.pose.position.y = float(trajectory_points[-1][1])
            end_marker.pose.position.z = 0.15
            end_marker.pose.orientation.w = 1.0
            end_marker.scale.x = 0.2
            end_marker.scale.y = 0.2
            end_marker.scale.z = 0.2
            end_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Bright red for end
            end_marker.lifetime.sec = 0  # Persistent
            
            marker_array.markers.append(end_marker)
            
            # Add waypoint markers every 50 points for better performance
            for i in range(0, len(trajectory_points), 50):
                waypoint_marker = Marker()
                waypoint_marker.header.frame_id = 'world'
                waypoint_marker.header.stamp = self.get_clock().now().to_msg()
                waypoint_marker.ns = f'trajectory_tb{robot_id}'
                waypoint_marker.id = 10 + i  # Unique ID for each waypoint
                waypoint_marker.type = Marker.CYLINDER
                waypoint_marker.action = Marker.ADD
                waypoint_marker.pose.position.x = float(trajectory_points[i][0])
                waypoint_marker.pose.position.y = float(trajectory_points[i][1])
                waypoint_marker.pose.position.z = 0.05
                waypoint_marker.pose.orientation.w = 1.0
                waypoint_marker.scale.x = 0.08
                waypoint_marker.scale.y = 0.08
                waypoint_marker.scale.z = 0.1
                waypoint_marker.color = self.colors[robot_id % len(self.colors)]
                waypoint_marker.color.a = 0.8  # Semi-transparent
                waypoint_marker.lifetime.sec = 0  # Persistent
                
                marker_array.markers.append(waypoint_marker)
        
        return marker_array
    
    def republish_trajectories(self):
        """Republish trajectory markers (useful for Gazebo)"""
        self.load_and_publish_trajectories()


def main(args=None):
    rclpy.init(args=args)
    
    trajectory_visualizer = TrajectoryVisualizer()
    
    try:
        rclpy.spin(trajectory_visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        trajectory_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
