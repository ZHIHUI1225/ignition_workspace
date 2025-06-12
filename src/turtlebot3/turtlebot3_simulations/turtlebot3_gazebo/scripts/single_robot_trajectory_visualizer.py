#!/usr/bin/env python3
"""
Single Robot Trajectory Visualizer Node
Loads trajectory points for a single robot from JSON files and visualizes them in Gazebo Ignition
"""

import rclpy
from rclpy.node import Node
import json
import os
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
import math


class SingleRobotTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('single_robot_trajectory_visualizer')
        
        # Parameters
        self.declare_parameter('robot_namespace', 'turtlebot0')
        self.declare_parameter('trajectory_file', '')
        self.declare_parameter('case', 'simple_maze')
        self.declare_parameter('robot_id', 0)
        
        self.robot_namespace = self.get_parameter('robot_namespace').get_parameter_value().string_value
        self.trajectory_file = self.get_parameter('trajectory_file').get_parameter_value().string_value
        self.case = self.get_parameter('case').get_parameter_value().string_value
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().integer_value
        
        self.get_logger().info(f"Single Robot Trajectory Visualizer started for: {self.robot_namespace}")
        self.get_logger().info(f"Trajectory file: {self.trajectory_file}")
        
        # Colors for different robots
        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),  # Red
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),  # Green
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8),  # Blue
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8),  # Yellow
            ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8),  # Magenta
            ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8),  # Cyan
        ]
        self.color = colors[self.robot_id % len(colors)]
        
        # Publishers
        self.path_publisher = self.create_publisher(
            Path, f'/{self.robot_namespace}/trajectory_path', 10)
        
        self.marker_publisher = self.create_publisher(
            MarkerArray, f'/{self.robot_namespace}/trajectory_markers', 10)
        
        # Load and publish trajectory
        self.load_and_publish_trajectory()
        
        # Timer to republish markers periodically (Gazebo sometimes needs this)
        self.timer = self.create_timer(5.0, self.republish_trajectory)
        
        # Store loaded data for republishing
        self.path_msg = None
        self.marker_array = None
        
    def load_and_publish_trajectory(self):
        """Load trajectory data from JSON file and publish as paths and markers"""
        if os.path.exists(self.trajectory_file):
            self.get_logger().info(f"Loading trajectory from: {self.trajectory_file}")
            self.load_trajectory()
        else:
            self.get_logger().warn(f"Trajectory file not found: {self.trajectory_file}")
    
    def load_trajectory(self):
        """Load the robot's trajectory and publish it"""
        try:
            with open(self.trajectory_file, 'r') as f:
                data = json.load(f)
            
            trajectory_points = data['Trajectory']
            self.get_logger().info(f"Loaded {len(trajectory_points)} trajectory points for {self.robot_namespace}")
            
            # Create Path message for RViz-style visualization
            self.path_msg = self.create_path_message(trajectory_points)
            
            # Create Marker messages for Gazebo visualization
            self.marker_array = self.create_trajectory_markers(trajectory_points)
            
            # Publish both
            self.path_publisher.publish(self.path_msg)
            self.marker_publisher.publish(self.marker_array)
            
            self.get_logger().info(f"Published trajectory visualization for {self.robot_namespace}")
            
        except Exception as e:
            self.get_logger().error(f"Error loading trajectory for {self.robot_namespace}: {str(e)}")
    
    def create_path_message(self, trajectory_points):
        """Create a Path message from trajectory points"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.frame_id = 'world'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in trajectory_points:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'world'
            pose_stamped.header.stamp = path_msg.header.stamp
            
            # Position (convert from cm to m if needed)
            pose_stamped.pose.position.x = float(point[0])
            pose_stamped.pose.position.y = float(point[1])
            pose_stamped.pose.position.z = 0.05  # Slightly above ground
            
            # Orientation from yaw angle
            yaw = float(point[2])
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = math.sin(yaw / 2.0)
            pose_stamped.pose.orientation.w = math.cos(yaw / 2.0)
            
            path_msg.poses.append(pose_stamped)
        
        return path_msg
    
    def create_trajectory_markers(self, trajectory_points):
        """Create Marker messages for Gazebo visualization"""
        marker_array = MarkerArray()
        
        # Create line strip marker for the path
        line_marker = Marker()
        line_marker.header.frame_id = 'world'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = f'{self.robot_namespace}_trajectory'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.02  # Line width
        line_marker.color = self.color
        
        # Add points to line marker
        for point in trajectory_points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.05  # Slightly above ground
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # Create arrow markers for waypoints (every 10th point to avoid clutter)
        for i, point in enumerate(trajectory_points[::10]):  # Every 10th point
            arrow_marker = Marker()
            arrow_marker.header.frame_id = 'world'
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = f'{self.robot_namespace}_waypoints'
            arrow_marker.id = i + 1
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            
            # Position
            arrow_marker.pose.position.x = float(point[0])
            arrow_marker.pose.position.y = float(point[1])
            arrow_marker.pose.position.z = 0.1
            
            # Orientation from yaw
            yaw = float(point[2])
            arrow_marker.pose.orientation.x = 0.0
            arrow_marker.pose.orientation.y = 0.0
            arrow_marker.pose.orientation.z = math.sin(yaw / 2.0)
            arrow_marker.pose.orientation.w = math.cos(yaw / 2.0)
            
            # Scale
            arrow_marker.scale.x = 0.1  # Length
            arrow_marker.scale.y = 0.02  # Width
            arrow_marker.scale.z = 0.02  # Height
            
            # Color (slightly transparent)
            color = ColorRGBA()
            color.r = self.color.r
            color.g = self.color.g
            color.b = self.color.b
            color.a = 0.6
            arrow_marker.color = color
            
            marker_array.markers.append(arrow_marker)
        
        return marker_array
    
    def republish_trajectory(self):
        """Republish trajectory markers periodically"""
        if self.path_msg is not None and self.marker_array is not None:
            # Update timestamps
            current_time = self.get_clock().now().to_msg()
            self.path_msg.header.stamp = current_time
            
            for marker in self.marker_array.markers:
                marker.header.stamp = current_time
            
            # Republish
            self.path_publisher.publish(self.path_msg)
            self.marker_publisher.publish(self.marker_array)


def main(args=None):
    rclpy.init(args=args)
    
    visualizer = SingleRobotTrajectoryVisualizer()
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
