#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
import math

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_test_publisher')
        
        # Declare parameters with default values
        self.declare_parameter(
            'linear_x', 0.1,
            ParameterDescriptor(description='Linear velocity in X direction (m/s)')
        )
        self.declare_parameter(
            'linear_y', 0.0,
            ParameterDescriptor(description='Linear velocity in Y direction (m/s)')
        )
        self.declare_parameter(
            'angular_z', 0.0,
            ParameterDescriptor(description='Angular velocity around Z axis (rad/s)')
        )
        self.declare_parameter(
            'publish_rate', 10.0,
            ParameterDescriptor(description='Publishing rate in Hz')
        )
        
        # Get parameters
        self.linear_x = self.get_parameter('linear_x').get_parameter_value().double_value
        self.linear_y = self.get_parameter('linear_y').get_parameter_value().double_value
        self.angular_z = self.get_parameter('angular_z').get_parameter_value().double_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Create publisher
        self.cmd_vel_publisher = self.create_publisher(Twist, '/robot0/cmd_vel', 10)
        
        # Create timer for publishing
        timer_period = 1.0 / publish_rate  # Convert Hz to seconds
        self.timer = self.create_timer(timer_period, self.publish_cmd_vel)
        
        # Create the twist message
        self.twist_msg = Twist()
        self.twist_msg.linear.x = self.linear_x
        self.twist_msg.linear.y = self.linear_y
        self.twist_msg.linear.z = 0.0
        self.twist_msg.angular.x = 0.0
        self.twist_msg.angular.y = 0.0
        self.twist_msg.angular.z = self.angular_z
        
        # Log startup information
        self.get_logger().info(f"CMD_VEL Test Publisher Started")
        self.get_logger().info(f"Publishing to /robot0/cmd_vel at {publish_rate} Hz")
        self.get_logger().info(f"Linear velocity: x={self.linear_x:.3f} m/s, y={self.linear_y:.3f} m/s")
        self.get_logger().info(f"Angular velocity: z={self.angular_z:.3f} rad/s")
        
        # Counter for periodic status updates
        self.publish_count = 0
        self.status_interval = int(publish_rate * 5)  # Log status every 5 seconds
        
    def publish_cmd_vel(self):
        """Publish the constant cmd_vel message"""
        # Update timestamp
        self.twist_msg.linear.x = self.linear_x
        self.twist_msg.linear.y = self.linear_y
        self.twist_msg.angular.z = self.angular_z
        
        # Publish the message
        self.cmd_vel_publisher.publish(self.twist_msg)
        
        self.publish_count += 1
        
        # Periodic status logging
        if self.publish_count % self.status_interval == 0:
            self.get_logger().info(f"Published {self.publish_count} cmd_vel messages")
            self.get_logger().info(f"Commanded velocities: linear_x={self.linear_x:.3f} m/s, angular_z={self.angular_z:.3f} rad/s")
    
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info(f"Shutting down CMD_VEL publisher. Total messages published: {self.publish_count}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        cmd_vel_publisher = CmdVelPublisher()
        
        # Log instructions for user
        cmd_vel_publisher.get_logger().info("=" * 60)
        cmd_vel_publisher.get_logger().info("CMD_VEL Test Publisher Instructions:")
        cmd_vel_publisher.get_logger().info("- Publishing constant velocity commands to /robot0/cmd_vel")
        cmd_vel_publisher.get_logger().info("- Use RViz to visualize robot movement")
        cmd_vel_publisher.get_logger().info("- Press Ctrl+C to stop")
        cmd_vel_publisher.get_logger().info("=" * 60)
        
        rclpy.spin(cmd_vel_publisher)
        
    except KeyboardInterrupt:
        pass
    finally:
        if 'cmd_vel_publisher' in locals():
            cmd_vel_publisher.destroy_node()
        rclpy.shutdown()
        print("CMD_VEL Test Publisher shutdown complete")

if __name__ == '__main__':
    main()
