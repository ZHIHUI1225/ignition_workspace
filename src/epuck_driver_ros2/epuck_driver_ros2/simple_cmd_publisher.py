#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select
import threading


class SimpleCmdPublisher(Node):
    """Simple node to publish constant cmd_vel commands with keyboard control."""
    
    def __init__(self):
        super().__init__('simple_cmd_publisher')
        
        # Declare parameters
        self.declare_parameter('linear_x', 0.0)
        self.declare_parameter('linear_y', 0.0) 
        self.declare_parameter('linear_z', 0.0)
        self.declare_parameter('angular_x', 0.0)
        self.declare_parameter('angular_y', 0.0)
        self.declare_parameter('angular_z', 0.0)
        self.declare_parameter('topic_name', '/cmd_vel')
        self.declare_parameter('publish_rate', 10.0)  # Hz
        
        # Get parameters
        self.linear_x = self.get_parameter('linear_x').get_parameter_value().double_value
        self.linear_y = self.get_parameter('linear_y').get_parameter_value().double_value
        self.linear_z = self.get_parameter('linear_z').get_parameter_value().double_value
        self.angular_x = self.get_parameter('angular_x').get_parameter_value().double_value
        self.angular_y = self.get_parameter('angular_y').get_parameter_value().double_value
        self.angular_z = self.get_parameter('angular_z').get_parameter_value().double_value
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Create publisher
        self.cmd_vel_pub = self.create_publisher(Twist, self.topic_name, 10)
        
        # Create timer
        timer_period = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(timer_period, self.publish_cmd_vel)
        
        # Store original and stop commands
        self.original_cmd = Twist()
        self.original_cmd.linear.x = self.linear_x
        self.original_cmd.linear.y = self.linear_y
        self.original_cmd.linear.z = self.linear_z
        self.original_cmd.angular.x = self.angular_x
        self.original_cmd.angular.y = self.angular_y
        self.original_cmd.angular.z = self.angular_z
        
        self.stop_cmd = Twist()  # All zeros by default
        
        # Control state
        self.is_stopped = False
        self.current_cmd = self.original_cmd
        
        # Keyboard thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        self.get_logger().info(f'Publishing cmd_vel to {self.topic_name}')
        self.get_logger().info(f'Linear: ({self.linear_x}, {self.linear_y}, {self.linear_z})')
        self.get_logger().info(f'Angular: ({self.angular_x}, {self.angular_y}, {self.angular_z})')
        self.get_logger().info("Press 's' to STOP, 'r' to RESUME, 'q' to quit")
    
    def get_key(self):
        """Get a single key press from terminal"""
        try:
            settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                key = ''
                if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                    key = sys.stdin.read(1)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            return key
        except:
            return ''
    
    def keyboard_listener(self):
        """Listen for keyboard input in a separate thread"""
        while rclpy.ok():
            key = self.get_key()
            
            if key == 's' or key == 'S':
                self.is_stopped = True
                self.current_cmd = self.stop_cmd
                self.get_logger().info("🛑 STOP - Publishing zero velocity")
                
            elif key == 'r' or key == 'R':
                self.is_stopped = False
                self.current_cmd = self.original_cmd
                self.get_logger().info("✅ RESUME - Publishing original velocity")
                
            elif key == 'q' or key == 'Q':
                self.get_logger().info("Quitting...")
                rclpy.shutdown()
                break
            elif key == '\x03':  # Ctrl+C
                break
    
    def publish_cmd_vel(self):
        """Publish the current cmd_vel message."""
        self.cmd_vel_pub.publish(self.current_cmd)


def main(args=None):
    rclpy.init(args=args)
    
    cmd_publisher = SimpleCmdPublisher()
    
    try:
        rclpy.spin(cmd_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        cmd_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
