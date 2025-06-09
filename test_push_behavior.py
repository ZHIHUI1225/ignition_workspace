#!/usr/bin/env python3
"""
Test script to verify PushObject behavior cmd_vel publishing functionality
"""
import sys
import os
sys.path.append('/root/workspace/install/behaviour_tree/lib/python3.10/site-packages')

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from behaviour_tree.behaviors.manipulation_behaviors import PushObject
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import threading
import numpy as np


class TestNode(Node):
    def __init__(self):
        super().__init__('test_push_behavior')
        
        # Declare parameters
        self.declare_parameter('robot_namespace', 'turtlebot0')
        self.declare_parameter('case', 'simple_maze')
        
        # Create cmd_vel listener to verify publishing
        self.cmd_vel_received = False
        self.last_cmd_vel = None
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/turtlebot0/cmd_vel', self.cmd_vel_callback, 10)
        
        # Create fake odometry publisher
        self.odom_pub = self.create_publisher(
            Odometry, '/turtlebot0/odom_map', 10)
        
        # Create fake pose publishers for parcel and relay
        self.parcel_pub = self.create_publisher(
            PoseStamped, '/parcel0/pose', 10)
        self.relay_pub = self.create_publisher(
            PoseStamped, '/Relaypoint1/pose', 10)
        
        # Create and setup PushObject behavior
        self.push_behavior = PushObject("TestPush", "turtlebot0")
        setup_success = self.push_behavior.setup(node=self)
        
        if setup_success:
            self.get_logger().info("PushObject behavior setup successful!")
            
            # Start publishing fake data
            self.timer = self.create_timer(0.1, self.publish_fake_data)
            
            # Initialize and start the push behavior
            self.push_behavior.initialise()
            self.get_logger().info("PushObject behavior initialized!")
            
            # Monitor behavior for 10 seconds
            self.test_timer = self.create_timer(1.0, self.check_behavior_status)
            self.test_start_time = time.time()
            
        else:
            self.get_logger().error("Failed to setup PushObject behavior!")
    
    def cmd_vel_callback(self, msg):
        """Callback to monitor cmd_vel messages"""
        self.cmd_vel_received = True
        self.last_cmd_vel = msg
        self.get_logger().info(f"✅ CMD_VEL RECEIVED: v={msg.linear.x:.3f}, ω={msg.angular.z:.3f}")
    
    def publish_fake_data(self):
        """Publish fake odometry and pose data for testing"""
        # Publish fake robot odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'world'
        odom_msg.pose.pose.position.x = 1.0
        odom_msg.pose.pose.position.y = 1.0
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.angular.z = 0.0
        self.odom_pub.publish(odom_msg)
        
        # Publish fake parcel pose (far from relay to keep behavior running)
        parcel_msg = PoseStamped()
        parcel_msg.header.stamp = self.get_clock().now().to_msg()
        parcel_msg.header.frame_id = 'world'
        parcel_msg.pose.position.x = 2.0
        parcel_msg.pose.position.y = 2.0
        parcel_msg.pose.position.z = 0.0
        parcel_msg.pose.orientation.w = 1.0
        self.parcel_pub.publish(parcel_msg)
        
        # Publish fake relay pose
        relay_msg = PoseStamped()
        relay_msg.header.stamp = self.get_clock().now().to_msg()
        relay_msg.header.frame_id = 'world'
        relay_msg.pose.position.x = 5.0  # Far from parcel
        relay_msg.pose.position.y = 5.0
        relay_msg.pose.position.z = 0.0
        relay_msg.pose.orientation.w = 1.0
        self.relay_pub.publish(relay_msg)
    
    def check_behavior_status(self):
        """Check behavior status and cmd_vel publishing"""
        elapsed = time.time() - self.test_start_time
        
        # Update the behavior tree status
        status = self.push_behavior.update()
        
        self.get_logger().info(f"⏱️  Test time: {elapsed:.1f}s")
        self.get_logger().info(f"🔄 Behavior status: {status}")
        self.get_logger().info(f"📡 CMD_VEL received: {self.cmd_vel_received}")
        
        if self.last_cmd_vel:
            self.get_logger().info(f"📊 Last CMD_VEL: v={self.last_cmd_vel.linear.x:.3f}, ω={self.last_cmd_vel.angular.z:.3f}")
        
        # Check publisher status
        if hasattr(self.push_behavior, 'cmd_vel_pub') and self.push_behavior.cmd_vel_pub:
            self.get_logger().info(f"✅ Publisher exists: {self.push_behavior.cmd_vel_pub}")
        else:
            self.get_logger().error("❌ No cmd_vel publisher found!")
        
        # Check if behavior is active
        if hasattr(self.push_behavior, 'pushing_active'):
            self.get_logger().info(f"🏃 Pushing active: {self.push_behavior.pushing_active}")
        
        # Check trajectory loading
        if hasattr(self.push_behavior, 'ref_trajectory') and self.push_behavior.ref_trajectory:
            self.get_logger().info(f"📍 Trajectory loaded: {len(self.push_behavior.ref_trajectory)} points")
        else:
            self.get_logger().error("❌ No reference trajectory loaded!")
        
        if elapsed >= 10.0:
            self.get_logger().info("🏁 Test completed!")
            # Terminate the behavior
            self.push_behavior.terminate('TEST_COMPLETE')
            rclpy.shutdown()


def main():
    rclpy.init()
    
    test_node = TestNode()
    
    # Use multi-threaded executor for proper timer handling
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(test_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
