#!/usr/bin/env python3
"""
Test script to verify ROS subscriptions are working correctly
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import time
import threading

class TestSubscriber(Node):
    def __init__(self):
        super().__init__('test_subscriber')
        
        # Storage for received data
        self.robot_pose = None
        self.parcel_pose = None
        self.robot_callback_count = 0
        self.parcel_callback_count = 0
        
        # Create subscriptions
        self.robot_sub = self.create_subscription(
            Odometry, '/turtlebot0/odom_map', self.robot_callback, 10)
        
        self.parcel_sub = self.create_subscription(
            PoseStamped, '/parcel0/pose', self.parcel_callback, 10)
        
        print("TestSubscriber initialized with subscriptions")
        print("Robot subscription:", self.robot_sub)
        print("Parcel subscription:", self.parcel_sub)
        
    def robot_callback(self, msg):
        self.robot_pose = msg.pose.pose
        self.robot_callback_count += 1
        if self.robot_callback_count <= 3:
            print(f"Robot callback #{self.robot_callback_count}: ({self.robot_pose.position.x:.3f}, {self.robot_pose.position.y:.3f})")
    
    def parcel_callback(self, msg):
        self.parcel_pose = msg.pose
        self.parcel_callback_count += 1
        if self.parcel_callback_count <= 3:
            print(f"Parcel callback #{self.parcel_callback_count}: ({self.parcel_pose.position.x:.3f}, {self.parcel_pose.position.y:.3f})")

def main():
    rclpy.init()
    
    test_node = TestSubscriber()
    
    print("Starting test - waiting for callbacks...")
    
    # Spin for 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10.0:
        rclpy.spin_once(test_node, timeout_sec=0.1)
        
        # Check status every 2 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 2 == 0 and int(elapsed * 10) % 20 == 0:  # Every 2 seconds
            robot_received = test_node.robot_pose is not None
            parcel_received = test_node.parcel_pose is not None
            print(f"After {elapsed:.1f}s - Robot: {robot_received} ({test_node.robot_callback_count} msgs), Parcel: {parcel_received} ({test_node.parcel_callback_count} msgs)")
    
    print("Test completed")
    print(f"Final status - Robot: {test_node.robot_pose is not None}, Parcel: {test_node.parcel_pose is not None}")
    
    test_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
