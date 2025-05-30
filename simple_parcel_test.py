#!/usr/bin/env python3
"""
Simple test to verify parcel subscription
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import time

class SimpleParcelTest(Node):
    def __init__(self):
        super().__init__('simple_parcel_test')
        self.parcel_pose = None
        
        # Subscribe to parcel0/pose
        self.parcel_sub = self.create_subscription(
            PoseStamped,
            '/parcel0/pose',
            self.parcel_callback,
            10
        )
        self.get_logger().info('Subscribed to /parcel0/pose')
        
    def parcel_callback(self, msg):
        self.parcel_pose = msg
        self.get_logger().info(f'Received parcel0 pose: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}')

def main():
    rclpy.init()
    node = SimpleParcelTest()
    
    print("Testing parcel subscription for 5 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        
    if node.parcel_pose:
        print(f"SUCCESS: Received parcel pose: x={node.parcel_pose.pose.position.x:.3f}, y={node.parcel_pose.pose.position.y:.3f}")
    else:
        print("FAILED: No parcel pose received")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
