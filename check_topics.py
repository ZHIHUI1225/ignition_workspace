#!/usr/bin/env python3
"""
Quick script to check if the required topics are being published
"""

import rclpy
from rclpy.node import Node
import time

def check_topics():
    """Check if the required topics are available"""
    rclpy.init()
    node = Node('topic_checker')
    
    # Wait a moment for discovery
    time.sleep(2.0)
    
    # Get list of available topics
    topics_and_types = node.get_topic_names_and_types()
    topic_names = [topic for topic, types in topics_and_types]
    
    print("=== Available Topics ===")
    for topic in sorted(topic_names):
        print(f"  {topic}")
    
    print("\n=== Checking Required Topics ===")
    
    # Check robot topics
    robot_topics = ['/turtlebot0/odom_map', '/turtlebot1/odom_map']
    for topic in robot_topics:
        if topic in topic_names:
            print(f"✓ {topic} - Available")
        else:
            print(f"❌ {topic} - NOT FOUND")
    
    # Check parcel topics
    parcel_topics = ['/parcel0/pose', '/parcel1/pose', '/parcel2/pose']
    for topic in parcel_topics:
        if topic in topic_names:
            print(f"✓ {topic} - Available")
        else:
            print(f"❌ {topic} - NOT FOUND")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    check_topics()
