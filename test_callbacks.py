#!/usr/bin/env python3
"""
Simple test script to verify callback functionality in ApproachObject behavior.
This test will help debug the missing pose data issue.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import py_trees
import py_trees_ros.trees
import threading
import time

# Import the fixed ApproachObject behavior
from src.behaviour_tree.behaviour_tree.behaviors.movement_behaviors import ApproachObject

def test_callback_functionality():
    """Test if ApproachObject behavior receives callbacks properly"""
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Create shared ROS node (this simulates the main behavior tree node)
        shared_node = rclpy.create_node("test_shared_node")
        print(f"Created shared node: {shared_node.get_name()}")
        
        # Create publishers to simulate pose data
        robot_odom_pub = shared_node.create_publisher(
            Odometry, '/turtlebot0/odom_map', 10)
        parcel_pose_pub = shared_node.create_publisher(
            PoseStamped, '/parcel0/pose', 10)
        
        # Create the ApproachObject behavior
        approach_behavior = ApproachObject("TestApproach", "turtlebot0")
        
        # Set up behavior tree with the shared node
        approach_behavior.setup(node=shared_node)
        approach_behavior.initialise()
        
        print("ApproachObject behavior set up successfully")
        
        # Create ROS executor with the shared node
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(shared_node)
        
        # Start ROS executor in background
        def spin_ros():
            executor.spin()
        
        ros_thread = threading.Thread(target=spin_ros)
        ros_thread.daemon = True
        ros_thread.start()
        
        print("ROS executor started")
        
        # Give some time for setup
        time.sleep(1.0)
        
        # Create and publish test pose data
        print("Publishing test pose data...")
        
        # Robot odometry message
        robot_odom = Odometry()
        robot_odom.header.stamp = shared_node.get_clock().now().to_msg()
        robot_odom.header.frame_id = "map"
        robot_odom.pose.pose.position.x = 1.0
        robot_odom.pose.pose.position.y = 2.0
        robot_odom.pose.pose.position.z = 0.0
        robot_odom.pose.pose.orientation.w = 1.0
        
        # Parcel pose message
        parcel_pose = PoseStamped()
        parcel_pose.header.stamp = shared_node.get_clock().now().to_msg()
        parcel_pose.header.frame_id = "map"
        parcel_pose.pose.position.x = 3.0
        parcel_pose.pose.position.y = 4.0
        parcel_pose.pose.position.z = 0.0
        parcel_pose.pose.orientation.w = 1.0
        
        # Publish data repeatedly for 5 seconds
        for i in range(50):  # 5 seconds at 10Hz
            robot_odom_pub.publish(robot_odom)
            parcel_pose_pub.publish(parcel_pose)
            
            # Test the behavior's update method
            status = approach_behavior.update()
            
            print(f"Iteration {i+1}: robot_pose={approach_behavior.robot_pose is not None}, "
                  f"parcel_pose={approach_behavior.parcel_pose is not None}, "
                  f"status={status}")
            
            time.sleep(0.1)  # 10Hz
            
            # Stop if we get pose data
            if approach_behavior.robot_pose is not None and approach_behavior.parcel_pose is not None:
                print("SUCCESS: Both poses received!")
                break
        else:
            print("FAILED: Poses not received after 5 seconds")
        
        # Clean up
        approach_behavior.terminate(py_trees.common.Status.INVALID)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass

if __name__ == "__main__":
    test_callback_functionality()
