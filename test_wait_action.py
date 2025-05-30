#!/usr/bin/env python3
"""
Test script to verify WaitAction functionality with pose monitoring
"""

import rclpy
from rclpy.node import Node
import sys
import os

# Add the behavior tree module to path
sys.path.append('/root/workspace/src/behaviour_tree')

from behaviour_tree.behaviors.basic_behaviors import WaitAction
import py_trees

def test_wait_action():
    """Test the WaitAction with pose monitoring"""
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        print("Testing WaitAction with pose monitoring...")
        print("="*50)
        
        # Create WaitAction instance with robot namespace
        wait_action = WaitAction(
            name="TestWaitAction",
            duration=10.0,  # 10 second timeout
            robot_namespace="tb0",
            distance_threshold=0.1  # 10cm threshold
        )
        
        print(f"Created WaitAction for robot: {wait_action.robot_namespace}")
        print(f"Monitoring relay point: Relaypoint{wait_action.relay_number}")
        print(f"Distance threshold: {wait_action.distance_threshold}m")
        print("="*50)
        
        # Initialize the action
        wait_action.initialise()
        
        # Run for a few iterations to show monitoring
        for i in range(3):
            print(f"\nIteration {i+1}:")
            status = wait_action.update()
            print(f"Status: {status}")
            
            # Print current pose information
            if wait_action.robot_pose:
                print(f"Robot pose: x={wait_action.robot_pose.position.x:.3f}, y={wait_action.robot_pose.position.y:.3f}")
            else:
                print("Robot pose: Not available")
                
            if wait_action.relay_pose:
                print(f"Relay pose: x={wait_action.relay_pose.pose.position.x:.3f}, y={wait_action.relay_pose.pose.position.y:.3f}")
            else:
                print("Relay pose: Not available")
                
            if wait_action.parcel_pose:
                print(f"Parcel pose: x={wait_action.parcel_pose.pose.position.x:.3f}, y={wait_action.parcel_pose.pose.position.y:.3f}")
            else:
                print(f"Parcel pose: Not available (current index: {wait_action.current_parcel_index})")
            
            # Check if completed
            if status == py_trees.common.Status.SUCCESS:
                print("SUCCESS: Parcel is within range of relay point!")
                break
            elif status == py_trees.common.Status.FAILURE:
                print("FAILURE: Timeout reached without proximity condition met")
                break
                
            # Small delay
            rclpy.spin_once(wait_action.node, timeout_sec=1.0)
        
        # Clean up
        wait_action.terminate(py_trees.common.Status.INVALID)
        print("\nTest completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    test_wait_action()
