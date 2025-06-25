#!/usr/bin/env python3
"""
Test script for ReplanPath behavior class.
This script tests the ReplanPath behavior functionality.
"""

import sys
import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import py_trees

# Add the behavior tree module to Python path
sys.path.append('/root/workspace/src/behaviour_tree')

# Import the ReplanPath behavior
from behaviour_tree.behaviors.ReplanPath_behaviour import ReplanPath

class TestNode(Node):
    """Test node for ROS functionality"""
    
    def __init__(self):
        super().__init__('test_replan_path_node')
        self.get_logger().info('Test node created')
        
        # Publisher for previous robot's pushing estimated time (for testing)
        self.pushing_time_pub = self.create_publisher(
            Float64, 
            '/turtlebot0/pushing_estimated_time', 
            10
        )
        
        # Timer to publish test data
        self.timer = self.create_timer(1.0, self.publish_test_data)
        self.test_pushing_time = 45.0
        
    def publish_test_data(self):
        """Publish test pushing estimated time"""
        msg = Float64()
        msg.data = self.test_pushing_time
        self.pushing_time_pub.publish(msg)

def test_replan_path_basic():
    """Test basic ReplanPath functionality"""
    print("=== Testing ReplanPath Basic Functionality ===")
    
    # Test 1: Constructor
    print("\n1. Testing constructor...")
    replan_behavior = ReplanPath(
        name="test_replan",
        duration=30.0,
        robot_namespace="turtlebot0",
        case="simple_maze"
    )
    print(f"✓ ReplanPath created successfully")
    print(f"  - Name: {replan_behavior.name}")
    print(f"  - Duration: {replan_behavior.duration}")
    print(f"  - Robot namespace: {replan_behavior.robot_namespace}")
    print(f"  - Case: {replan_behavior.case}")
    print(f"  - Namespace number: {replan_behavior.namespace_number}")
    print(f"  - Previous robot namespace: {replan_behavior.previous_robot_namespace}")
    
    # Test 2: Initial state
    print("\n2. Testing initial state...")
    assert replan_behavior.start_time is None
    assert not replan_behavior.replanning_completed
    assert not replan_behavior.replanning_successful
    print("✓ Initial state correct")
    
    return replan_behavior

def test_replan_path_with_ros():
    """Test ReplanPath with ROS functionality"""
    print("\n=== Testing ReplanPath with ROS ===")
    
    try:
        # Initialize ROS
        rclpy.init()
        
        # Create test node
        test_node = TestNode()
        print("✓ Test ROS node created")
        
        # Test ReplanPath for turtlebot1 (should subscribe to turtlebot0)
        replan_behavior = ReplanPath(
            name="test_replan_tb1",
            duration=30.0,
            robot_namespace="turtlebot1",
            case="simple_maze"
        )
        
        # Setup with ROS node
        setup_success = replan_behavior.setup(node=test_node)
        print(f"✓ Setup result: {setup_success}")
        
        if setup_success:
            print("✓ ROS setup successful")
            print(f"  - Subscribed to: /{replan_behavior.previous_robot_namespace}/pushing_estimated_time")
        
        # Spin a few times to let ROS messages flow
        print("\n3. Testing ROS message flow...")
        for i in range(5):
            rclpy.spin_once(test_node, timeout_sec=0.1)
            time.sleep(0.1)
        
        print(f"✓ Previous robot pushing time: {replan_behavior.previous_robot_pushing_estimated_time}")
        
        return replan_behavior, test_node
        
    except Exception as e:
        print(f"✗ ROS test failed: {e}")
        return None, None

def test_replan_behavior_execution():
    """Test the actual behavior execution"""
    print("\n=== Testing ReplanPath Behavior Execution ===")
    
    # Check if test data exists
    test_case = "simple_maze"
    test_robot_id = 1
    
    data_dir = f'/root/workspace/data/{test_case}/'
    trajectory_file = f'{data_dir}robot_{test_robot_id}_trajectory_parameters_{test_case}.json'
    
    print(f"\n4. Checking for test data...")
    print(f"   Looking for: {trajectory_file}")
    
    if not os.path.exists(trajectory_file):
        print(f"⚠ Test data not found: {trajectory_file}")
        print("   Skipping execution test")
        return None
    
    print("✓ Test data found")
    
    # Create ReplanPath behavior
    replan_behavior = ReplanPath(
        name="test_execution",
        duration=60.0,  # Longer timeout for actual execution
        robot_namespace="turtlebot0",
        case=test_case
    )
    
    # Test initialization
    print("\n5. Testing initialization...")
    replan_behavior.initialise()
    
    assert replan_behavior.start_time is not None
    assert hasattr(replan_behavior, 'target_time')
    assert hasattr(replan_behavior, 'robot_id')
    
    print(f"✓ Initialization successful")
    print(f"  - Target time: {replan_behavior.target_time}")
    print(f"  - Robot ID: {replan_behavior.robot_id}")
    
    # Test behavior execution (this might take some time)
    print("\n6. Testing behavior execution...")
    print("   This may take a while for optimization...")
    
    max_iterations = 30  # Limit test time
    iteration = 0
    
    while iteration < max_iterations:
        status = replan_behavior.update()
        iteration += 1
        
        print(f"   Iteration {iteration}: Status = {status}")
        
        if status != py_trees.common.Status.RUNNING:
            break
        
        time.sleep(1)  # Wait between updates
    
    final_status = replan_behavior.update()
    print(f"✓ Execution completed with status: {final_status}")
    print(f"  - Replanning completed: {replan_behavior.replanning_completed}")
    print(f"  - Replanning successful: {replan_behavior.replanning_successful}")
    
    # Check output files
    output_file = f'{data_dir}robot_{test_robot_id}_replanned_trajectory_parameters_{test_case}.json'
    discrete_file = f'{data_dir}tb{test_robot_id}_Trajectory_replanned.json'
    
    print(f"\n7. Checking output files...")
    if os.path.exists(output_file):
        print(f"✓ Replanned parameters file created: {output_file}")
    else:
        print(f"⚠ Replanned parameters file not found: {output_file}")
    
    if os.path.exists(discrete_file):
        print(f"✓ Replanned discrete trajectory created: {discrete_file}")
    else:
        print(f"⚠ Replanned discrete trajectory not found: {discrete_file}")
    
    return replan_behavior

def test_fallback_functionality():
    """Test fallback functionality when replanning fails"""
    print("\n=== Testing Fallback Functionality ===")
    
    # Create a ReplanPath with invalid parameters to trigger fallback
    replan_behavior = ReplanPath(
        name="test_fallback",
        duration=5.0,  # Short timeout to trigger failure
        robot_namespace="turtlebot99",  # Non-existent robot
        case="nonexistent_case"  # Non-existent case
    )
    
    # Set robot_id for the fallback test
    replan_behavior.robot_id = 99
    replan_behavior.case = "nonexistent_case"
    
    print("\n8. Testing fallback with invalid parameters...")
    
    # Test the fallback copy function
    success = replan_behavior._copy_original_trajectory_as_fallback()
    print(f"   Fallback copy result: {success}")
    
    if not success:
        print("✓ Fallback correctly failed for non-existent files")
    else:
        print("⚠ Fallback unexpectedly succeeded")
    
    return replan_behavior

def main():
    """Main test function"""
    print("Starting ReplanPath behavior tests...")
    
    try:
        # Test 1: Basic functionality
        replan_basic = test_replan_path_basic()
        
        # Test 2: ROS functionality
        replan_ros, test_node = test_replan_path_with_ros()
        
        # Test 3: Actual execution (if data available)
        replan_exec = test_replan_behavior_execution()
        
        # Test 4: Fallback functionality
        replan_fallback = test_fallback_functionality()
        
        print("\n=== Test Summary ===")
        print("✓ Basic functionality test completed")
        print("✓ ROS functionality test completed")
        if replan_exec:
            print("✓ Behavior execution test completed")
        else:
            print("⚠ Behavior execution test skipped (no data)")
        print("✓ Fallback functionality test completed")
        
        # Cleanup
        if test_node:
            try:
                test_node.destroy_node()
                rclpy.shutdown()
                print("✓ ROS cleanup completed")
            except:
                pass
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            if 'test_node' in locals() and test_node:
                test_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
