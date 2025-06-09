#!/usr/bin/env python3
"""
Test script to verify the PushObject behavior timer frequency
"""
import rclpy
from rclpy.node import Node
import time
import threading
from rclpy.executors import MultiThreadedExecutor
from behaviour_tree.behaviors.manipulation_behaviors import PushObject

class TimerTestNode(Node):
    def __init__(self):
        super().__init__('timer_test_node')
        
        # Create PushObject behavior
        self.push_behavior = PushObject(name="TestPush", robot_namespace="turtlebot0")
        
        # Setup the behavior with this node
        self.push_behavior.setup(node=self)
        
        # Initialize to start the timer
        self.push_behavior.initialise()
        
        self.get_logger().info("Timer test started - monitoring for 10 seconds...")
        
        # Create a timer to stop the test after 10 seconds
        self.test_timer = self.create_timer(10.0, self.stop_test)
        self.test_running = True
        
    def stop_test(self):
        """Stop the test after 10 seconds"""
        self.get_logger().info("Test completed - stopping...")
        self.test_running = False
        
        # Stop the push behavior
        self.push_behavior.pushing_active = False
        
        # Cancel timers
        if self.push_behavior.control_timer:
            self.push_behavior.control_timer.cancel()
        self.test_timer.cancel()
        
        rclpy.shutdown()

def main():
    rclpy.init()
    
    # Create test node
    test_node = TimerTestNode()
    
    # Use MultiThreadedExecutor like the behavior tree
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(test_node)
    
    try:
        # Run for 10 seconds
        start_time = time.time()
        while rclpy.ok() and test_node.test_running and (time.time() - start_time) < 11:
            executor.spin_once(timeout_sec=0.1)
            
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        executor.shutdown()
        test_node.destroy_node()

if __name__ == '__main__':
    main()
