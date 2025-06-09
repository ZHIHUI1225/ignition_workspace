#!/usr/bin/env python3

"""
Test the complete behavior tree system with MPC control loop timing.
This test validates that the PushObject behavior's timer runs at 10Hz even 
when executed within the full behavior tree context.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import time
import threading
import signal
import sys
import py_trees
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray

# Import behavior tree modules
sys.path.append('/root/workspace/src/behaviour_tree')
from behaviour_tree.behaviors.manipulation_behaviors import PushObject
from behaviour_tree.behaviors.basic_behaviors import WaitForPush


class TestNode(Node):
    """Test node to simulate the behavior tree environment"""
    
    def __init__(self):
        super().__init__('test_full_system_node')
        
        # Create publishers for simulation
        self.parcel_pub = self.create_publisher(PoseStamped, '/parcel0/pose', 10)
        self.odom_pub = self.create_publisher(PoseStamped, '/turtlebot0/odom', 10)
        
        # Timer frequencies tracking
        self.timer_callbacks = []
        self.timer_lock = threading.Lock()
        
        # Create test timer to monitor frequency
        self.monitoring_timer = self.create_timer(1.0, self.monitor_frequency)
        self.start_time = time.time()
        
        print("[TestNode] Initialized test node for full system validation")

    def monitor_frequency(self):
        """Monitor and report timer frequency"""
        with self.timer_lock:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Filter callbacks from last second
            recent_callbacks = [t for t in self.timer_callbacks if (current_time - t) <= 1.0]
            frequency = len(recent_callbacks)
            
            if elapsed >= 2.0:  # Start reporting after 2 seconds
                print(f"[TestNode] MPC Timer Frequency: {frequency:.1f} Hz (target: 10 Hz)")
                
                if frequency >= 9.0 and frequency <= 11.0:
                    print(f"[TestNode] ✓ Timer frequency is within acceptable range!")
                else:
                    print(f"[TestNode] ✗ Timer frequency is outside acceptable range (9-11 Hz)")

    def record_timer_callback(self):
        """Record when a timer callback occurs"""
        with self.timer_lock:
            self.timer_callbacks.append(time.time())
            # Keep only last 100 callbacks to prevent memory growth
            if len(self.timer_callbacks) > 100:
                self.timer_callbacks = self.timer_callbacks[-100:]


def create_test_behavior_tree(test_node):
    """Create a simple behavior tree for testing"""
    
    # Create blackboard and set up robot namespace
    blackboard = py_trees.blackboard.Client(name="TestClient")
    blackboard.register_key(key='turtlebot0/current_parcel_index', access=py_trees.common.Access.WRITE)
    blackboard.register_key(key='turtlebot0/parcel_poses', access=py_trees.common.Access.WRITE)
    blackboard.turtlebot0.current_parcel_index = 0
    blackboard.turtlebot0.parcel_poses = [[1.0, 2.0, 0.0]]  # Test parcel position
    
    # Create PushObject behavior with test node
    push_behavior = PushObject(
        name="TestPushObject",
        robot_namespace='turtlebot0'
    )
    
    # Set up the behavior with the test node
    push_behavior.setup(node=test_node)
    
    # Set the blackboard manually
    push_behavior.blackboard = blackboard
    
    # Hook into timer callback to monitor frequency
    original_control_timer_callback = push_behavior.control_timer_callback
    def monitored_control_timer_callback():
        test_node.record_timer_callback()
        return original_control_timer_callback()
    push_behavior.control_timer_callback = monitored_control_timer_callback
    
    # Create WaitForPush behavior (this was causing issues before)
    wait_behavior = WaitForPush(
        name="TestWaitForPush",
        robot_namespace='turtlebot0',
        duration=10.0,
        distance_threshold=0.08
    )
    
    # Setup the wait behavior with the test node
    wait_behavior.setup(node=test_node, blackboard=blackboard)
    
    # Create simple sequence behavior tree
    root = py_trees.composites.Sequence(
        name="TestSequence",
        memory=False,
        children=[wait_behavior, push_behavior]
    )
    
    return py_trees.trees.BehaviourTree(root)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n[TestFullSystem] Shutting down...")
    rclpy.shutdown()
    sys.exit(0)


def main():
    """Main test function"""
    print("[TestFullSystem] Starting full system test with behavior tree execution...")
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create test node
        test_node = TestNode()
        
        # Create MultiThreadedExecutor (same as in the real system)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(test_node)
        
        # Create behavior tree
        behavior_tree = create_test_behavior_tree(test_node)
        
        print("[TestFullSystem] Starting behavior tree execution...")
        print("[TestFullSystem] This will test if MPC control runs at 10Hz within behavior tree context")
        print("[TestFullSystem] Press Ctrl+C to stop")
        
        # Run the behavior tree in a separate thread
        def run_behavior_tree():
            # Tick the behavior tree at 1Hz (typical behavior tree frequency)
            while rclpy.ok():
                behavior_tree.tick()
                time.sleep(1.0)  # 1Hz behavior tree ticking
        
        bt_thread = threading.Thread(target=run_behavior_tree, daemon=True)
        bt_thread.start()
        
        # Run the executor (this will handle all ROS callbacks including timers)
        print("[TestFullSystem] Running MultiThreadedExecutor...")
        executor.spin()
        
    except KeyboardInterrupt:
        print("\n[TestFullSystem] Test interrupted by user")
    except Exception as e:
        print(f"[TestFullSystem] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'executor' in locals():
            executor.shutdown()
        if 'test_node' in locals():
            test_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[TestFullSystem] Test completed")


if __name__ == '__main__':
    main()
