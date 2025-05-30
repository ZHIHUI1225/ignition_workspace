#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Private Snapshot Streams Demo

This example demonstrates how to create and use private snapshot streams for 
py_trees_ros behavior trees, based on Tutorial 4 from the py_trees_ros tutorials.

Key Features:
1. Default snapshot stream configuration via ROS parameters
2. Private snapshot streams accessible via /tree/snapshot_streams namespace
3. Tree introspection with py-trees-tree-watcher
4. Blackboard activity and data streaming

Usage:
    # Terminal 1: Run this demo
    python3 snapshot_streams_demo.py
    
    # Terminal 2: Connect to private snapshot stream
    py-trees-tree-watcher --namespace=/tree/snapshot_streams
    
    # Terminal 3: Connect to default snapshot stream  
    py-trees-tree-watcher -a -s -b /tree/snapshots
"""

import py_trees
import py_trees_ros.trees
import py_trees_ros.visitors
import rclpy
import rclpy.executors
import threading
import time
import operator
from rclpy.parameter import Parameter


class DemoRobotBehaviour(py_trees.behaviour.Behaviour):
    """A simple behavior that demonstrates robot-like activity."""
    
    def __init__(self, name: str, activity: str = "working"):
        super().__init__(name)
        self.activity = activity
        self.counter = 0
        
    def setup(self, **kwargs):
        """Setup behavior with ROS node."""
        try:
            self.node = kwargs['node']
            self.logger = self.node.get_logger()
        except KeyError:
            raise KeyError("'node' not found in kwargs")
            
    def initialise(self):
        """Initialize the behavior."""
        self.counter = 0
        self.feedback_message = f"Starting {self.activity}"
        
    def update(self):
        """Update the behavior state."""
        self.counter += 1
        self.feedback_message = f"{self.activity} step {self.counter}"
        
        if self.counter < 5:
            return py_trees.common.Status.RUNNING
        else:
            self.feedback_message = f"Completed {self.activity}"
            return py_trees.common.Status.SUCCESS


class BatteryCheckBehaviour(py_trees.behaviour.Behaviour):
    """Behavior that checks battery status from blackboard."""
    
    def __init__(self, name: str, low_threshold: float = 30.0):
        super().__init__(name)
        self.low_threshold = low_threshold
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="battery_percentage", 
            access=py_trees.common.Access.READ
        )
        
    def update(self):
        """Check battery status."""
        battery_level = getattr(self.blackboard, "battery_percentage", 100.0)
        
        if battery_level < self.low_threshold:
            self.feedback_message = f"Battery low: {battery_level}%"
            return py_trees.common.Status.FAILURE
        else:
            self.feedback_message = f"Battery OK: {battery_level}%"
            return py_trees.common.Status.SUCCESS


class BatterySimulator(py_trees.behaviour.Behaviour):
    """Simulates battery drainage for demo purposes."""
    
    def __init__(self, name: str, initial_level: float = 100.0):
        super().__init__(name)
        self.battery_level = initial_level
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="battery_percentage",
            access=py_trees.common.Access.WRITE
        )
        
    def update(self):
        """Simulate battery drainage."""
        self.battery_level -= 0.5  # Drain 0.5% per tick
        self.battery_level = max(0.0, self.battery_level)
        
        # Update blackboard
        self.blackboard.battery_percentage = self.battery_level
        self.feedback_message = f"Battery: {self.battery_level:.1f}%"
        
        return py_trees.common.Status.SUCCESS


def create_demo_tree() -> py_trees.behaviour.Behaviour:
    """
    Create a demo behavior tree for snapshot stream demonstration.
    
    This tree structure mirrors the patterns from Tutorial 4:
    - Data gathering (battery simulation)
    - Priority tasks (battery emergency)
    - Worker tasks (robot operations)
    """
    
    # Root parallel: data gathering + task execution
    root = py_trees.composites.Parallel(
        name="Demo Robot",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )
    
    # Data gathering branch
    data_gathering = py_trees.composites.Sequence(name="Data Gathering", memory=True)
    battery_sim = BatterySimulator("Battery Simulator")
    
    # Task execution branch  
    tasks = py_trees.composites.Selector(name="Tasks", memory=False)
    
    # Emergency task: battery check
    battery_check = BatteryCheckBehaviour("Battery Check", low_threshold=25.0)
    battery_emergency = py_trees.decorators.FailureIsSuccess(
        name="Battery Emergency Handler",
        child=battery_check
    )
    
    # Worker tasks
    work_sequence = py_trees.composites.Sequence(name="Work Sequence", memory=True)
    
    # Check if work is requested (using a simple success trigger)
    work_trigger = py_trees.behaviours.Success("Trigger Work")
    
    # Actual work behaviors
    scanning = DemoRobotBehaviour("Scanner", "scanning environment")
    navigation = DemoRobotBehaviour("Navigator", "navigating to target")
    manipulation = DemoRobotBehaviour("Manipulator", "grasping object")
    
    # Idle task
    idle = py_trees.behaviours.Running(name="Idle")
    
    # Assemble the tree
    data_gathering.add_child(battery_sim)
    
    work_sequence.add_children([work_trigger, scanning, navigation, manipulation])
    tasks.add_children([battery_emergency, work_sequence, idle])
    
    root.add_children([data_gathering, tasks])
    
    return root


class SnapshotStreamDemo:
    """
    Demo class that shows how to create and manage snapshot streams.
    
    This class demonstrates:
    1. Setting up default snapshot streams via parameters
    2. Creating private snapshot streams accessible to tree watchers
    3. Proper ROS node management for behavior trees
    """
    
    def __init__(self):
        self.tree = None
        self.node = None
        self.executor = None
        
    def setup_ros_node(self):
        """Create and configure the ROS node for the behavior tree."""
        self.node = rclpy.create_node("tree")  # Use "tree" as node name (convention)
        
        self.node.get_logger().info("ROS node configured for snapshot streams")
        
    def create_behavior_tree(self):
        """Create the behavior tree with snapshot stream support."""
        root = create_demo_tree()
        
        # Create behavior tree with the configured node
        self.tree = py_trees_ros.trees.BehaviourTree(root=root)
        
        # Setup the tree - this enables snapshot streams
        self.tree.setup(timeout=15, node=self.node)
        
        self.node.get_logger().info("Behavior tree created and configured")
        
    def print_stream_info(self):
        """Print information about available snapshot streams."""
        node_name = self.node.get_name()
        namespace = self.node.get_namespace()
        
        print("\n" + "="*80)
        print("SNAPSHOT STREAM INFORMATION")
        print("="*80)
        print(f"Node name: {node_name}")
        print(f"Node namespace: {namespace}")
        
    def run_demo(self, duration: int = 30):
        """
        Run the behavior tree demo.
        
        Args:
            duration: How long to run the demo in seconds
        """
        # Setup ROS executor for background processing
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        
        # Start ROS spinning in background thread
        def spin_ros():
            while rclpy.ok():
                self.executor.spin_once(timeout_sec=0.1)
                
        ros_thread = threading.Thread(target=spin_ros)
        ros_thread.daemon = True
        ros_thread.start()
        
        print(f"\nRunning behavior tree demo for {duration} seconds...")
        print("Watch the tree activity using the commands shown above.")
        print("Press Ctrl+C to stop early.\n")
        
        # Main tree execution loop
        start_time = time.time()
        tick_count = 0
        
        try:
            while rclpy.ok() and (time.time() - start_time) < duration:
                # Tick the tree
                self.tree.tick()
                
                # Print status every 20 ticks (roughly every 2 seconds)
                if tick_count % 20 == 0:
                    tip = self.tree.tip()
                    if tip:
                        print(f"[{tick_count:3d}] Tree tip: {tip.name} ({tip.status})")
                    else:
                        print(f"[{tick_count:3d}] Tree has no active tip")
                
                tick_count += 1
                time.sleep(0.1)  # 10Hz execution
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            
        print(f"\nDemo completed after {tick_count} ticks")
        
    def shutdown(self):
        """Clean shutdown of the demo."""
        if self.tree:
            self.tree.shutdown()
        if self.executor:
            self.executor.shutdown()


def main():
    """Main entry point for the snapshot streams demo."""
    rclpy.init()
    
    demo = SnapshotStreamDemo()
    
    try:
        # Setup the demo
        demo.setup_ros_node() 
        demo.create_behavior_tree()
        demo.print_stream_info()
        
        # Print tree structure
        print("\n" + "="*80)
        print("BEHAVIOR TREE STRUCTURE")
        print("="*80)
        print(py_trees.display.ascii_tree(demo.tree.root))
        print("="*80)
        
        # Run the demo
        demo.run_demo(duration=30)
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
