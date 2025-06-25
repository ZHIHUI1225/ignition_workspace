#!/usr/bin/env python3
"""
Main behavior tree module using modular behavior structure.
Simplified version that imports behaviors from the behaviors package.
"""

import py_trees
import py_trees.console as console
import rclpy
from rclpy.node import Node
import time
import threading
import traceback
import py_trees_ros.trees
import py_trees_ros.blackboard
import py_trees_ros.utilities
import py_trees_ros.visitors
import py_trees.display
from py_trees_ros_interfaces.srv import OpenSnapshotStream, CloseSnapshotStream

# Import all behaviors from the modular structure
from .behaviors import create_root


def setup_snapshot_streams(node, robot_namespace=""):
    """Setup snapshot streams for PyTrees Viewer connection"""
    # Setup default snapshot stream parameters for PyTrees Viewer
    try:
        node.declare_parameter('default_snapshot_stream', True)
        node.declare_parameter('default_snapshot_blackboard_data', True) 
        node.declare_parameter('default_snapshot_blackboard_activity', True)
    except:
        # Parameters may already be declared
        pass
    
    # Build snapshot stream topic name
    if robot_namespace:
        snapshot_topic = f"/{robot_namespace}/tree/snapshot_streams"
    else:
        snapshot_topic = "/tree/snapshot_streams"
    
    node.get_logger().info("Snapshot streams configured for PyTrees Viewer")
    node.get_logger().info(f"Connect PyTrees Viewer to: {snapshot_topic}")
    
    return snapshot_topic


def setup_behaviors_with_node(root, ros_node):
    """
    Recursively traverse the behavior tree and call setup(node=ros_node) on all behaviors.
    This fixes the issue where PyTrees tree.setup() doesn't pass the node to individual behaviors.
    """
    def setup_behavior_recursive(behavior):
        # Call setup on this behavior if it has a setup method
        if hasattr(behavior, 'setup') and callable(getattr(behavior, 'setup')):
            try:
                result = behavior.setup(node=ros_node)
                if result:
                    print(f"[SETUP] ✓ Successfully setup behavior: {behavior.name}")
                else:
                    print(f"[SETUP] ✗ Failed to setup behavior: {behavior.name}")
            except Exception as e:
                print(f"[SETUP] ✗ Error setting up behavior {behavior.name}: {e}")
        
        # Recursively setup children if this is a composite behavior
        if hasattr(behavior, 'children'):
            for child in behavior.children:
                setup_behavior_recursive(child)
    
    print(f"[SETUP] Starting recursive behavior setup with shared ROS node...")
    setup_behavior_recursive(root)
    print(f"[SETUP] Completed recursive behavior setup")


def main():
    """Main function - start behavior tree with PyTrees Viewer support"""
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Create temporary node to get parameters
        temp_node = rclpy.create_node("param_reader")
        
        # Declare and get parameters
        try:
            temp_node.declare_parameter('robot_id', 0)
            temp_node.declare_parameter('robot_namespace', 'turtlebot0')
            robot_id = temp_node.get_parameter('robot_id').get_parameter_value().integer_value
            robot_namespace = temp_node.get_parameter('robot_namespace').get_parameter_value().string_value
        except Exception as e:
            print(f"Warning: Could not get robot parameters: {e}")
            robot_id = 0
            robot_namespace = "turtlebot0"
        
        # Destroy temporary node
        temp_node.destroy_node()
        
        # Create ROS node for executor and snapshot publishing 
        # 🔧 关键修复：为每个机器人创建唯一的节点名称避免冲突
        unique_node_name = f"tree_{robot_id}"
        ros_node = rclpy.create_node(unique_node_name)
        
        # 🔧 关键优化：为每个机器人BT节点创建专用线程池和回调组
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        # 🔧 优化：移除重复的ThreadPoolExecutor，直接使用MultiThreadedExecutor
        # 注释掉专用ThreadPoolExecutor，避免线程资源重复
        # robot_dedicated_threadpool = ThreadPoolExecutor(
        #     max_workers=4,
        #     thread_name_prefix=f"{robot_namespace}_bt_pool"
        # )
        robot_dedicated_threadpool = None  # 将在后面使用MultiThreadedExecutor
        
        # 创建专用回调组
        robot_dedicated_callback_group = MutuallyExclusiveCallbackGroup()
        
        # 将专用资源存储为节点属性，供behaviors使用
        ros_node.robot_dedicated_callback_group = robot_dedicated_callback_group
        ros_node.robot_dedicated_threadpool = robot_dedicated_threadpool
        
        print(f"🎯 [{robot_namespace}] 创建专用资源:")
        print(f"   • CallbackGroup ID: {id(robot_dedicated_callback_group)}")
        print(f"   • ThreadPool ID: {id(robot_dedicated_threadpool)}")
        print(f"   • 专用线程数: 4")
        print(f"   • 线程名前缀: {robot_namespace}_bt_pool")
        
        # Declare robot parameters to main node
        ros_node.declare_parameter('robot_id', robot_id)
        ros_node.declare_parameter('robot_namespace', robot_namespace)
        
        print(f"="*80)
        print(f"BEHAVIOR TREE FOR ROBOT {robot_id} ({robot_namespace})")
        print(f"="*80)
        
        # Create behavior tree
        # Use robot_namespace directly since we now use "turtlebot{i}" format throughout
        root = create_root(robot_namespace)
        
        # CRITICAL FIX: Initialize blackboard variables BEFORE setup phase
        # This ensures behaviors can access blackboard keys during their setup() calls
        print(f"[SETUP] Pre-initializing blackboard variables for {robot_namespace}...")
        blackboard = py_trees.blackboard.Client()
        blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index",
            access=py_trees.common.Access.WRITE
        )
        blackboard.register_key(
            key=f"{robot_namespace}/system_failed", 
            access=py_trees.common.Access.WRITE
        )
        # Set the initial values
        blackboard.set(f"{robot_namespace}/current_parcel_index", 0)
        blackboard.set(f"{robot_namespace}/system_failed", False)
        print(f"[SETUP] ✓ Pre-initialized blackboard keys: {robot_namespace}/current_parcel_index=0, {robot_namespace}/system_failed=False")
        
        # Print behavior tree structure
        print("BEHAVIOR TREE STRUCTURE:")
        print("="*40)
        print(py_trees.display.ascii_tree(root))
        
        print(f"Tree is running for {robot_namespace}... (Ctrl+C to stop)")
        print("ROS topics will be published for PyTrees Viewer:")
        print(f"  - /{robot_namespace}/tree_log")
        print(f"  - /{robot_namespace}/tree_snapshot")  
        print(f"  - /{robot_namespace}/tree_updates")
        print("="*80)
        
        # Use py_trees_ros.trees.BehaviourTree to create ROS-integrated behavior tree
        tree = py_trees_ros.trees.BehaviourTree(
            root=root,
            unicode_tree_debug=True
        )
        
        # Setup behavior tree - pass ROS node to enable snapshot streams
        tree.setup(timeout=15.0, node=ros_node)
        
        # CRITICAL FIX: Manually setup all behaviors with the shared ROS node
        # PyTrees tree.setup() doesn't automatically pass node parameter to individual behaviors
        setup_behaviors_with_node(root, ros_node)
        
        # Setup snapshot streams for PyTrees Viewer (with error handling)
        try:
            snapshot_topic = setup_snapshot_streams(ros_node, robot_namespace)
            
            # Add TreeToMsgVisitor for PyTrees Viewer
            tree_to_msg_visitor = py_trees_ros.visitors.TreeToMsgVisitor()
            tree.add_visitor(tree_to_msg_visitor)
        except Exception as e:
            print(f"Warning: Could not setup PyTrees viewer integration: {e}")
            print("Continuing without viewer integration...")
        
        # 🔧 关键优化：为每个BT节点创建独立的MultiThreadedExecutor
        # 每个机器人的BT节点使用独立的执行器和线程池，彻底隔离消息处理流
        robot_id = ros_node.get_parameter('robot_id').value
        robot_namespace = ros_node.get_parameter('robot_namespace').value
        
        # 为每个机器人分配独立的线程池，避免线程竞争
        threads_per_robot = 6  # 每个机器人专用6个线程 (之前8个线程优化为6个)
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=threads_per_robot)
        executor.add_node(ros_node)
        
        # 🔧 重要：将MultiThreadedExecutor赋值给节点，供behaviors使用
        ros_node.robot_dedicated_threadpool = executor  # 复用executor替代单独的ThreadPoolExecutor
        
        print(f"🔧 [{robot_namespace}] 创建独立MultiThreadedExecutor: {threads_per_robot}线程专用")
        print(f"   • 优化说明: 移除重复ThreadPoolExecutor, 统一使用MultiThreadedExecutor")
        print(f"   • 线程数从8个减少至6个，消除资源重复")
        print(f"🎯 [{robot_namespace}] 执行器ID: {id(executor)}")
        print(f"🧵 [{robot_namespace}] 线程池独立隔离，避免与其他机器人竞争")
        
        # Use manual ticking loop to ensure ROS topic processing
        iteration_count = 0
        shutdown_requested = False
        
        # Create background thread for ROS executor
        def spin_ros():
            while rclpy.ok() and not shutdown_requested:
                try:
                    executor.spin_once(timeout_sec=0.1)
                except Exception as e:
                    if rclpy.ok() and not shutdown_requested:
                        print(f"ROS executor error: {e}")
                    break
        
        # Start ROS executor thread
        ros_thread = threading.Thread(target=spin_ros)
        ros_thread.daemon = True  # Daemon thread, automatically ends when main thread ends
        ros_thread.start()
        
        while rclpy.ok() and not shutdown_requested:
            tree.tick()
            
            # Print status every 50 ticks
            if iteration_count % 50 == 0:
                tree_status = tree.root.status
                print(f"--- Tick {iteration_count + 1} ---")
                print(f"Tree status: {tree_status}")
                
                # Display tree status
                print(py_trees.display.ascii_tree(tree.root, show_status=True))
            
            # If tree completes, reset and continue
            if tree.root.status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
                tree.root.stop(py_trees.common.Status.INVALID)
                
            iteration_count += 1
            time.sleep(0.5)  # 2Hz rate
        
        print("Behavior tree execution completed.")
        
    except KeyboardInterrupt:
        print("\nShutting down behavior tree...")
        shutdown_requested = True
        if 'tree' in locals():
            tree.shutdown()
    except Exception as e:
        print(f"Error running behavior tree: {e}")
        import traceback
        traceback.print_exc()
        shutdown_requested = True
    finally:
        # Signal shutdown to background thread
        shutdown_requested = True
        
        # 关闭ROS executor（已整合专用线程池功能）
        if 'executor' in locals():
            print(f"🛑 [{robot_namespace}] 关闭MultiThreadedExecutor...")
            executor.shutdown(timeout_sec=2.0)
        
        # Wait for background thread to finish
        if 'ros_thread' in locals():
            ros_thread.join(timeout=2.0)
        
        # Clean up ROS resources
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
