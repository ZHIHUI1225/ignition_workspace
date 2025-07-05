#!/usr/bin/env python3
"""
Main behavior tree module using modular behavior structure.
Simplified version that imports behaviors from the behaviors package.
"""

# 🔧 CRITICAL: Set environment variables BEFORE any imports that might use BLAS/LAPACK
# import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1' 
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

import py_trees
import py_trees.console as console
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
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


class SharedCallbackGroupManager:
    """Shared callback group manager for all behaviors in a robot - FIXES CALLBACK GROUP PROLIFERATION"""
    def __init__(self, robot_id):
        self.robot_id = robot_id
        print(f"🔧 [{robot_id}] Creating SharedCallbackGroupManager...")
        
        # Create ONLY 3 callback groups per robot (instead of 1 per behavior)
        self.sensor_group = ReentrantCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup() 
        self.coordination_group = ReentrantCallbackGroup()
        
        print(f"   ✅ Sensor group: {id(self.sensor_group)}")
        print(f"   ✅ Control group: {id(self.control_group)}")
        print(f"   ✅ Coordination group: {id(self.coordination_group)}")
    
    def get_group(self, group_type='sensor'):
        """Get callback group by type"""
        if group_type == 'control':
            return self.control_group
        elif group_type == 'coordination':
            return self.coordination_group
        else:
            return self.sensor_group


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


def create_managed_subscription(node, msg_type, topic, callback, qos_profile=10, callback_group_type='sensing'):
    """
    Create a managed subscription that handles lifecycle and prevents duplicates.
    
    Args:
        node: ROS node with subscription_registry and callback_group_pool
        msg_type: Message type class
        topic: Topic name string
        callback: Callback function
        qos_profile: QoS profile (default 10)
        callback_group_type: Type of callback group ('control', 'sensing', 'coordination', 'monitoring')
    
    Returns:
        subscription object or existing subscription if already exists
    """
    registry = node.subscription_registry
    pool = node.callback_group_pool
    
    # Check if subscription already exists
    if topic in registry['active_subscriptions']:
        # Increment reference count
        registry['subscription_counts'][topic] = registry['subscription_counts'].get(topic, 0) + 1
        print(f"[SUBSCRIPTION] Reusing existing subscription for {topic} (refs: {registry['subscription_counts'][topic]})")
        return registry['active_subscriptions'][topic]
    
    # Create new subscription with appropriate callback group
    callback_group = pool.get(callback_group_type, pool['sensing'])  # Default to sensing
    
    try:
        subscription = node.create_subscription(
            msg_type, topic, callback, qos_profile, callback_group=callback_group
        )
        
        # Register subscription
        registry['active_subscriptions'][topic] = subscription
        registry['subscription_counts'][topic] = 1
        
        print(f"[SUBSCRIPTION] Created new managed subscription for {topic} with {callback_group_type} callback group")
        return subscription
        
    except Exception as e:
        print(f"[SUBSCRIPTION] ERROR: Failed to create subscription for {topic}: {e}")
        return None


def destroy_managed_subscription(node, topic):
    """
    Destroy a managed subscription with reference counting.
    
    Args:
        node: ROS node with subscription_registry
        topic: Topic name string
    
    Returns:
        True if subscription was destroyed, False if still has references
    """
    registry = node.subscription_registry
    
    if topic not in registry['active_subscriptions']:
        print(f"[SUBSCRIPTION] WARNING: Attempted to destroy non-existent subscription: {topic}")
        return True
    
    # Decrement reference count
    registry['subscription_counts'][topic] -= 1
    
    # Only destroy if no more references
    if registry['subscription_counts'][topic] <= 0:
        try:
            subscription = registry['active_subscriptions'][topic]
            node.destroy_subscription(subscription)
            
            # Clean up registry
            del registry['active_subscriptions'][topic]
            del registry['subscription_counts'][topic]
            
            print(f"[SUBSCRIPTION] Destroyed managed subscription for {topic}")
            return True
            
        except Exception as e:
            print(f"[SUBSCRIPTION] ERROR: Failed to destroy subscription for {topic}: {e}")
            return False
    else:
        print(f"[SUBSCRIPTION] Kept subscription for {topic} (refs: {registry['subscription_counts'][topic]})")
        return False


def cleanup_all_managed_subscriptions(node):
    """
    Clean up all managed subscriptions for a node.
    
    Args:
        node: ROS node with subscription_registry
    """
    registry = node.subscription_registry
    
    print(f"[SUBSCRIPTION] Cleaning up {len(registry['active_subscriptions'])} managed subscriptions...")
    
    # Destroy all active subscriptions
    for topic, subscription in list(registry['active_subscriptions'].items()):
        try:
            node.destroy_subscription(subscription)
            print(f"[SUBSCRIPTION] Cleaned up subscription: {topic}")
        except Exception as e:
            print(f"[SUBSCRIPTION] ERROR cleaning up {topic}: {e}")
    
    # Clear registry
    registry['active_subscriptions'].clear()
    registry['subscription_counts'].clear()
    
    # Execute cleanup callbacks
    for cleanup_callback in registry['cleanup_callbacks']:
        try:
            cleanup_callback()
        except Exception as e:
            print(f"[SUBSCRIPTION] ERROR in cleanup callback: {e}")
    
    registry['cleanup_callbacks'].clear()
    print(f"[SUBSCRIPTION] All managed subscriptions cleaned up")


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
        
        # 🔧 关键优化：为每个机器人BT节点创建标准化回调组池
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
        
        # 🔧 CRITICAL FIX: Create shared callback group manager to prevent proliferation
        ros_node.shared_callback_manager = SharedCallbackGroupManager(robot_id)
        
        # 🎯 创建标准化回调组池 - FIXED: Use shared groups instead of creating new ones
        callback_group_pool = {
            'control': ros_node.shared_callback_manager.control_group,
            'sensing': ros_node.shared_callback_manager.sensor_group,
            'coordination': ros_node.shared_callback_manager.coordination_group,
            'monitoring': ros_node.shared_callback_manager.sensor_group  # Reuse sensor group
        }
        
        # 🔧 订阅管理器 - 统一管理订阅生命周期，避免重复创建/销毁
        subscription_registry = {
            'active_subscriptions': {},  # {topic_name: subscription_object}
            'subscription_counts': {},   # {topic_name: reference_count}
            'cleanup_callbacks': []      # [cleanup_function_list]
        }
        
        # 将标准化资源存储为节点属性，供behaviors共享使用
        ros_node.callback_group_pool = callback_group_pool
        ros_node.subscription_registry = subscription_registry
        ros_node.robot_dedicated_threadpool = None  # 将在后面使用MultiThreadedExecutor
        
        # 🔧 CRITICAL FIX: Add robot_dedicated_callback_group for behaviors compatibility
        ros_node.robot_dedicated_callback_group = ros_node.shared_callback_manager.control_group
        
        print(f"🎯 [{robot_namespace}] 创建标准化回调组池:")
        print(f"   • Control CallbackGroup ID: {id(callback_group_pool['control'])}")
        print(f"   • Sensing CallbackGroup ID: {id(callback_group_pool['sensing'])}")
        print(f"   • Coordination CallbackGroup ID: {id(callback_group_pool['coordination'])}")
        print(f"   • Monitoring CallbackGroup ID: {id(callback_group_pool['monitoring'])}")
        print(f"   • 订阅注册器: {len(subscription_registry)} 组件")
        print(f"   • 回调组池统一管理，避免behaviors重复创建")
        
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
        
        # 为每个机器人分配独立的线程池，避免线程竞争 - FURTHER OPTIMIZED
        threads_per_robot = 2  # 🔧 FURTHER REDUCED from 4 to 2 threads to fix proliferation issue
        
        # 🔧 EXPERIMENTAL: Use SingleThreadedExecutor for simpler robots to minimize threads
        if robot_id == 0:
            # Robot 0 uses single threaded executor
            executor = rclpy.executors.SingleThreadedExecutor()
            threads_per_robot = 1
            print(f"🧵 [{robot_namespace}] Using SingleThreadedExecutor for minimal thread usage")
        else:
            # Other robots use multi-threaded with minimal threads
            executor = rclpy.executors.MultiThreadedExecutor(num_threads=threads_per_robot)
            print(f"🧵 [{robot_namespace}] Using MultiThreadedExecutor with {threads_per_robot} threads")
        executor.add_node(ros_node)
        
        # 🔧 重要：将MultiThreadedExecutor赋值给节点，供behaviors使用
        ros_node.robot_dedicated_threadpool = executor  # 复用executor替代单独的ThreadPoolExecutor
        
        print(f"🔧 [{robot_namespace}] 创建优化的MultiThreadedExecutor: {threads_per_robot}线程专用")
        print(f"   • 优化说明: 移除重复ThreadPoolExecutor, 统一使用MultiThreadedExecutor")
        print(f"   • 线程数从4个进一步减少至2个，彻底修复线程增殖问题")
        print(f"   🎯 目标总系统线程数: {threads_per_robot * 3} (替代之前的97个)")
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
            
            # Enhanced tree completion handling with reset control
            tree_status = tree.root.status
            if tree_status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
                
                # Log completion details
                completion_info = {
                    'iteration': iteration_count + 1,
                    'status': tree_status,
                    'robot': robot_namespace,
                    'timestamp': time.time()
                }
                print(f"🏁 [{robot_namespace}] Tree completed at iteration {iteration_count + 1}")
                print(f"📊 Final status: {tree_status}")
                
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
        
        # 🔧 新增：清理所有管理的订阅，避免资源泄漏
        if 'ros_node' in locals():
            print(f"🧹 [{robot_namespace}] 清理管理的订阅...")
            cleanup_all_managed_subscriptions(ros_node)
        
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
