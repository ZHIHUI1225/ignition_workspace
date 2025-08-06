#!/usr/bin/env python3
"""
Base behavior class with standardized callback group and subscription management.
This solves the callback group collision and subscription lifecycle issues.
"""

import py_trees
import threading
import time
import traceback
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy


class ManagedBehavior(py_trees.behaviour.Behaviour):
    """
    Base class for behaviors with standardized callback group and subscription management.
    Solves issues:
    - Multiple behaviors creating separate callback groups
    - Mixed usage of callback group types
    - Improper cleanup of callback groups
    - Excessive subscription creation/destruction
    - Non-blocking lock issues
    """
    
    def __init__(self, name, robot_namespace="robot0", **kwargs):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.node = None
        
        # 🔧 标准化锁管理 - 使用阻塞锁避免callback跳过
        self._state_lock = threading.RLock()  # 可重入锁，避免死锁
        self._subscription_lock = threading.Lock()
        
        # 🔧 订阅生命周期管理
        self._managed_subscriptions = set()  # 跟踪此behavior创建的订阅
        self._cleanup_registered = False
        
        # 🔧 状态管理
        self._setup_completed = False
        self._terminating = False
        
        # 🔧 回调计数器（用于调试）
        self._callback_counts = {}
        
        print(f"[{self.name}] ManagedBehavior initialized for {robot_namespace}")
    
    def get_callback_group(self, callback_type='sensing'):
        """
        Get standardized callback group from the node's pool.
        
        Args:
            callback_type: 'control', 'sensing', 'coordination'
        
        Returns:
            Appropriate callback group from the shared pool
        """
        if not self.node or not hasattr(self.node, 'callback_group_pool'):
            print(f"[{self.name}] WARNING: No callback group pool available, using default")
            return None
        
        callback_group = self.node.callback_group_pool.get(callback_type, self.node.callback_group_pool['sensing'])
        print(f"[{self.name}] Using {callback_type} callback group: {id(callback_group)}")
        return callback_group
    
    def create_managed_subscription(self, msg_type, topic, callback, qos_profile=10, callback_group_type='sensing'):
        """
        Create a managed subscription using the node's subscription registry.
        
        Args:
            msg_type: Message type class
            topic: Topic name
            callback: Callback function  
            qos_profile: QoS profile
            callback_group_type: Type of callback group to use
        
        Returns:
            Subscription object
        """
        if not self.node:
            print(f"[{self.name}] ERROR: Cannot create subscription - no ROS node")
            return None
        
        # 使用节点的管理订阅功能
        from .subscription_manager import create_managed_subscription
        subscription = create_managed_subscription(
            self.node, msg_type, topic, callback, qos_profile, callback_group_type
        )
        
        if subscription:
            # 记录此behavior创建的订阅
            self._managed_subscriptions.add(topic)
            print(f"[{self.name}] Created managed subscription: {topic}")
        
        return subscription
    
    def destroy_managed_subscription(self, topic):
        """
        Destroy a managed subscription.
        
        Args:
            topic: Topic name to destroy
        """
        if not self.node:
            return
        
        from .subscription_manager import destroy_managed_subscription
        if destroy_managed_subscription(self.node, topic):
            self._managed_subscriptions.discard(topic)
            print(f"[{self.name}] Destroyed managed subscription: {topic}")
    
    def create_reliable_qos_profile(self):
        """Create a reliable QoS profile for critical subscriptions."""
        return QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
    
    def create_best_effort_qos_profile(self):
        """Create a best effort QoS profile for high-frequency data."""
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
    
    def safe_callback_wrapper(self, callback_name, original_callback):
        """
        Wrap callbacks with proper error handling and state management.
        
        Args:
            callback_name: Name for debugging
            original_callback: Original callback function
            
        Returns:
            Wrapped callback function
        """
        def wrapped_callback(*args, **kwargs):
            # Skip if terminating
            if self._terminating:
                return
            
            # Update callback count for debugging
            self._callback_counts[callback_name] = self._callback_counts.get(callback_name, 0) + 1
            
            # 🔧 使用阻塞锁确保回调不被跳过
            with self._state_lock:
                try:
                    # Debug output every 100 calls
                    if self._callback_counts[callback_name] % 100 == 1:
                        print(f"[{self.name}] {callback_name} callback #{self._callback_counts[callback_name]}")
                    
                    return original_callback(*args, **kwargs)
                    
                except Exception as e:
                    print(f"[{self.name}] ERROR in {callback_name} callback: {e}")
                    traceback.print_exc()
        
        return wrapped_callback
    
    def setup(self, **kwargs):
        """
        Base setup method - must be called by subclasses.
        
        Args:
            **kwargs: Should contain 'node' parameter
        """
        if 'node' in kwargs:
            self.node = kwargs['node']
            
            # 验证节点具有必需的属性
            if not hasattr(self.node, 'callback_group_pool'):
                print(f"[{self.name}] ERROR: Node missing callback_group_pool")
                return False
            
            if not hasattr(self.node, 'subscription_registry'):
                print(f"[{self.name}] ERROR: Node missing subscription_registry")
                return False
            
            # 注册清理回调
            if not self._cleanup_registered:
                self.node.subscription_registry['cleanup_callbacks'].append(self._cleanup_all_subscriptions)
                self._cleanup_registered = True
            
            self._setup_completed = True
            print(f"[{self.name}] Base setup completed successfully")
            return True
        else:
            print(f"[{self.name}] ERROR: No ROS node provided in setup")
            return False
    
    def initialise(self):
        """Base initialise method with state management."""
        if not self._setup_completed:
            print(f"[{self.name}] WARNING: initialise() called before setup() completed")
        
        self._terminating = False
        print(f"[{self.name}] Behavior initialized")
    
    def terminate(self, new_status):
        """Base terminate method with proper cleanup."""
        print(f"[{self.name}] Terminating with status: {new_status}")
        
        # Set termination flag to stop callbacks
        self._terminating = True
        
        # Small delay to allow callbacks to exit
        time.sleep(0.01)
        
        # Clean up subscriptions created by this behavior
        self._cleanup_behavior_subscriptions()
        
        print(f"[{self.name}] Termination cleanup completed")
    
    def _cleanup_behavior_subscriptions(self):
        """Clean up subscriptions created by this specific behavior."""
        with self._subscription_lock:
            subscriptions_to_cleanup = self._managed_subscriptions.copy()
            
            for topic in subscriptions_to_cleanup:
                try:
                    self.destroy_managed_subscription(topic)
                except Exception as e:
                    print(f"[{self.name}] ERROR cleaning up subscription {topic}: {e}")
            
            self._managed_subscriptions.clear()
            print(f"[{self.name}] Cleaned up {len(subscriptions_to_cleanup)} behavior subscriptions")
    
    def _cleanup_all_subscriptions(self):
        """Callback for node-wide cleanup - registered during setup."""
        print(f"[{self.name}] Node-wide cleanup callback executed")
        self._cleanup_behavior_subscriptions()
    
    def get_state_safely(self, state_getter_func):
        """
        Safely get state with proper locking.
        
        Args:
            state_getter_func: Function to get state
            
        Returns:
            State value or None if failed
        """
        try:
            with self._state_lock:
                return state_getter_func()
        except Exception as e:
            print(f"[{self.name}] ERROR getting state: {e}")
            return None
    
    def set_state_safely(self, state_setter_func):
        """
        Safely set state with proper locking.
        
        Args:
            state_setter_func: Function to set state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._state_lock:
                state_setter_func()
                return True
        except Exception as e:
            print(f"[{self.name}] ERROR setting state: {e}")
            return False
