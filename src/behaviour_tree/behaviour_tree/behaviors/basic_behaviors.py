#!/usr/bin/env python3
"""
Basic behavior classes for the behavior tree system.
Contains utility behaviors like waiting, resetting, and message printing.
"""

# 🔧 CRITICAL: Set environment variables BEFORE any imports that might use BLAS/LAPACK
import os

import time
import math
import re
import os
import json
import copy
import numpy as np
import py_trees
import rclpy
import casadi as ca
import traceback
import threading
import tf_transformations as tf  # Renamed for convenience
import pyinotify  # For monitoring file system events
from scipy.interpolate import CubicSpline
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Float64, Bool
from std_srvs.srv import Trigger




class WaitForPush(py_trees.behaviour.Behaviour):
    """
    Wait behavior for pushing phase - waits for parcel to be near relay point.
    Success condition: 
    1. Parcel is within distance threshold of relay point (from trajectory) AND
    2. For non-robot0 robots: last robot is OUT of relay point range
    
    Note: Relay points are now loaded from trajectory files instead of subscribed topics.
    """
    
    def __init__(self, name, duration=60.0, robot_namespace="robot0", distance_threshold=0.14):
        super().__init__(name)
        self.duration = duration
        self.start_time = 0
        self.robot_namespace = robot_namespace
        self.distance_threshold = distance_threshold
        
        # Flag to track if behavior has been terminated
        self._terminated = False
        
        # Extract namespace number for relay point indexing
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        self.relay_number = self.namespace_number  # Relay point is tb{i} -> Relaypoint{i}
        
        # Set default case name for trajectory files
        self.case_name = "experi"
        
        # Determine last robot (previous robot in sequence)
        self.last_robot_number = self.namespace_number - 1 if self.namespace_number > 0 else None
        self.is_first_robot = (self.robot_namespace == "robot0")
        
        # ROS2 components (will be initialized in setup)
        self.node = None
        self.callback_group = None  # Add callback group for thread isolation
        self.robot_pose_sub = None
        # relay_pose_sub removed as we're now using trajectory points
        self.parcel_pose_sub = None
        self.last_robot_pose_sub = None
        
        # Pose storage
        self.robot_pose = None
        self.relay_pose = None
        self.parcel_pose = None
        self.last_robot_pose = None  # New: track last robot position
        self.current_parcel_index = 0
        
        # Setup blackboard access for namespaced current_parcel_index (keep this one)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.READ
        )
        
        # ROS2 topics for pushing coordination instead of blackboard
        self.pushing_finished_pub = None
        self.pushing_finished_sub = None
        self.pushing_estimated_time_pub = None
        self.previous_robot_pushing_finished = False  # Track previous robot's status
        
        # Set default pushing estimated time (45 seconds) - will be published via ROS topic
        self.pushing_estimated_time = 45.0
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'robot0' -> 0, 'robot1' -> 1)"""
        import re
        match = re.search(r'robot(\d+)', namespace)
        return int(match.group(1)) if match else 0

    def get_previous_robot_namespace(self, current_namespace):
        """Get the namespace of the previous robot in sequence"""
        current_number = self.extract_namespace_number(current_namespace)
        if current_number <= 0:
            return None  # robot0 has no previous robot
        previous_number = current_number - 1
        return f"robot{previous_number}"

    def check_previous_robot_finished(self):
        """Check if the previous robot has finished pushing via ROS topic"""
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
        
        # For robot0, default to True (no previous robot)
        if previous_robot_namespace is None:
            # Avoid excessive logging
            return True
        
        # For other robots, check the previous robot's pushing_finished flag via ROS topic
        # Only log occasionally to reduce CPU usage
        if not hasattr(self, '_check_print_count'):
            self._check_print_count = 0
        self._check_print_count += 1
        
        if self._check_print_count % 20 == 1:  # Print only once every 20 calls
            print(f"[{self.name}] DEBUG: Current robot: {self.robot_namespace}, checking previous robot: {previous_robot_namespace}")
            print(f"[{self.name}] DEBUG: Previous robot pushing finished status: {self.previous_robot_pushing_finished}")
        
        return self.previous_robot_pushing_finished
    
    def previous_robot_pushing_finished_callback(self, msg):
        """Callback for previous robot's pushing finished status"""
        # Store the previous value to check if it changed
        old_value = self.previous_robot_pushing_finished
        self.previous_robot_pushing_finished = msg.data
        
        # Only log status changes to reduce CPU overhead
        if old_value != msg.data:
            previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
            
            # Add timestamp to help track when we actually got an update
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Only print on state change to reduce CPU load
            if msg.data:
                print(f"[{self.name}] ✅ [{timestamp}] CONFIRMED: Previous robot {previous_robot_namespace} has FINISHED pushing!")
            else:
                print(f"[{self.name}] 🔄 [{timestamp}] Previous robot {previous_robot_namespace} is still pushing")
    
    def setup_pushing_coordination_topics(self):
        """Setup ROS2 topics for pushing coordination"""
        if self.node is None:
            print(f"[{self.name}] ERROR: Cannot setup pushing coordination - ROS node is None!")
            return
        
        # Publisher for this robot's pushing_finished status
        topic_name = f'/{self.robot_namespace}/pushing_finished'
        self.pushing_finished_pub = self.node.create_publisher(
            Bool,
            topic_name,
            10
        )
        print(f"[{self.name}] ✅ Created publisher for {topic_name}")
        
        # Publisher for this robot's pushing_estimated_time
        self.pushing_estimated_time_pub = self.node.create_publisher(
            Float64,
            f'/{self.robot_namespace}/pushing_estimated_time',
            10
        )
        
        # Subscriber for previous robot's pushing_finished status with callback group
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
        if previous_robot_namespace:
            prev_topic_name = f'/{previous_robot_namespace}/pushing_finished'
            self.pushing_finished_sub = self.node.create_subscription(
                Bool,
                prev_topic_name,
                self.previous_robot_pushing_finished_callback,
                10,
                callback_group=self.callback_group
            )
            print(f"[{self.name}] ✅ Subscribed to {prev_topic_name} topic with callback group")
            
            # Debug - publish a test message as the current robot to check topic communication
            test_msg = Bool()
            test_msg.data = False
            self.pushing_finished_pub.publish(test_msg)
            print(f"[{self.name}] 🔍 Published test message to {topic_name}")
        
        # Publish initial values
        self.publish_pushing_estimated_time()
        print(f"[{self.name}] ✅ Setup pushing coordination topics complete")
    
    def publish_pushing_estimated_time(self):
        """Publish the pushing estimated time via ROS topic"""
        if self.pushing_estimated_time_pub:
            msg = Float64()
            msg.data = self.pushing_estimated_time
            self.pushing_estimated_time_pub.publish(msg)
            print(f"[{self.name}] DEBUG: Published pushing_estimated_time = {self.pushing_estimated_time}s")
    
    def setup_subscriptions(self):
        """Setup ROS2 subscriptions with callback group for thread isolation"""
        # Robot pose subscription - 使用专属线程池处理
        self.robot_pose_sub = self.node.create_subscription(
            Odometry,
            f'/robot{self.namespace_number}/odom',
            lambda msg: self.handle_callback_in_dedicated_pool(msg, 'robot_pose'),
            10,
            callback_group=self.callback_group
        )
        
        # Last robot pose subscription (only for non-robot0 robots) - 使用专属线程池处理
        self.last_robot_pose_sub = None
        if not self.is_first_robot and self.last_robot_number is not None:
            self.last_robot_pose_sub = self.node.create_subscription(
                Odometry,
                f'/robot{self.last_robot_number}/odom',
                lambda msg: self.handle_callback_in_dedicated_pool(msg, 'last_robot_pose'),
                10,
                callback_group=self.callback_group
            )
        
        # Setup pushing coordination topics
        self.setup_pushing_coordination_topics()
        
        # Initialize parcel subscription as None - will be set up in initialise() when blackboard is ready
        self.parcel_pose_sub = None
        print(f"[{self.name}] DEBUG: Subscriptions created with dedicated threadpool for {self.robot_namespace}")
        print(f"[{self.name}] DEBUG: Parcel subscription will be created in initialise() when blackboard is ready")
    
    def robot_pose_callback(self, msg):
        """Callback for robot pose updates"""
        # Check if behavior status indicates it should be terminated
        if hasattr(self, '_terminated') and self._terminated:
            print(f"[{self.name}] ⚠️ WARNING: Received callback after termination! This should not happen.")
            return
            
        self.robot_pose = msg.pose.pose
        # Only print occasionally to avoid spam
        if not hasattr(self, '_robot_pose_count'):
            self._robot_pose_count = 0
        self._robot_pose_count += 1
        if self._robot_pose_count % 50 == 1:  # Print every 50 callbacks
            print(f"[{self.name}] DEBUG: Robot pose updated: ({self.robot_pose.position.x:.3f}, {self.robot_pose.position.y:.3f}), count: {self._robot_pose_count}")

    
    
    def last_robot_pose_callback(self, msg):
        """Callback for last robot pose updates"""
        # Check if behavior has been terminated
        if hasattr(self, '_terminated') and self._terminated:
            return
            
        self.last_robot_pose = msg.pose.pose
    
    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates"""
        # Check if behavior has been terminated
        if hasattr(self, '_terminated') and self._terminated:
            return
            
        # Create a simple Pose object from the Odometry message
        from geometry_msgs.msg import Pose
        pose = Pose()
        pose.position = msg.pose.pose.position
        pose.orientation = msg.pose.pose.orientation
        self.parcel_pose = pose
        # print(f"[{self.name}] DEBUG: Received parcel{self.current_parcel_index} pose: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")
    
    def setup_parcel_subscription(self):
        """Set up parcel subscription when blackboard is ready"""
        if self.node is None:
            print(f"[{self.name}] WARNING: Cannot setup parcel subscription - no ROS node")
            return False
            
        try:
            # Get current parcel index from blackboard (with safe fallback)
            current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
            self.current_parcel_index = current_parcel_index
            print(f"[{self.name}] DEBUG: Retrieved parcel index from blackboard: {current_parcel_index}")
            
            # Clean up existing subscription if it exists
            if self.parcel_pose_sub is not None:
                self.node.destroy_subscription(self.parcel_pose_sub)
                print(f"[{self.name}] DEBUG: Destroyed existing parcel subscription")
                
            # Create new parcel subscription with dedicated threadpool for thread isolation
            parcel_topic = f'/parcel{current_parcel_index}/odom'
            self.parcel_pose_sub = self.node.create_subscription(
                Odometry,
                parcel_topic,
                lambda msg: self.handle_callback_in_dedicated_pool(msg, 'parcel_pose'),
                10,
                callback_group=self.callback_group
            )
            print(f"[{self.name}] ✓ Successfully subscribed to {parcel_topic} with dedicated threadpool")
            return True
            
        except Exception as e:
            print(f"[{self.name}] ERROR: Failed to setup parcel subscription: {e}")
            return False
    
    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        # Handle different pose message types
        if hasattr(pose1, 'pose'):
            # Could be a PoseStamped message
            if hasattr(pose1.pose, 'position'):
                pos1 = pose1.pose.position
            # Could be PoseWithCovariance (from Odometry.pose)
            elif hasattr(pose1.pose, 'pose'):
                pos1 = pose1.pose.pose.position
            else:
                pos1 = pose1.pose
        else:
            pos1 = pose1.position
            
        if hasattr(pose2, 'pose'):
            # Could be a PoseStamped message
            if hasattr(pose2.pose, 'position'):
                pos2 = pose2.pose.position
            # Could be PoseWithCovariance (from Odometry.pose)
            elif hasattr(pose2.pose, 'pose'):
                pos2 = pose2.pose.pose.position
            else:
                pos2 = pose2.pose
        else:
            pos2 = pose2.position
        
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return math.sqrt(dx*dx + dy*dy)
    
    def check_parcel_in_relay_range(self):
        """Check if parcel is within range of relay point"""
        if self.parcel_pose is None or self.relay_pose is None:
            return False
        
        distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
        return distance <= self.distance_threshold
    
    def check_last_robot_out_of_relay_range(self):
        """Check if last robot is OUT of relay point range"""
        # For robot0 (first robot), this condition is always satisfied
        if self.is_first_robot:
            return True
        
        # For other robots, check if last robot is out of range
        if self.last_robot_pose is None or self.relay_pose is None:
            return False  # If we can't determine, assume not satisfied
        
        distance = self.calculate_distance(self.last_robot_pose, self.relay_pose)
        is_out_of_range = distance > self.distance_threshold
        return is_out_of_range
    
    def check_success_conditions(self):
        """Check if all success conditions are met"""
        parcel_in_range = self.check_parcel_in_relay_range()
        last_robot_out = self.check_last_robot_out_of_relay_range()
        
        return parcel_in_range and last_robot_out
    
    def initialise(self):
        self.start_time = time.time()
        
        # Set up parcel subscription now that blackboard should be ready
        if not self.setup_parcel_subscription():
            print(f"[{self.name}] WARNING: Failed to setup parcel subscription, using default index 0")
            self.current_parcel_index = 0
        
        # Ensure relay point is loaded from trajectory
        if self.relay_pose is None:
            success, relay_pose = load_relay_point_from_trajectory(
                robot_namespace=self.robot_namespace,
                node=self.node,
                case_name=self.case_name
            )
            
            if success:
                self.relay_pose = relay_pose
                print(f"[{self.name}] ✅ Loaded relay point from trajectory during initialization")
            else:
                print(f"[{self.name}] ⚠️ Failed to load relay point from trajectory")
            
        # Dynamic feedback message that includes current status
        self.feedback_message = f"[{self.robot_namespace}] PUSH wait for parcel{self.current_parcel_index} -> relay{self.relay_number}"
        print(f"[{self.name}] Starting PUSH wait for {self.duration}s...")
        print(f"[{self.name}] Monitoring parcel{self.current_parcel_index} -> trajectory endpoint as relay point")
        if not self.is_first_robot and self.last_robot_number is not None:
            print(f"[{self.name}] Also monitoring that last robot (tb{self.last_robot_number}) is out of relay range")
        
        # Check current conditions at initialization
        parcel_in_range = self.check_parcel_in_relay_range()
        last_robot_out = self.check_last_robot_out_of_relay_range()
        print(f"[{self.name}] Initial conditions - Parcel in range: {parcel_in_range}, Last robot out: {last_robot_out}")
    
    def update(self) -> py_trees.common.Status:
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor
        
        elapsed = time.time() - self.start_time
        
        # Check timeout condition first
        if elapsed >= self.duration:
            from .tree_builder import report_node_failure
            error_msg = f"WaitForPush timeout after {elapsed:.1f}s - previous robot coordination failed"
            report_node_failure(self.name, error_msg, self.robot_namespace)
            print(f"[{self.name}] FAILURE: {error_msg}")
            return py_trees.common.Status.FAILURE
        
        # First check if previous robot has finished pushing - robot0 should skip this check
        previous_finished = True
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
        
        # Only check previous robot for non-robot0 robots
        if not self.is_first_robot and previous_robot_namespace is not None:
            previous_finished = self.check_previous_robot_finished()
        
        # Only print status occasionally to reduce CPU load
        if not hasattr(self, '_status_print_count'):
            self._status_print_count = 0
        self._status_print_count += 1
        
        if self._status_print_count % 10 == 0:  # Print only every 10 cycles
            if self.is_first_robot:
                print(f"[{self.name}] DEBUG: Elapsed: {elapsed:.1f}s, First robot (no previous robot check needed)")
            else:
                print(f"[{self.name}] DEBUG: Elapsed: {elapsed:.1f}s, Previous robot finished: {previous_finished}")
        
        if not previous_finished:
            if self._status_print_count % 10 == 0:  # Print only every 10 cycles
                print(f"[{self.name}] DEBUG: Still waiting for {previous_robot_namespace} to finish pushing")
            self.feedback_message = f"[{self.robot_namespace}] Waiting for {previous_robot_namespace} to finish pushing..."
            return py_trees.common.Status.RUNNING
        
        # For non-robot0 robots, also check if last robot is out of relay range
        if not self.is_first_robot:
            last_robot_out = self.check_last_robot_out_of_relay_range()
            
            if self._status_print_count % 10 == 0:  # Print only every 10 cycles
                print(f"[{self.name}] DEBUG: Last robot out of relay range: {last_robot_out}")
            
            if not last_robot_out and self.last_robot_number is not None:
                if self._status_print_count % 10 == 0:  # Print only every 10 cycles
                    print(f"[{self.name}] DEBUG: Still waiting for last robot (tb{self.last_robot_number}) to move out of relay range")
                self.feedback_message = f"[{self.robot_namespace}] Waiting for tb{self.last_robot_number} to move out of relay range..."
                return py_trees.common.Status.RUNNING
        
        # Both conditions met (previous robot finished AND last robot out of range for non-robot0)
        print(f"[{self.name}] DEBUG: All conditions satisfied, returning SUCCESS")
        print(f"[{self.name}] SUCCESS: Ready to proceed with pushing!")
        return py_trees.common.Status.SUCCESS
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        # Don't destroy the shared node here - it's managed by the behavior tree
        # Just clean up subscriptions if needed
        print(f"[{self.name}] 🔧 TERMINATING: Cleaning up subscriptions, status: {new_status}")
        
        # Mark as terminated to prevent further callbacks
        self._terminated = True
        
        if hasattr(self, 'robot_pose_sub') and self.robot_pose_sub:
            try:
                print(f"[{self.name}] 🔧 Destroying robot_pose_sub...")
                self.node.destroy_subscription(self.robot_pose_sub)
                self.robot_pose_sub = None
                print(f"[{self.name}] ✅ Successfully destroyed robot_pose_sub")
            except Exception as e:
                print(f"[{self.name}] ❌ Error destroying robot_pose_sub: {e}")
                pass
                
        # No need to destroy relay_pose_sub anymore - we're loading from trajectory files
        
        if hasattr(self, 'parcel_pose_sub') and self.parcel_pose_sub:
            try:
                self.node.destroy_subscription(self.parcel_pose_sub)
                self.parcel_pose_sub = None
            except:
                pass
        if hasattr(self, 'last_robot_pose_sub') and self.last_robot_pose_sub:
            try:
                self.node.destroy_subscription(self.last_robot_pose_sub)
                self.last_robot_pose_sub = None
            except:
                pass
        # Clean up pushing coordination topics
        if hasattr(self, 'pushing_finished_sub') and self.pushing_finished_sub:
            try:
                self.node.destroy_subscription(self.pushing_finished_sub)
                self.pushing_finished_sub = None
            except:
                pass
        super().terminate(new_status)
    
    def setup(self, **kwargs):
        """
        Setup ROS2 components using shared node from behavior tree.
        This method is called when the behavior tree is initialized.
        """
        # Get the shared ROS node from kwargs or create one if needed
        if 'node' in kwargs:
            self.node = kwargs['node']
        else:
            print(f"[{self.name}] ❌ ERROR: No shared ROS node provided! Behaviors must use shared node to prevent thread proliferation.")
            return False
        
        # 🔧 CRITICAL FIX: Use shared callback groups to prevent proliferation
        if hasattr(self.node, 'shared_callback_manager'):
            self.callback_group = self.node.shared_callback_manager.get_group('coordination')
            print(f"[{self.name}] ✅ Using shared coordination callback group: {id(self.callback_group)}")
        elif hasattr(self.node, 'robot_dedicated_callback_group'):
            self.callback_group = self.node.robot_dedicated_callback_group
            print(f"[{self.name}] ✅ 使用机器人专用回调组: {id(self.callback_group)}")
        else:
            print(f"[{self.name}] ❌ 错误：没有找到shared_callback_manager，无法使用共享回调组")
            return False
        
        # Setup ROS subscriptions now that we have a node
        self.setup_subscriptions()
        
        # Call parent setup
        return super().setup(**kwargs)
    
    def handle_callback_in_dedicated_pool(self, msg, callback_type):
        """
        优化版本：直接处理回调，利用MutuallyExclusiveCallbackGroup实现隔离
        移除了重复的ThreadPoolExecutor, 减少线程资源消耗
        """
        # 直接在专属CallbackGroup的线程中处理，无需额外线程池
        return self._execute_callback_in_pool(msg, callback_type)
    
    def _execute_callback_in_pool(self, msg, callback_type):
        """
        在专属线程池中执行实际的回调处理逻辑
        """
        # First check if behavior has been terminated - skip callbacks if so
        if hasattr(self, '_terminated') and self._terminated:
            return
            
        try:
            if callback_type == 'robot_pose':
                self.robot_pose_callback(msg)
            elif callback_type == 'last_robot_pose':
                self.last_robot_pose_callback(msg)
            elif callback_type == 'parcel_pose':
                self.parcel_pose_callback(msg)
            else:
                print(f"[{self.name}] WARNING: Unknown callback type: {callback_type}")
        except Exception as e:
            print(f"[{self.name}] ERROR in dedicated pool callback: {e}")
            import traceback
            traceback.print_exc()


class WaitForPick(py_trees.behaviour.Behaviour):
    """
    Wait behavior for picking phase using inotify for file system monitoring.
    Success condition: 
    - For robot0 (first robot): Always succeed immediately
    - For non-robot0 robots: Success only if replanned trajectory file exists from last robot
    """
    
    def __init__(self, name, duration=2.0, robot_namespace="robot0"):
        super().__init__(name)
        self.duration = duration
        self.start_time = 0
        self.robot_namespace = robot_namespace
        
        # Extract namespace number
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        
        # Multi-robot coordination logic
        self.is_first_robot = (self.namespace_number == 0)
        self.last_robot_number = self.namespace_number - 1 if not self.is_first_robot else None
        
        # File-based coordination instead of ROS messages
        self.case_name = "experi"  # Default case name
        self._file_exists = True if self.is_first_robot else False  # Internal state, use property access
        
        # File monitoring components
        self.watch_manager = None
        self.notifier = None
        self.watch_descriptor = None
        self.watch_path = f"/root/workspace/data/{self.case_name}"
        # Set default target file name - or use "none" placeholder for first robot
        self.target_file = "none_placeholder.json" if self.last_robot_number is None else f"tb{self.last_robot_number}_Trajectory_replanned.json"
        
        # No ROS node needed for file-based coordination
        self.node = None
        
    @property
    def replanned_file_exists(self):
        """Property accessor for file existence state - only updated by EventHandler"""
        return self._file_exists
    
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'robot0' -> 0, 'robot1' -> 1)"""
        match = re.search(r'robot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def check_replanned_file_exists(self):
        """Initial check if the replanned trajectory file exists from the last robot"""
        if self.is_first_robot:
            return True  # First robot doesn't need to wait for files
        
        # Construct the expected file path for the last robot's replanned trajectory
        import os
        # For first robot, use placeholder path; otherwise use actual replanned file path
        replanned_file_path = "/root/workspace/data/none_placeholder.json" if self.last_robot_number is None else f"/root/workspace/data/{self.case_name}/tb{self.last_robot_number}_Trajectory_replanned.json"
        
        file_exists = os.path.exists(replanned_file_path)
        if file_exists and not self._file_exists:
            # First time detecting the file
            print(f"[{self.name}] Found replanned file during initial check: {replanned_file_path}")
            self._file_exists = True
        
        return file_exists
    
    def check_success_conditions(self):
        """Check if success conditions are met for pick phase"""
        # Simplified logic: first robot always succeeds, others succeed when file exists
        return self.is_first_robot or self.replanned_file_exists
    
    def setup_file_watcher(self):
        """Set up inotify watcher for file creation events"""
        if self.is_first_robot:
            return True  # First robot doesn't need to monitor files
            
        try:
            # Ensure the directory to watch exists
            if not os.path.exists(self.watch_path):
                try:
                    os.makedirs(self.watch_path, exist_ok=True)
                    print(f"[{self.name}] Created watch directory: {self.watch_path}")
                except Exception as e:
                    print(f"[{self.name}] Error creating watch directory: {str(e)}")
                    # Continue anyway - the directory might be created later
            
            # Setup inotify watcher
            self.watch_manager = pyinotify.WatchManager()
            
            # Event handler for file creation/modification
            class EventHandler(pyinotify.ProcessEvent):
                def __init__(self, waitforpick_instance):
                    self.waitforpick = waitforpick_instance
                    
                def process_IN_CREATE(self, event):
                    self._check_target_file(event)
                    
                def process_IN_CLOSE_WRITE(self, event):
                    self._check_target_file(event)
                    
                def process_IN_MOVED_TO(self, event):
                    self._check_target_file(event)
                    
                def _check_target_file(self, event):
                    if event.name == self.waitforpick.target_file:
                        self.waitforpick._file_exists = True
                        print(f"[{self.waitforpick.name}] ⚡ File event detected: {event.pathname} was created/modified")
            
            # Set up the notifier with our event handler
            handler = EventHandler(self)
            self.notifier = pyinotify.ThreadedNotifier(self.watch_manager, handler)
            self.notifier.daemon = True  # Set as daemon thread to avoid blocking process exit
            self.notifier.start()
            
            # Add a watch for the directory with auto_add to watch new subdirectories
            mask = pyinotify.IN_CREATE | pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MOVED_TO
            self.watch_descriptor = self.watch_manager.add_watch(self.watch_path, mask, rec=False, auto_add=True)
            
            if self.watch_path in self.watch_descriptor:
                watch_success = (self.watch_descriptor[self.watch_path] > 0)
                if watch_success:
                    print(f"[{self.name}] ✓ File watcher set up for directory: {self.watch_path}")
                    print(f"[{self.name}] ✓ Monitoring for file: {self.target_file}")
                else:
                    print(f"[{self.name}] ⚠️ Watch descriptor for {self.watch_path} appears invalid")
            else:
                print(f"[{self.name}] ⚠️ Directory {self.watch_path} not in watch descriptors")
            
            # Also do an initial check in case the file already exists
            self.check_replanned_file_exists()
            
            return True
        except Exception as e:
            print(f"[{self.name}] ❌ Error setting up file watcher: {str(e)}")
            return False
    
    def _retry_file_watcher_setup(self):
        """Try to re-setup the file watcher if it has failed or stopped working"""
        if self.is_first_robot:
            return True  # First robot doesn't need file monitoring
        
        # Clean up any existing watcher to prevent resource leaks
        if self.notifier is not None:
            try:
                self.notifier.stop()
                print(f"[{self.name}] Stopped existing file watcher before retry")
            except Exception as e:
                print(f"[{self.name}] Error stopping existing watcher: {str(e)}")
                
        # Reset components
        self.watch_manager = None
        self.notifier = None
        self.watch_descriptor = None
        
        # Try to setup the watcher again
        retry_success = self.setup_file_watcher()
        if retry_success:
            print(f"[{self.name}] ✅ Successfully restored file monitoring")
        else:
            print(f"[{self.name}] ❌ Failed to restore file monitoring")
            
        return retry_success
    
    def initialise(self):
        """Initialize the behavior when it starts running"""
        self.start_time = time.time()
        self.feedback_message = f"[{self.robot_namespace}] Waiting for pick phase"
        print(f"[{self.name}] Starting PICK wait for {self.duration}s...")
        
        if self.is_first_robot:
            print(f"[{self.name}] robot0: Always ready for PICK")
        else:
            # Set up the file watcher using inotify
            if not self.setup_file_watcher():
                print(f"[{self.name}] ERROR: Failed to set up file watcher - file detection may not work!")
            
            if self.last_robot_number is not None:
                print(f"[{self.name}] tb{self.namespace_number}: Waiting for replanned file from tb{self.last_robot_number}")
            else:
                print(f"[{self.name}] tb{self.namespace_number}: First robot - no previous robot file needed")
    
    def update(self) -> py_trees.common.Status:
        """Main update method - fully event-driven, no polling"""
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor
        
        elapsed = time.time() - self.start_time
        
        # Check success conditions - file existence flag is updated by inotify events in background
        if self.check_success_conditions():
            if self.is_first_robot:
                print(f"[{self.name}] SUCCESS: robot0 ready for PICK!")
            else:
                robot_identifier = f"tb{self.last_robot_number}" if self.last_robot_number is not None else "previous robot"
                print(f"[{self.name}] SUCCESS: Replanned file found from {robot_identifier}, ready for PICK!")
            return py_trees.common.Status.SUCCESS
        
        # Check and attempt to restore inotify watcher if needed (every 5 seconds)
        # This handles cases where the directory was deleted/recreated or other monitoring failures
        if not self.is_first_robot and elapsed % 5.0 < 0.1:
            watcher_inactive = (
                self.notifier is None or 
                self.watch_manager is None or 
                not self.watch_descriptor or 
                self.watch_path not in self.watch_descriptor
            )
            
            if watcher_inactive:
                print(f"[{self.name}] WARNING: File monitoring appears to be inactive, attempting to restore...")
                self._retry_file_watcher_setup()
            
            # Also check if the watch directory exists, if not, retry setup
            elif not os.path.exists(self.watch_path):
                print(f"[{self.name}] WARNING: Watch directory {self.watch_path} no longer exists, attempting to restore...")
                self._retry_file_watcher_setup()
        
        # Timeout condition - FAILURE if conditions not met
        if elapsed >= self.duration:
            if self.is_first_robot:
                # This should never happen for robot0, but just in case
                print(f"[{self.name}] WARNING: robot0 timeout (should not happen)")
                return py_trees.common.Status.SUCCESS
            else:
                from .tree_builder import report_node_failure
                # Safer error message that handles None case
                robot_identifier = f"tb{self.last_robot_number}" if self.last_robot_number is not None else "previous robot"
                error_msg = f"WaitForPick timeout after {elapsed:.1f}s - replanned file not found from {robot_identifier}"
                report_node_failure(self.name, error_msg, self.robot_namespace)
                # Safer error message that handles None case
                robot_identifier = f"tb{self.last_robot_number}" if self.last_robot_number is not None else "previous robot"
                print(f"[{self.name}] TIMEOUT: PICK wait FAILED - replanned file not found from {robot_identifier}")
                return py_trees.common.Status.FAILURE
        
        # Status update every second
        if elapsed % 1.0 < 0.1:  # Print every second
            if self.is_first_robot:
                print(f"[{self.name}] PICK wait... {elapsed:.1f}/{self.duration}s | robot0 ready")
            else:
                print(f"[{self.name}] PICK wait... {elapsed:.1f}/{self.duration}s | Replanned file exists: {self.replanned_file_exists}")
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        # Clean up file watcher components
        if self.notifier is not None:
            try:
                self.notifier.stop()
                print(f"[{self.name}] File watcher notifier stopped")
            except Exception as e:
                print(f"[{self.name}] Error stopping notifier: {str(e)}")
        
        # Release watch descriptors and manager
        if self.watch_manager is not None:
            try:
                if self.watch_descriptor and self.watch_path in self.watch_descriptor:
                    self.watch_manager.rm_watch(self.watch_descriptor[self.watch_path])
                self.watch_manager = None
                print(f"[{self.name}] Watch manager cleaned up")
            except Exception as e:
                print(f"[{self.name}] Error cleaning up watch manager: {str(e)}")
            
        # Clean up ROS node if it exists
        if hasattr(self, 'node') and self.node is not None:
            try:
                self.node.destroy_node()
                print(f"[{self.name}] Node destroyed")
            except:
                pass
            
        super().terminate(new_status)


# Keep the original WaitAction class for backward compatibility
class WaitAction(py_trees.behaviour.Behaviour):
    """Wait action behavior - improved version with pose monitoring and proximity checking"""
    
    def __init__(self, name, duration, robot_namespace="robot0", distance_threshold=0.08):
        super().__init__(name)
        self.duration = duration
        self.start_time = 0
        self.robot_namespace = robot_namespace
        self.distance_threshold = distance_threshold
        
        # Extract namespace number for relay point indexing
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        self.relay_number = self.namespace_number  # Relay point is tb{i} -> Relaypoint{i}
        
        # NOTE: ROS node should be provided via setup() method to use shared node
        # Creating separate nodes causes thread proliferation (9 threads per node)
        self.node = None  # Will be set in setup()
        
        # Pose storage
        self.robot_pose = None
        self.relay_pose = None  # Will be loaded from trajectory file
        self.parcel_pose = None
        self.current_parcel_index = 0
        
        # Subscriptions - initialized in setup()
        self.robot_pose_sub = None
        self.parcel_index_sub = None
        self.parcel_pose_sub = None
        
        # Case name for trajectory file
        self.case_name = "experi"  # Default case name
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'robot0' -> 0, 'robot1' -> 1)"""
        match = re.search(r'robot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def setup(self, **kwargs):
        """Setup ROS node and subscriptions"""
        try:
            self.node = kwargs.get('node')
            if self.node is None:
                print(f"[{self.name}] No ROS node provided")
                return False
            
            # Setup ROS subscriptions now that we have a node
            self.setup_subscriptions()
            
            # Load relay point from trajectory file
            success, relay_pose = load_relay_point_from_trajectory(
                robot_namespace=self.robot_namespace,
                node=self.node,
                case_name=self.case_name
            )
            
            if success:
                self.relay_pose = relay_pose
                print(f"[{self.name}] ✅ Loaded relay point from trajectory")
            else:
                print(f"[{self.name}] ⚠️ Failed to load relay point from trajectory, will try again later")
            
            return True
            
        except Exception as e:
            print(f"[{self.name}] Setup failed: {e}")
            return False
    
    def setup_subscriptions(self):
        """Setup ROS subscriptions"""
        # Subscribe to robot pose (Odometry)
        self.robot_pose_sub = self.node.create_subscription(
            Odometry,
            f'/robot{self.namespace_number}/odom',
            self.robot_pose_callback,
            10
        )
        
        # Subscribe to current parcel index
        self.parcel_index_sub = self.node.create_subscription(
            Int32,
            f'/{self.robot_namespace}/current_parcel_index',
            self.current_index_callback,
            10
        )
        
        # Parcel subscription will be created in update_parcel_subscription
        self.update_parcel_subscription()
        
        print(f"[{self.name}] ✅ Subscriptions set up for {self.robot_namespace}")
    
    def robot_pose_callback(self, msg):
        """Callback for robot pose updates - handles Odometry message"""
        self.robot_pose = msg.pose.pose
    
    def relay_pose_callback(self, msg):
        """Callback for relay point pose updates (compatibility stub)"""
        print(f"[{self.name}] WARNING: relay_pose_callback called - this should not happen")
        # We don't update self.relay_pose here anymore as it's loaded from trajectory
    
    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates"""
        self.parcel_pose = msg.pose.pose  # Extract pose from Odometry message
        print(f"[{self.name}] Received parcel{self.current_parcel_index} pose: x={msg.pose.pose.position.x:.3f}, y={msg.pose.pose.position.y:.3f}")
    
    def current_index_callback(self, msg):
        """Callback for current parcel index updates"""
        old_index = self.current_parcel_index
        self.current_parcel_index = msg.data
        
        # Update parcel subscription if index changed
        if old_index != self.current_parcel_index:
            self.update_parcel_subscription()
    
    def update_parcel_subscription(self):
        """Update parcel pose subscription based on current index"""
        # Destroy old subscription if it exists
        if self.parcel_pose_sub is not None:
            self.node.destroy_subscription(self.parcel_pose_sub)
        
        # Create new subscription for current parcel
        self.parcel_pose_sub = self.node.create_subscription(
            Odometry,
            f'/parcel{self.current_parcel_index}/odom',
            self.parcel_pose_callback,
            10
        )
        print(f"[{self.name}] Updated parcel subscription to parcel{self.current_parcel_index}")
    
    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        # Handle different pose message types
        # pose1 could be from Odometry (robot_pose) - extract position directly
        # pose2 could be from PoseStamped (relay/parcel) - extract from .pose.position
        if hasattr(pose1, 'pose'):
            # Could be a PoseStamped message
            if hasattr(pose1.pose, 'position'):
                pos1 = pose1.pose.position
            # Could be PoseWithCovariance (from Odometry.pose)
            elif hasattr(pose1.pose, 'pose'):
                pos1 = pose1.pose.pose.position
            else:
                pos1 = pose1.pose
        else:
            pos1 = pose1.position
            
        if hasattr(pose2, 'pose'):
            # Could be a PoseStamped message
            if hasattr(pose2.pose, 'position'):
                pos2 = pose2.pose.position
            # Could be PoseWithCovariance (from Odometry.pose)
            elif hasattr(pose2.pose, 'pose'):
                pos2 = pose2.pose.pose.position
            else:
                pos2 = pose2.pose
        else:
            pos2 = pose2.position
        
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return math.sqrt(dx*dx + dy*dy)
    
    def check_parcel_in_relay_range(self):
        """Check if parcel is within range of relay point"""
        if self.parcel_pose is None or self.relay_pose is None:
            return False
        
        distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
        return distance <= self.distance_threshold
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"[{self.robot_namespace}] Waiting for {self.duration}s and monitoring parcel proximity"
        print(f"[{self.name}] Starting wait for {self.duration}s with parcel monitoring...")
        print(f"[{self.name}] Monitoring robot: tb{self.namespace_number}, using relay point from trajectory")
        
        # Load relay point from trajectory if not already loaded
        if self.relay_pose is None:
            success, relay_pose = load_relay_point_from_trajectory(
                robot_namespace=self.robot_namespace,
                node=self.node,
                case_name=self.case_name
            )
            
            if success:
                self.relay_pose = relay_pose
                print(f"[{self.name}] ✅ Loaded relay point from trajectory during initialization")
            else:
                print(f"[{self.name}] ⚠️ Failed to load relay point from trajectory")
        
        # Create initial parcel subscription when behavior starts
        self.update_parcel_subscription()
    
    def update(self) -> py_trees.common.Status:
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor
        
        elapsed = time.time() - self.start_time
        
        # Check if parcel is in relay range (primary success condition)
        if self.check_parcel_in_relay_range():
            print(f"[{self.name}] SUCCESS: parcel{self.current_parcel_index} is within range of relay point!")
            return py_trees.common.Status.SUCCESS
        
        # Check timeout condition
        if elapsed >= self.duration:
            print(f"[{self.name}] TIMEOUT: Wait completed after {self.duration}s, parcel not in range")
            return py_trees.common.Status.FAILURE
        
        # Still running - provide status update
        parcel_relay_dist = self.calculate_distance(self.parcel_pose, self.relay_pose) if self.parcel_pose and self.relay_pose else float('inf')
        print(f"[{self.name}] Waiting... {elapsed:.1f}/{self.duration}s | parcel{self.current_parcel_index} to relay point dist: {parcel_relay_dist:.3f}m (threshold: {self.distance_threshold}m)")
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        # Stop the robot
        if hasattr(self, 'cmd_vel_pub') and self.cmd_vel_pub:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
        
        self.feedback_message = f"WaitAction terminated with status: {new_status}"


class MobileRobotMPC:
    """Simple MPC controller for mobile robot"""
    
    def __init__(self):
        self.last_control = [0.0, 0.0]  # [linear_vel, angular_vel]
    
    def update_control(self, current_state, target_state):
        """
        Update control based on current and target states
        
        Args:
            current_state: [x, y, theta] current robot state
            target_state: [x, y, theta] target robot state
            
        Returns:
            control: [linear_vel, angular_vel] control commands
        """
        try:
            # Simple proportional controller
            dx = target_state[0] - current_state[0]
            dy = target_state[1] - current_state[1]
            dtheta = target_state[2] - current_state[2]
            
            # Normalize angle difference
            while dtheta > math.pi:
                dtheta -= 2 * math.pi
            while dtheta < -math.pi:
                dtheta += 2 * math.pi
            
            # Calculate distance to target
            distance = math.sqrt(dx**2 + dy**2)
            
            # Simple control law
            if distance > 0.05:  # If not at target
                # Calculate desired heading
                desired_theta = math.atan2(dy, dx)
                heading_error = desired_theta - current_state[2]
                
                # Normalize heading error
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi
                
                # Control gains
                kp_linear = 0.5
                kp_angular = 1.0
                
                # Generate control commands
                linear_vel = min(kp_linear * distance, 0.3)  # Max 0.3 m/s
                angular_vel = kp_angular * heading_error
                
                # Limit angular velocity
                angular_vel = max(-1.0, min(1.0, angular_vel))
                
                # If heading error is large, prioritize turning
                if abs(heading_error) > 0.5:
                    linear_vel *= 0.5
                
                self.last_control = [linear_vel, angular_vel]
                return self.last_control
            else:
                # At target, stop
                self.last_control = [0.0, 0.0]
                return self.last_control
                
        except Exception as e:
            print(f"MPC error: {e}")
            return [0.0, 0.0]


class StopSystem(py_trees.behaviour.Behaviour):
    """Stop system behavior - stops the system for a specified duration"""
    
    def __init__(self, name, duration=1.0):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Stopping system for {self.duration}s"
        print(f"[{self.name}] Starting to stop system...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            print(f"[{self.name}] System stopped!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class CheckPairComplete(py_trees.behaviour.Behaviour):
    """Check if a pair operation is complete behavior - both robot and parcel out of relay range"""
    
    def __init__(self, name, robot_namespace="robot0", distance_threshold=0.25):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.distance_threshold = distance_threshold
        
        # Extract namespace number for relay point indexing
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        self.relay_number = self.namespace_number  # Relay point is tb{i} -> Relaypoint{i}
        
        # ROS setup
        self.node = None
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        
        # Pose storage
        self.robot_pose = None
        self.relay_pose = None  # Will be loaded from trajectory file
        self.parcel_pose = None
        self.current_parcel_index = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Case name for trajectory file
        self.case_name = "experi"  # Default case name
        
        # Setup blackboard access for current_parcel_index
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.READ
        )
    
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'robot0' -> 0, 'robot1' -> 1)"""
        match = re.search(r'robot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def setup(self, **kwargs):
        """Setup ROS node and subscriptions"""
        try:
            self.node = kwargs.get('node')
            if self.node is None:
                print(f"[{self.name}] No ROS node provided")
                return False
            
            # Subscribe to robot pose (Odometry)
            self.robot_pose_sub = self.node.create_subscription(
                Odometry,
                f'/robot{self.namespace_number}/odom',
                self.robot_pose_callback,
                10
            )
            
            # Load relay point from trajectory file
            success, relay_pose = load_relay_point_from_trajectory(
                robot_namespace=self.robot_namespace,
                node=self.node,
                case_name=self.case_name
            )
            
            if success:
                self.relay_pose = relay_pose
                print(f"[{self.name}] ✅ Loaded relay point from trajectory")
            else:
                print(f"[{self.name}] ⚠️ Failed to load relay point from trajectory, will try again later")
            
            # Parcel subscription will be created in initialise() when behavior starts
            
            self.node.get_logger().info(f'CheckPairComplete setup complete for {self.robot_namespace}')
            return True
            
        except Exception as e:
            print(f"[{self.name}] Setup failed: {e}")
            return False
    
    def update_parcel_subscription(self):
        """Update subscription to the correct parcel topic based on current index"""
        if self.node is None:
            return
            
        # Unsubscribe from previous parcel topic if it exists
        if self.parcel_pose_sub is not None:
            self.node.destroy_subscription(self.parcel_pose_sub)
        
        # Subscribe to current parcel topic
        parcel_topic = f'/parcel{self.current_parcel_index}/odom'
        self.parcel_pose_sub = self.node.create_subscription(
            Odometry, parcel_topic, self.parcel_pose_callback, 10)
        
        self.node.get_logger().info(f'[{self.name}] Updated parcel subscription to: {parcel_topic}')
    
    def robot_pose_callback(self, msg):
        """Callback for robot pose updates - handles Odometry message"""
        with self.lock:
            self.robot_pose = msg.pose.pose
    
    def relay_pose_callback(self, msg):
        """Callback for relay point pose updates"""
        with self.lock:
            self.relay_pose = msg
    
    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates"""
        with self.lock:
            self.parcel_pose = msg.pose.pose  # Extract pose from Odometry message
    
    def update_parcel_subscription(self):
        """Update subscription to the correct parcel topic based on current index from blackboard"""
        if self.node is None:
            return
        
        # Read current_parcel_index from blackboard
        try:
            current_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
        except (KeyError, AttributeError):
            current_index = 0
        
        # Only update subscription if the index has actually changed
        if current_index != self.current_parcel_index or self.parcel_pose_sub is None:
            old_index = self.current_parcel_index
            self.current_parcel_index = current_index
            
            # Unsubscribe from previous parcel topic if it exists
            if self.parcel_pose_sub is not None:
                self.node.destroy_subscription(self.parcel_pose_sub)
            
            # Subscribe to current parcel topic
            parcel_topic = f'/parcel{self.current_parcel_index}/odom'
            self.parcel_pose_sub = self.node.create_subscription(
                Odometry, parcel_topic, self.parcel_pose_callback, 10)
            
            self.node.get_logger().info(f'[{self.name}] Updated parcel subscription: {old_index} -> {self.current_parcel_index}')
    
    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        # Handle different pose message types
        if hasattr(pose1, 'pose'):
            # Could be a PoseStamped message
            if hasattr(pose1.pose, 'position'):
                pos1 = pose1.pose.position
            # Could be PoseWithCovariance (from Odometry.pose)
            elif hasattr(pose1.pose, 'pose'):
                pos1 = pose1.pose.pose.position
            else:
                pos1 = pose1.pose
        else:
            pos1 = pose1.position
            
        if hasattr(pose2, 'pose'):
            # Could be a PoseStamped message
            if hasattr(pose2.pose, 'position'):
                pos2 = pose2.pose.position
            # Could be PoseWithCovariance (from Odometry.pose)
            elif hasattr(pose2.pose, 'pose'):
                pos2 = pose2.pose.pose.position
            else:
                pos2 = pose2.pose
        else:
            pos2 = pose2.position
        
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return math.sqrt(dx*dx + dy*dy)
    
    def check_robot_out_of_relay_range(self):
        """Check if robot is OUT of relay point range"""
        if self.robot_pose is None or self.relay_pose is None:
            return False
        
        distance = self.calculate_distance(self.robot_pose, self.relay_pose)
        is_out_of_range = distance > self.distance_threshold
        return is_out_of_range
    
    def check_parcel_out_of_relay_range(self):
        """Check if parcel is OUT of relay point range"""
        if self.parcel_pose is None or self.relay_pose is None:
            return False
        
        distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
        is_out_of_range = distance > self.distance_threshold
        return is_out_of_range
    
    def initialise(self):
        """Initialize the behavior when it starts - ensure parcel subscription is up to date"""
        print(f"[{self.name}] 初始化CheckPairComplete行为...")
        
        # 🔧 关键修复：确保在行为开始时正确设置parcel订阅
        self.update_parcel_subscription()
        
        # 重置位姿数据，强制等待新的数据
        with self.lock:
            self.robot_pose = None
            self.relay_pose = None  
            self.parcel_pose = None
            
        print(f"[{self.name}] 当前包裹索引: {self.current_parcel_index}")
        print(f"[{self.name}] 等待位姿数据...")

    def update(self):
        # Update parcel subscription from blackboard
        self.update_parcel_subscription()
        
        with self.lock:
            # Check if both robot and parcel are out of relay range
            robot_out = self.check_robot_out_of_relay_range()
            parcel_out = self.check_parcel_out_of_relay_range()
            
            # Calculate distances for debugging
            robot_dist = self.calculate_distance(self.robot_pose, self.relay_pose) if self.robot_pose and self.relay_pose else float('inf')
            parcel_dist = self.calculate_distance(self.parcel_pose, self.relay_pose) if self.parcel_pose and self.relay_pose else float('inf')
            
            # 🔍 详细诊断parcel_dist为inf的原因
            parcel_status = "OK"
            if self.parcel_pose is None:
                parcel_status = f"parcel_pose=None (订阅话题: /parcel{self.current_parcel_index}/pose)"
            elif self.relay_pose is None:
                parcel_status = f"relay_pose=None (订阅话题: /Relaypoint{self.relay_number}/pose)"
            
            if parcel_dist == float('inf'):
                print(f"[{self.name}] ⚠️  Parcel距离为inf的原因: {parcel_status}")
                print(f"[{self.name}] 🔍 当前包裹索引: {self.current_parcel_index}")
                print(f"[{self.name}] 🔍 包裹订阅状态: {self.parcel_pose_sub is not None}")
                print(f"[{self.name}] 🔍 中继点订阅状态: {self.relay_pose_sub is not None}")
            
            print(f"[{self.name}] Robot dist: {robot_dist:.2f}, Parcel dist: {parcel_dist:.2f}, Threshold: {self.distance_threshold}")
            print(f"[{self.name}] Robot out: {robot_out}, Parcel out: {parcel_out}")
            
            if robot_out and parcel_out:
                print(f"[{self.name}] Pair operation complete - both out of range")
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING

    def initialise(self):
        """Initialize the behavior when it starts - ensure parcel subscription is up to date"""
        print(f"[{self.name}] 初始化CheckPairComplete行为...")
        
        # 🔧 关键修复：确保在行为开始时正确设置parcel订阅
        self.update_parcel_subscription()
        
        # 重置位姿数据，强制等待新的数据
        with self.lock:
            self.robot_pose = None
            self.relay_pose = None  
            self.parcel_pose = None
            
        print(f"[{self.name}] 当前包裹索引: {self.current_parcel_index}")
        print(f"[{self.name}] 等待位姿数据...")


class IncrementIndex(py_trees.behaviour.Behaviour):
    """Increment index behavior - increments the current parcel index and spawns new parcel via service for robot0"""
    
    def __init__(self, name, robot_namespace="robot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        
        # Setup blackboard access (only for current_parcel_index)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.WRITE
        )
        
        # ROS2 topic for pushing coordination instead of blackboard
        self.pushing_finished_pub = None
        
        # ROS setup for service client (only for robot0)
        self.is_first_robot = (robot_namespace == "robot0")
        self.node = None
        self.spawn_service_client = None
        
        # Parcel spawning tracking
        self.spawned_parcels = set()  # Track which parcels have been spawned
        self.max_parcels = 5  # Maximum number of parcels to spawn
        self.pending_spawn_requests = {}  # Track pending service calls
        
        # Initialize with parcel 0 already spawned (assuming it exists at start)
        if self.is_first_robot:
            self.spawned_parcels.add(0)
            print(f"[{self.name}] Initialized - parcel0 assumed to be already spawned")
    
    def setup(self, **kwargs):
        """Setup ROS node and service client"""
        try:
            self.node = kwargs.get('node')
            if self.node is None:
                print(f"[{self.name}] No ROS node provided")
                return False
            
            # Create publisher for pushing_finished coordination
            self.pushing_finished_pub = self.node.create_publisher(
                Bool, f'/{self.robot_namespace}/pushing_finished', 10)
            print(f"[{self.name}] DEBUG: Created pushing_finished topic publisher")
            
            # Only setup service client for robot0
            if self.is_first_robot:
                # Create service client for spawning parcels
                self.spawn_service_client = self.node.create_client(
                    Trigger, 
                    '/spawn_next_parcel_service'
                )
                
                # Wait for service to be available
                if not self.spawn_service_client.wait_for_service(timeout_sec=5.0):
                    print(f"[{self.name}] Spawn service not available")
                    return False
                
                print(f"[{self.name}] Spawn service client ready")
            
            return True
            
        except Exception as e:
            print(f"[{self.name}] Setup failed: {e}")
            return False
    
    def _should_spawn_new_parcel(self, new_index):
        """Check if a new parcel should be spawned"""
        if not self.is_first_robot:
            return False
            
        # Only spawn if:
        # 1. We haven't reached max parcels
        # 2. This parcel hasn't been spawned yet
        # 3. We don't have a pending request for this parcel
        if (new_index <= self.max_parcels and 
            new_index not in self.spawned_parcels and 
            new_index not in self.pending_spawn_requests):
            return True
        return False
    
    def _spawn_new_parcel(self, parcel_index):
        """Spawn a new parcel using the spawn_next_parcel_service"""
        if not self.is_first_robot or self.spawn_service_client is None:
            return False
            
        try:
            # Create service request
            request = Trigger.Request()
            
            # Call service asynchronously
            future = self.spawn_service_client.call_async(request)
            self.pending_spawn_requests[parcel_index] = future
            
            print(f"[{self.name}] Requesting spawn for parcel{parcel_index} via service")
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error calling spawn service for parcel{parcel_index}: {e}")
            return False
    
    def _check_spawn_requests(self):
        """Check status of pending spawn requests"""
        completed_requests = []
        
        for parcel_index, future in self.pending_spawn_requests.items():
            if future.done():
                try:
                    response = future.result()
                    if response.success:
                        self.spawned_parcels.add(parcel_index)
                        print(f"[{self.name}] Successfully spawned parcel{parcel_index}: {response.message}")
                    else:
                        print(f"[{self.name}] Failed to spawn parcel{parcel_index}: {response.message}")
                except Exception as e:
                    print(f"[{self.name}] Service call failed for parcel{parcel_index}: {e}")
                
                completed_requests.append(parcel_index)
        
        # Remove completed requests
        for parcel_index in completed_requests:
            del self.pending_spawn_requests[parcel_index]
    
    def _check_parcel_spawned(self, parcel_index):
        """Check if a parcel has been successfully spawned"""
        return parcel_index in self.spawned_parcels
    
    def update(self):
        try:
            # Check status of any pending spawn requests
            if self.is_first_robot:
                self._check_spawn_requests()
            
            # Get current index
            current_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
            new_index = current_index + 1
            
            # For robot0, check if we need to spawn a new parcel
            if self.is_first_robot:
                if self._should_spawn_new_parcel(new_index):
                    spawn_success = self._spawn_new_parcel(new_index)
                    if not spawn_success:
                        from .tree_builder import report_node_failure
                        error_msg = f"Failed to spawn parcel{new_index} via service call"
                        report_node_failure(self.name, error_msg, self.robot_namespace)
                        print(f"[{self.name}] Failed to request spawn for parcel{new_index}, keeping current index")
                        return py_trees.common.Status.FAILURE
                
                # Check if we have a pending request for this parcel
                if new_index in self.pending_spawn_requests:
                    print(f"[{self.name}] Waiting for parcel{new_index} spawn service to complete...")
                    return py_trees.common.Status.RUNNING
                
                # Verify the parcel is available before incrementing
                if not self._check_parcel_spawned(new_index):
                    if new_index <= self.max_parcels:
                        print(f"[{self.name}] Parcel{new_index} not yet spawned, waiting...")
                        return py_trees.common.Status.RUNNING
                    else:
                        from .tree_builder import report_node_failure
                        error_msg = f"Reached maximum parcels ({self.max_parcels}) - operation complete"
                        report_node_failure(self.name, error_msg, self.robot_namespace)
                        print(f"[{self.name}] Reached maximum parcels ({self.max_parcels}), stopping")
                        return py_trees.common.Status.FAILURE
            
            # Update the index
            setattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', new_index)
            
            # Reset pushing_finished flag to False for the new parcel via ROS topic
            if self.pushing_finished_pub:
                msg = Bool()
                msg.data = False
                self.pushing_finished_pub.publish(msg)
                print(f"[{self.name}] DEBUG: Published pushing_finished = False via ROS topic for new parcel{new_index}")
            
            print(f"[{self.name}] Incremented index from {current_index} to {new_index}")
            
            if self.is_first_robot:
                print(f"[{self.name}] Spawned parcels: {sorted(list(self.spawned_parcels))}")
            
            return py_trees.common.Status.SUCCESS
            
        except Exception as e:
            from .tree_builder import report_node_failure
            error_msg = f"IncrementIndex error: {str(e)}"
            report_node_failure(self.name, error_msg, self.robot_namespace)
            print(f"[{self.name}] Error incrementing index: {e}")
            return py_trees.common.Status.FAILURE


def load_relay_point_from_trajectory(robot_namespace, node=None, case_name="simple_maze"):
    """Load relay point from trajectory file
    
    Args:
        robot_namespace (str): The robot namespace (e.g., 'robot0')
        node (rclpy.node.Node, optional): ROS node for timestamp. Defaults to None.
        case_name (str, optional): Case name for trajectory file path. Defaults to "simple_maze".
        
    Returns:
        tuple: (success, relay_pose_msg) - where relay_pose_msg is a PoseStamped message or None if failed
    """
    try:
        # Extract namespace number
        match = re.search(r'robot(\d+)', robot_namespace)
        namespace_number = int(match.group(1)) if match else 0
        
        # Determine the trajectory file path
        trajectory_file_path = f"/root/workspace/data/{case_name}/tb{namespace_number}_Trajectory.json"
        
        # Check if file exists
        if not os.path.exists(trajectory_file_path):
            print(f"[load_relay_point] WARNING: Trajectory file not found: {trajectory_file_path}")
            return False, None
            
        # Load the trajectory data
        with open(trajectory_file_path, 'r') as json_file:
            data = json.load(json_file)
            trajectory = data.get('Trajectory', [])
            
            if not trajectory:
                print(f"[load_relay_point] WARNING: Empty trajectory in file: {trajectory_file_path}")
                return False, None
            
            # Use the first point of the trajectory as the relay point
            first_point = trajectory[0]

            # Create a PoseStamped message for the relay point
            relay_pose_msg = PoseStamped()
            relay_pose_msg.header.frame_id = "map"
            relay_pose_msg.header.stamp = node.get_clock().now().to_msg() if node else None
            
            # Set position from trajectory point
            relay_pose_msg.pose.position.x = first_point[0]
            relay_pose_msg.pose.position.y = first_point[1]

            # Set orientation if available
            if len(first_point) > 2:
                # Convert heading to quaternion
                quat = tf.quaternion_from_euler(0, 0, first_point[2])
                relay_pose_msg.pose.orientation.x = quat[0]
                relay_pose_msg.pose.orientation.y = quat[1]
                relay_pose_msg.pose.orientation.z = quat[2]
                relay_pose_msg.pose.orientation.w = quat[3]
            
            print(f"[load_relay_point] ✅ Successfully loaded relay point from trajectory: ({relay_pose_msg.pose.position.x:.3f}, {relay_pose_msg.pose.position.y:.3f})")
            return True, relay_pose_msg
            
    except Exception as e:
        print(f"[load_relay_point] ERROR: Failed to load relay point from trajectory: {str(e)}")
        traceback.print_exc()
        return False, None