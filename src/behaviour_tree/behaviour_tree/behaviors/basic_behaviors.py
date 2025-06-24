#!/usr/bin/env python3
"""
Basic behavior classes for the behavior tree system.
Contains utility behaviors like waiting, resetting, and message printing.
"""
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
import tf_transformations
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
    1. Parcel is within distance threshold of relay point AND
    2. For non-turtlebot0 robots: last robot is OUT of relay point range
    """
    
    def __init__(self, name, duration=60.0, robot_namespace="turtlebot0", distance_threshold=0.14):
        super().__init__(name)
        self.duration = duration
        self.start_time = 0
        self.robot_namespace = robot_namespace
        self.distance_threshold = distance_threshold
        
        # Extract namespace number for relay point indexing
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        self.relay_number = self.namespace_number  # Relay point is tb{i} -> Relaypoint{i}
        
        # Determine last robot (previous robot in sequence)
        self.last_robot_number = self.namespace_number - 1 if self.namespace_number > 0 else None
        self.is_first_robot = (self.robot_namespace == "turtlebot0")
        
        # ROS2 components (will be initialized in setup)
        self.node = None
        self.callback_group = None  # Add callback group for thread isolation
        self.robot_pose_sub = None
        self.relay_pose_sub = None
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
        """Extract number from namespace (e.g., 'turtlebot0' -> 0, 'turtlebot1' -> 1)"""
        import re
        match = re.search(r'turtlebot(\d+)', namespace)
        return int(match.group(1)) if match else 0

    def get_previous_robot_namespace(self, current_namespace):
        """Get the namespace of the previous robot in sequence"""
        current_number = self.extract_namespace_number(current_namespace)
        if current_number <= 0:
            return None  # turtlebot0 has no previous robot
        previous_number = current_number - 1
        return f"turtlebot{previous_number}"

    def check_previous_robot_finished(self):
        """Check if the previous robot has finished pushing via ROS topic"""
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
        
        # For turtlebot0, default to True (no previous robot)
        if previous_robot_namespace is None:
            print(f"[{self.name}] DEBUG: No previous robot (this is turtlebot0), returning True")
            return True
        
        # For other robots, check the previous robot's pushing_finished flag via ROS topic
        print(f"[{self.name}] DEBUG: Current robot: {self.robot_namespace}, checking previous robot: {previous_robot_namespace}")
        print(f"[{self.name}] DEBUG: Previous robot pushing finished status: {self.previous_robot_pushing_finished}")
        
        return self.previous_robot_pushing_finished
    
    def previous_robot_pushing_finished_callback(self, msg):
        """Callback for previous robot's pushing finished status"""
        self.previous_robot_pushing_finished = msg.data
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
        print(f"[{self.name}] DEBUG: Received {previous_robot_namespace}/pushing_finished = {msg.data}")
    
    def setup_pushing_coordination_topics(self):
        """Setup ROS2 topics for pushing coordination"""
        if self.node is None:
            return
        
        # Publisher for this robot's pushing_finished status
        self.pushing_finished_pub = self.node.create_publisher(
            Bool,
            f'/{self.robot_namespace}/pushing_finished',
            10
        )
        
        # Publisher for this robot's pushing_estimated_time
        self.pushing_estimated_time_pub = self.node.create_publisher(
            Float64,
            f'/{self.robot_namespace}/pushing_estimated_time',
            10
        )
        
        # Subscriber for previous robot's pushing_finished status with callback group
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
        if previous_robot_namespace:
            self.pushing_finished_sub = self.node.create_subscription(
                Bool,
                f'/{previous_robot_namespace}/pushing_finished',
                self.previous_robot_pushing_finished_callback,
                10,
                callback_group=self.callback_group
            )
            print(f"[{self.name}] DEBUG: Subscribed to {previous_robot_namespace}/pushing_finished topic with callback group")
        
        # Publish initial values
        self.publish_pushing_estimated_time()
        print(f"[{self.name}] DEBUG: Setup pushing coordination topics with callback group")
    
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
            f'/turtlebot{self.namespace_number}/odom_map',
            lambda msg: self.handle_callback_in_dedicated_pool(msg, 'robot_pose'),
            10,
            callback_group=self.callback_group
        )
        
        # Relay point pose subscription - 使用专属线程池处理
        self.relay_pose_sub = self.node.create_subscription(
            PoseStamped,
            f'/Relaypoint{self.relay_number}/pose',
            lambda msg: self.handle_callback_in_dedicated_pool(msg, 'relay_pose'),
            10,
            callback_group=self.callback_group
        )
        
        # Last robot pose subscription (only for non-turtlebot0 robots) - 使用专属线程池处理
        self.last_robot_pose_sub = None
        if not self.is_first_robot and self.last_robot_number is not None:
            self.last_robot_pose_sub = self.node.create_subscription(
                Odometry,
                f'/turtlebot{self.last_robot_number}/odom_map',
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
        self.robot_pose = msg.pose.pose
        # Only print occasionally to avoid spam
        if not hasattr(self, '_robot_pose_count'):
            self._robot_pose_count = 0
        self._robot_pose_count += 1
        if self._robot_pose_count % 50 == 1:  # Print every 50 callbacks
            print(f"[{self.name}] DEBUG: Robot pose updated: ({self.robot_pose.position.x:.3f}, {self.robot_pose.position.y:.3f})")
    
    def relay_pose_callback(self, msg):
        """Callback for relay point pose updates"""
        self.relay_pose = msg
        # print(f"[{self.name}] DEBUG: Relay{self.relay_number} pose updated: ({msg.position.x:.3f}, {msg.position.y:.3f})")
    
    def last_robot_pose_callback(self, msg):
        """Callback for last robot pose updates"""
        self.last_robot_pose = msg.pose.pose
    
    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates"""
        self.parcel_pose = msg
        # print(f"[{self.name}] DEBUG: Received parcel{self.current_parcel_index} pose: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}")
    
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
            parcel_topic = f'/parcel{current_parcel_index}/pose'
            self.parcel_pose_sub = self.node.create_subscription(
                PoseStamped,
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
            pos1 = pose1.pose.position
        else:
            pos1 = pose1.position
            
        if hasattr(pose2, 'pose'):
            pos2 = pose2.pose.position
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
        # For turtlebot0 (first robot), this condition is always satisfied
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
        
        # Dynamic feedback message that includes current status
        self.feedback_message = f"[{self.robot_namespace}] PUSH wait for parcel{self.current_parcel_index} -> relay{self.relay_number}"
        print(f"[{self.name}] Starting PUSH wait for {self.duration}s...")
        print(f"[{self.name}] Monitoring parcel{self.current_parcel_index} -> Relaypoint{self.relay_number}")
        if not self.is_first_robot:
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
        
        # First check if previous robot has finished pushing
        previous_finished = self.check_previous_robot_finished()
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
        
        print(f"[{self.name}] DEBUG: Elapsed: {elapsed:.1f}s, Previous robot finished: {previous_finished}")
        
        if not previous_finished:
            print(f"[{self.name}] DEBUG: Still waiting for {previous_robot_namespace} to finish pushing")
            self.feedback_message = f"[{self.robot_namespace}] Waiting for {previous_robot_namespace} to finish pushing..."
            return py_trees.common.Status.RUNNING
        
        # For non-turtlebot0 robots, also check if last robot is out of relay range
        if not self.is_first_robot:
            last_robot_out = self.check_last_robot_out_of_relay_range()
            print(f"[{self.name}] DEBUG: Last robot out of relay range: {last_robot_out}")
            
            if not last_robot_out:
                print(f"[{self.name}] DEBUG: Still waiting for last robot (tb{self.last_robot_number}) to move out of relay range")
                self.feedback_message = f"[{self.robot_namespace}] Waiting for tb{self.last_robot_number} to move out of relay range..."
                return py_trees.common.Status.RUNNING
        
        # Both conditions met (previous robot finished AND last robot out of range for non-turtlebot0)
        print(f"[{self.name}] DEBUG: All conditions satisfied, returning SUCCESS")
        print(f"[{self.name}] SUCCESS: Ready to proceed with pushing!")
        return py_trees.common.Status.SUCCESS
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        # Don't destroy the shared node here - it's managed by the behavior tree
        # Just clean up subscriptions if needed
        if hasattr(self, 'robot_pose_sub') and self.robot_pose_sub:
            try:
                self.node.destroy_subscription(self.robot_pose_sub)
                self.robot_pose_sub = None
            except:
                pass
        if hasattr(self, 'relay_pose_sub') and self.relay_pose_sub:
            try:
                self.node.destroy_subscription(self.relay_pose_sub)
                self.relay_pose_sub = None
            except:
                pass
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
        elif not hasattr(self, 'node') or self.node is None:
            # Create a shared node if one doesn't exist
            import rclpy
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node(f'wait_push_{self.robot_namespace}')
        
        # 🔧 关键优化：使用机器人专用的MutuallyExclusiveCallbackGroup实现线程隔离
        if hasattr(self.node, 'robot_dedicated_callback_group'):
            self.callback_group = self.node.robot_dedicated_callback_group
            print(f"[{self.name}] ✅ 使用机器人专用回调组: {id(self.callback_group)}")
        else:
            # 降级方案：创建独立的MutuallyExclusiveCallbackGroup
            self.callback_group = MutuallyExclusiveCallbackGroup()
            print(f"[{self.name}] ⚠️ 降级：创建独立回调组: {id(self.callback_group)}")
        
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
        try:
            if callback_type == 'robot_pose':
                self.robot_pose_callback(msg)
            elif callback_type == 'relay_pose':
                self.relay_pose_callback(msg)
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
    Wait behavior for picking phase.
    Success condition: 
    - For turtlebot0 (first robot): Always succeed immediately
    - For non-turtlebot0 robots: Success only if replanned trajectory file exists from last robot
    """
    
    def __init__(self, name, duration=2.0, robot_namespace="turtlebot0"):
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
        self.case_name = "simple_maze"  # Default case name
        self.replanned_file_exists = True if self.is_first_robot else False
        
        # No ROS node needed for file-based coordination
        self.node = None
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'turtlebot0' -> 0, 'turtlebot1' -> 1)"""
        match = re.search(r'turtlebot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def check_replanned_file_exists(self):
        """Check if the replanned trajectory file exists from the last robot"""
        if self.is_first_robot:
            return True  # First robot doesn't need to wait for files
        
        # Construct the expected file path for the last robot's replanned trajectory
        import os
        replanned_file_path = f"/root/workspace/data/{self.case_name}/tb{self.last_robot_number}_Trajectory_replanned.json"
        
        file_exists = os.path.exists(replanned_file_path)
        if file_exists and not self.replanned_file_exists:
            # First time detecting the file
            print(f"[{self.name}] Found replanned file: {replanned_file_path}")
            self.replanned_file_exists = True
        
        return file_exists
    
    def check_success_conditions(self):
        """Check if success conditions are met for pick phase"""
        # For turtlebot0 (first robot): always succeed
        if self.is_first_robot:
            return True
        
        # For non-turtlebot0 robots: success only if replanned file exists
        return self.check_replanned_file_exists()
    
    def initialise(self):
        """Initialize the behavior when it starts running"""
        self.start_time = time.time()
        self.feedback_message = f"[{self.robot_namespace}] Waiting for pick phase"
        print(f"[{self.name}] Starting PICK wait for {self.duration}s...")
        
        if self.is_first_robot:
            print(f"[{self.name}] turtlebot0: Always ready for PICK")
        else:
            print(f"[{self.name}] tb{self.namespace_number}: Waiting for replanned file from tb{self.last_robot_number}")
    
    def update(self) -> py_trees.common.Status:
        """Main update method"""
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor
        
        elapsed = time.time() - self.start_time
        
        # Check success conditions
        if self.check_success_conditions():
            if self.is_first_robot:
                print(f"[{self.name}] SUCCESS: turtlebot0 ready for PICK!")
            else:
                print(f"[{self.name}] SUCCESS: Replanned file found from tb{self.last_robot_number}, ready for PICK!")
            return py_trees.common.Status.SUCCESS
        
        # Timeout condition - FAILURE if conditions not met
        if elapsed >= self.duration:
            if self.is_first_robot:
                # This should never happen for turtlebot0, but just in case
                print(f"[{self.name}] WARNING: turtlebot0 timeout (should not happen)")
                return py_trees.common.Status.SUCCESS
            else:
                from .tree_builder import report_node_failure
                error_msg = f"WaitForPick timeout after {elapsed:.1f}s - replanned file not found from tb{self.last_robot_number}"
                report_node_failure(self.name, error_msg, self.robot_namespace)
                print(f"[{self.name}] TIMEOUT: PICK wait FAILED - replanned file not found from tb{self.last_robot_number}")
                return py_trees.common.Status.FAILURE
        
        # Status update every second
        if elapsed % 1.0 < 0.1:  # Print every second
            if self.is_first_robot:
                print(f"[{self.name}] PICK wait... {elapsed:.1f}/{self.duration}s | turtlebot0 ready")
            else:
                file_exists = self.check_replanned_file_exists()
                print(f"[{self.name}] PICK wait... {elapsed:.1f}/{self.duration}s | Replanned file exists: {file_exists}")
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        if hasattr(self, 'node') and self.node is not None:
            self.node.destroy_node()
        super().terminate(new_status)


# Keep the original WaitAction class for backward compatibility
class WaitAction(py_trees.behaviour.Behaviour):
    """Wait action behavior - improved version with pose monitoring and proximity checking"""
    
    def __init__(self, name, duration, robot_namespace="turtlebot0", distance_threshold=0.08):
        super().__init__(name)
        self.duration = duration
        self.start_time = 0
        self.robot_namespace = robot_namespace
        self.distance_threshold = distance_threshold
        
        # Extract namespace number for relay point indexing
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        self.relay_number = self.namespace_number  # Relay point is tb{i} -> Relaypoint{i}
        
        # Initialize ROS2 if not already created
        if not rclpy.ok():
            rclpy.init()
        
        # Create node for subscriptions
        self.node = rclpy.create_node(f'wait_action_{robot_namespace}')
        
        # Pose storage
        self.robot_pose = None
        self.relay_pose = None
        self.parcel_pose = None
        self.current_parcel_index = 0
        
        # Subscriptions
        self.robot_pose_sub = self.node.create_subscription(
            Odometry,
            f'/turtlebot{self.namespace_number}/odom_map',
            self.robot_pose_callback,
            10
        )
        
        self.relay_pose_sub = self.node.create_subscription(
            PoseStamped,
            f'/Relaypoint{self.relay_number}/pose',
            self.relay_pose_callback,
            10
        )
        
        self.parcel_index_sub = self.node.create_subscription(
            Int32,
            f'/{robot_namespace}/current_parcel_index',
            self.current_index_callback,
            10
        )
        
        # Will be updated when parcel index is received
        self.parcel_pose_sub = None
        
        # Parcel subscription will be created in initialise() when behavior starts
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'turtlebot0' -> 0, 'turtlebot1' -> 1)"""
        match = re.search(r'turtlebot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def robot_pose_callback(self, msg):
        """Callback for robot pose updates - handles Odometry message"""
        self.robot_pose = msg.pose.pose
    
    def relay_pose_callback(self, msg):
        """Callback for relay point pose updates"""
        self.relay_pose = msg
    
    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates"""
        self.parcel_pose = msg
        print(f"[{self.name}] Received parcel{self.current_parcel_index} pose: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}")
    
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
            PoseStamped,
            f'/parcel{self.current_parcel_index}/pose',
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
            # This is a PoseStamped message
            pos1 = pose1.pose.position
        else:
            # This is already a Pose message (from Odometry.pose.pose)
            pos1 = pose1.position
            
        if hasattr(pose2, 'pose'):
            # This is a PoseStamped message
            pos2 = pose2.pose.position
        else:
            # This is already a Pose message
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
        print(f"[{self.name}] Monitoring robot: tb{self.namespace_number}, relay: Relaypoint{self.relay_number}")
        
        # Create initial parcel subscription when behavior starts
        self.update_parcel_subscription()
    
    def update(self) -> py_trees.common.Status:
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor
        
        elapsed = time.time() - self.start_time
        
        # Check if parcel is in relay range (primary success condition)
        if self.check_parcel_in_relay_range():
            print(f"[{self.name}] SUCCESS: parcel{self.current_parcel_index} is within range of Relaypoint{self.relay_number}!")
            return py_trees.common.Status.SUCCESS
        
        # Check timeout condition
        if elapsed >= self.duration:
            print(f"[{self.name}] TIMEOUT: Wait completed after {self.duration}s, parcel not in range")
            return py_trees.common.Status.FAILURE
        
        # Still running - provide status update
        parcel_relay_dist = self.calculate_distance(self.parcel_pose, self.relay_pose) if self.parcel_pose and self.relay_pose else float('inf')
        print(f"[{self.name}] Waiting... {elapsed:.1f}/{self.duration}s | parcel{self.current_parcel_index} to Relay{self.relay_number} dist: {parcel_relay_dist:.3f}m (threshold: {self.distance_threshold}m)")
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        if hasattr(self, 'node') and self.node:
            self.node.destroy_node()
        super().terminate(new_status)



class ApproachObject(py_trees.behaviour.Behaviour):
    """
    Approach Object behavior - integrates with State_switch approaching_target logic.
    Uses MPC controller to make the robot approach the parcel based on the logic from State_switch.py.
    """

    def __init__(self, name="ApproachObject", robot_namespace="turtlebot0", approach_distance=0.12):
        """
        Initialize the ApproachObject behavior.
        
        Args:
            name: Name of the behavior node
            robot_namespace: The robot namespace (e.g., 'turtlebot0', 'turtlebot1')
            approach_distance: Distance to maintain from the parcel (default 0.12m)
        """
        super(ApproachObject, self).__init__(name)
        self.robot_namespace = robot_namespace
        self.approach_distance = approach_distance
        
        # Extract namespace number for topic subscriptions
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        
        # Pose storage
        self.robot_pose = None
        self.parcel_pose = None
        self.current_parcel_index = 0
        
        # State variables
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target_state = np.array([0.0, 0.0, 0.0])   # [x, y, theta]
        self.approaching_target = False
        
        # ROS2 components (will be initialized in setup)
        self.ros_node = None
        self.callback_group = None  # Will be initialized in setup
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        self.current_index_sub = None
        self.cmd_vel_pub = None
        
        # MPC controller
        self.mpc = MobileRobotMPC()
        
        # Threading lock for state protection
        self.lock = threading.Lock()
        
    def extract_namespace_number(self, namespace):
        """Extract numerical index from namespace string using regex"""
        import re
        match = re.search(r'\d+', namespace)
        return int(match.group()) if match else 0

    def setup(self, **kwargs):
        """Setup ROS connections"""
        try:
            # Get or create ROS node
            if 'node' in kwargs:
                self.ros_node = kwargs['node']
            else:
                import rclpy
                from rclpy.node import Node
                
                if not rclpy.ok():
                    rclpy.init()
                
                class ApproachObjectNode(Node):
                    def __init__(self):
                        super().__init__(f'approach_object_{self.robot_namespace}')
                
                self.ros_node = ApproachObjectNode()
            
            # 🔧 关键优化：使用机器人专用的MutuallyExclusiveCallbackGroup实现线程隔离
            if hasattr(self.ros_node, 'robot_dedicated_callback_group'):
                self.callback_group = self.ros_node.robot_dedicated_callback_group
                print(f"[{self.name}] ✅ 使用机器人专用回调组: {id(self.callback_group)}")
            else:
                # 降级方案：创建独立的MutuallyExclusiveCallbackGroup
                self.callback_group = MutuallyExclusiveCallbackGroup()
                print(f"[{self.name}] ⚠️ 降级：创建独立回调组: {id(self.callback_group)}")
            
            # Create command velocity publisher
            self.cmd_vel_pub = self.ros_node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            
            # Subscribe to robot pose (Odometry) with callback group
            self.robot_pose_sub = self.ros_node.create_subscription(
                Odometry, f'/turtlebot{self.namespace_number}/odom_map',
                self.robot_pose_callback, 10, callback_group=self.callback_group)
            
            # Subscribe to current parcel index with callback group
            self.current_index_sub = self.ros_node.create_subscription(
                Int32, f'/{self.robot_namespace}/current_parcel_index',
                self.current_index_callback, 10, callback_group=self.callback_group)
            
            # Initial parcel subscription (will be updated based on current index)
            self.update_parcel_subscription()
            
            self.ros_node.get_logger().info(
                f'ApproachObject setup complete for {self.robot_namespace} with callback group')
            return True
            
        except Exception as e:
            print(f"ApproachObject setup failed: {e}")
            return False

    def update_parcel_subscription(self):
        """Update subscription to the correct parcel topic based on current index"""
        if self.ros_node is None:
            return
            
        # Unsubscribe from previous parcel topic if it exists
        if self.parcel_pose_sub is not None:
            self.ros_node.destroy_subscription(self.parcel_pose_sub)
        
        # Subscribe to current parcel topic with callback group
        parcel_topic = f'/parcel{self.current_parcel_index}/pose'
        self.parcel_pose_sub = self.ros_node.create_subscription(
            PoseStamped, parcel_topic, self.parcel_pose_callback, 10,
            callback_group=self.callback_group)
        
        self.ros_node.get_logger().info(f'Updated parcel subscription to: {parcel_topic} with callback group')

    def robot_pose_callback(self, msg):
        """Callback for robot pose updates (Odometry message)"""
        with self.lock:
            self.robot_pose = msg.pose.pose
            # Update current state for MPC
            self.current_state = np.array([
                self.robot_pose.position.x,
                self.robot_pose.position.y,
                self.quaternion_to_yaw(self.robot_pose.orientation)
            ])

    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates (PoseStamped message)"""
        with self.lock:
            self.parcel_pose = msg.pose

    def current_index_callback(self, msg):
        """Callback for current parcel index updates"""
        new_index = msg.data
        if new_index != self.current_parcel_index:
            with self.lock:
                old_index = self.current_parcel_index
                self.current_parcel_index = new_index
                self.update_parcel_subscription()
                self.ros_node.get_logger().info(
                    f'ApproachObject updated parcel index: {old_index} -> {self.current_parcel_index}')

    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        # Handle different pose message types
        if hasattr(pose1, 'pose'):
            pose1 = pose1.pose
        if hasattr(pose2, 'pose'):
            pose2 = pose2.pose
            
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx**2 + dy**2)

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        quat_list = [x, y, z, w]
        euler = tf_transformations.euler_from_quaternion(quat_list)
        return euler[2]

    def get_direction(self, robot_theta, parcel_theta):
        """Get optimal approach direction - from State_switch.py"""
        # Normalize input angles to [-π, π]
        def normalize(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        
        robot_theta = normalize(robot_theta)
        parcel_theta = normalize(parcel_theta)
        
        # Generate candidate angles and normalize
        candidates = [
            parcel_theta,
            normalize(parcel_theta + np.pi/2),  # Turn right 90 degrees
            normalize(parcel_theta - np.pi/2),  # Turn left 90 degrees
            normalize(parcel_theta + np.pi),    # 180 degrees
        ]
        
        # Calculate minimum circular angle difference
        diffs = [abs(normalize(c - robot_theta)) for c in candidates]
        
        index_min = np.argmin(diffs)
        return candidates[index_min]

    def initialise(self):
        """Initialize the behavior when it starts running"""
        self.approaching_target = False
        self.feedback_message = "Initializing approach behavior"
        
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor

    def update(self):
        """
        Main update method - implements the approaching_target logic from State_switch.py
        """
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor

        with self.lock:
            # Check if we have the necessary pose data
            if self.robot_pose is None or self.parcel_pose is None:
                self.feedback_message = "Waiting for pose data..."
                return py_trees.common.Status.RUNNING

            # Calculate distance to parcel
            distance_to_parcel = self.calculate_distance(self.robot_pose, self.parcel_pose)
            
            # Check if we need to approach (distance > 0.25m as in State_switch.py)
            if distance_to_parcel > 0.25:
                self.approaching_target = True
                self.feedback_message = f"Approaching parcel - Distance: {distance_to_parcel:.2f}m"
                
                # Compute target state following State_switch.py logic
                self.target_state = np.array([
                    self.parcel_pose.position.x,
                    self.parcel_pose.position.y,
                    self.quaternion_to_yaw(self.parcel_pose.orientation)
                ])
                
                # Get optimal direction and apply offset (0.12m as in State_switch.py)
                optimal_direction = self.get_direction(
                    self.current_state[2],
                    self.target_state[2]
                )
                self.target_state[2] = optimal_direction
                self.target_state[0] = self.target_state[0] - self.approach_distance * math.cos(optimal_direction)
                self.target_state[1] = self.target_state[1] - self.approach_distance * math.sin(optimal_direction)
                
                # Generate and apply control using MPC
                u = self.mpc.update_control(self.current_state, self.target_state)
                
                if u is not None and self.cmd_vel_pub:
                    cmd = Twist()
                    cmd.linear.x = float(u[0])
                    cmd.angular.z = float(u[1])
                    self.cmd_vel_pub.publish(cmd)
                
                return py_trees.common.Status.RUNNING
                
            else:
                # Robot reached target position (distance <= 0.25m)
                if self.approaching_target:
                    self.approaching_target = False
                    self.feedback_message = "Target position reached, approach complete"
                    
                    # Stop the robot
                    if self.cmd_vel_pub:
                        cmd = Twist()
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
                        self.cmd_vel_pub.publish(cmd)
                    
                    return py_trees.common.Status.SUCCESS
                else:
                    # Already at target, return success immediately
                    self.feedback_message = f"Already at target - Distance: {distance_to_parcel:.2f}m"
                    return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        # Stop the robot
        if self.cmd_vel_pub:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
        
        self.feedback_message = f"ApproachObject terminated with status: {new_status}"


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
    
    def __init__(self, name, robot_namespace="turtlebot0", distance_threshold=0.25):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.distance_threshold = distance_threshold
        
        # Extract namespace number for relay point indexing
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        self.relay_number = self.namespace_number  # Relay point is tb{i} -> Relaypoint{i}
        
        # ROS setup
        self.node = None
        self.robot_pose_sub = None
        self.relay_pose_sub = None
        self.parcel_pose_sub = None
        
        # Pose storage
        self.robot_pose = None
        self.relay_pose = None
        self.parcel_pose = None
        self.current_parcel_index = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Setup blackboard access for current_parcel_index
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.READ
        )
    
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'turtlebot0' -> 0, 'turtlebot1' -> 1)"""
        match = re.search(r'turtlebot(\d+)', namespace)
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
                f'/turtlebot{self.namespace_number}/odom_map',
                self.robot_pose_callback,
                10
            )
            
            # Subscribe to relay point pose
            self.relay_pose_sub = self.node.create_subscription(
                PoseStamped,
                f'/Relaypoint{self.relay_number}/pose',
                self.relay_pose_callback,
                10
            )
            
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
        parcel_topic = f'/parcel{self.current_parcel_index}/pose'
        self.parcel_pose_sub = self.node.create_subscription(
            PoseStamped, parcel_topic, self.parcel_pose_callback, 10)
        
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
            self.parcel_pose = msg
    
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
            parcel_topic = f'/parcel{self.current_parcel_index}/pose'
            self.parcel_pose_sub = self.node.create_subscription(
                PoseStamped, parcel_topic, self.parcel_pose_callback, 10)
            
            self.node.get_logger().info(f'[{self.name}] Updated parcel subscription: {old_index} -> {self.current_parcel_index}')
    
    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        if pose1 is None or pose2 is None:
            return float('inf')
        
        # Handle different pose message types
        if hasattr(pose1, 'pose'):
            pos1 = pose1.pose.position
        else:
            pos1 = pose1.position
            
        if hasattr(pose2, 'pose'):
            pos2 = pose2.pose.position
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
    """Increment index behavior - increments the current parcel index and spawns new parcel via service for turtlebot0"""
    
    def __init__(self, name, robot_namespace="turtlebot0"):
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
        
        # ROS setup for service client (only for turtlebot0)
        self.is_first_robot = (robot_namespace == "turtlebot0")
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
            
            # Only setup service client for turtlebot0
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
            
            # For turtlebot0, check if we need to spawn a new parcel
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

