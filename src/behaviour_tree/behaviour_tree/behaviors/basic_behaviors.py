#!/usr/bin/env python3
"""
Basic behavior classes for the behavior tree system.
Contains utility behaviors like waiting, resetting, and message printing.
"""
import py_trees
import py_trees.behaviour
import py_trees.common
import py_trees.blackboard
import rclpy
import time
import math
import re
import threading
import casadi as ca
import numpy as np
import tf_transformations
import json
import os
import copy
from scipy.interpolate import CubicSpline
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Float64


class ResetFlags(py_trees.behaviour.Behaviour):
    """Reset system flags behavior"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def update(self):
        print(f"[{self.name}] Resetting flags...")
        return py_trees.common.Status.SUCCESS


class WaitForPush(py_trees.behaviour.Behaviour):
    """
    Wait behavior for pushing phase - waits for parcel to be near relay point.
    Success condition: 
    1. Parcel is within distance threshold of relay point AND
    2. For non-turtlebot0 robots: last robot is OUT of relay point range
    """
    
    def __init__(self, name, duration=10.0, robot_namespace="turtlebot0", distance_threshold=0.08):
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
        
        # Setup blackboard access for namespaced current_parcel_index
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/pushing_estimated_time", 
            access=py_trees.common.Access.WRITE
        )
        
        # Set default pushing estimated time (45 seconds)
        setattr(self.blackboard, f"{robot_namespace}/pushing_estimated_time", 45.0)
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'turtlebot0' -> 0, 'turtlebot1' -> 1)"""
        match = re.search(r'turtlebot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def setup_subscriptions(self):
        """Setup ROS2 subscriptions"""
        # Robot pose subscription
        self.robot_pose_sub = self.node.create_subscription(
            Odometry,
            f'/turtlebot{self.namespace_number}/odom_map',
            self.robot_pose_callback,
            10
        )
        
        # Relay point pose subscription
        self.relay_pose_sub = self.node.create_subscription(
            PoseStamped,
            f'/Relaypoint{self.relay_number}/pose',
            self.relay_pose_callback,
            10
        )
        
        # Last robot pose subscription (only for non-turtlebot0 robots)
        self.last_robot_pose_sub = None
        if not self.is_first_robot and self.last_robot_number is not None:
            self.last_robot_pose_sub = self.node.create_subscription(
                Odometry,
                f'/turtlebot{self.last_robot_number}/odom_map',
                self.last_robot_pose_callback,
                10
            )
        
        # Initial parcel subscription
        self.parcel_pose_sub = None
        self.update_parcel_subscription()
    
    def robot_pose_callback(self, msg):
        """Callback for robot pose updates"""
        self.robot_pose = msg.pose.pose
    
    def relay_pose_callback(self, msg):
        """Callback for relay point pose updates"""
        self.relay_pose = msg
    
    def last_robot_pose_callback(self, msg):
        """Callback for last robot pose updates"""
        self.last_robot_pose = msg.pose.pose
    
    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates"""
        self.parcel_pose = msg
        # print(f"[{self.name}] DEBUG: Received parcel{self.current_parcel_index} pose: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}")
    
    def update_parcel_subscription(self):
        """Update parcel pose subscription based on current index"""
        # Skip if node is not yet initialized
        if self.node is None:
            return
            
        # Read current_parcel_index from namespaced blackboard using registered client
        try:
            # Use the blackboard client to access the namespaced variable
            global_blackboard = py_trees.blackboard.Client()
            current_index = global_blackboard.get(f'{self.robot_namespace}/current_parcel_index')
        except (KeyError, AttributeError):
            # If key doesn't exist, use default value 0
            current_index = 0
        
        # Only update subscription if the index has actually changed
        if current_index != self.current_parcel_index or self.parcel_pose_sub is None:
            old_index = self.current_parcel_index
            self.current_parcel_index = current_index
            print(f"[{self.name}] Updated parcel index from blackboard: {old_index} -> {self.current_parcel_index}")
            
            # Destroy old subscription if it exists
            if self.parcel_pose_sub is not None:
                self.node.destroy_subscription(self.parcel_pose_sub)
                print(f"[{self.name}] DEBUG: Destroyed old parcel subscription")
            
            # Create new subscription for the updated index
            self.parcel_pose_sub = self.node.create_subscription(
                PoseStamped,
                f'/parcel{self.current_parcel_index}/pose',
                self.parcel_pose_callback,
                10
            )
            print(f"[{self.name}] DEBUG: Created subscription to /parcel{self.current_parcel_index}/pose")
    
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
        # Dynamic feedback message that includes current status
        self.feedback_message = f"PUSH wait for parcel{self.current_parcel_index} -> relay{self.relay_number}"
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
        
        # Update parcel subscription from blackboard
        self.update_parcel_subscription()
        
        elapsed = time.time() - self.start_time
        
        # Check both success conditions
        if self.check_success_conditions():
            print(f"[{self.name}] SUCCESS: All conditions met for PUSH!")
            print(f"[{self.name}] - Parcel{self.current_parcel_index} is near Relaypoint{self.relay_number}")
            if not self.is_first_robot:
                print(f"[{self.name}] - Last robot (tb{self.last_robot_number}) is out of relay range")
            return py_trees.common.Status.SUCCESS
        
        # Timeout condition - FAILURE if conditions not met
        if elapsed >= self.duration:
            print(f"[{self.name}] TIMEOUT: PUSH wait FAILED - conditions not satisfied")
            parcel_in_range = self.check_parcel_in_relay_range()
            last_robot_out = self.check_last_robot_out_of_relay_range()
            print(f"[{self.name}] - Parcel in range: {parcel_in_range}")
            print(f"[{self.name}] - Last robot out of range: {last_robot_out}")
            return py_trees.common.Status.FAILURE
        
        # Status update with detailed information
        parcel_relay_dist = self.calculate_distance(self.parcel_pose, self.relay_pose) if self.parcel_pose and self.relay_pose else float('inf')
        parcel_in_range = self.check_parcel_in_relay_range()
        last_robot_out = self.check_last_robot_out_of_relay_range()
        
        if elapsed % 1.0 < 0.1:  # Print every second
            print(f"[{self.name}] PUSH wait... {elapsed:.1f}/{self.duration}s")
            print(f"[{self.name}] - Parcel to relay dist: {parcel_relay_dist:.3f}m (in range: {parcel_in_range})")
            if not self.is_first_robot:
                last_robot_dist = self.calculate_distance(self.last_robot_pose, self.relay_pose) if self.last_robot_pose and self.relay_pose else float('inf')
                print(f"[{self.name}] - Last robot to relay dist: {last_robot_dist:.3f}m (out of range: {last_robot_out})")
        
        return py_trees.common.Status.RUNNING
    
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
        
        # Setup ROS subscriptions now that we have a node
        self.setup_subscriptions()
        
        # Call parent setup
        return super().setup(**kwargs)


class WaitForPick(py_trees.behaviour.Behaviour):
    """
    Wait behavior for picking phase.
    Success condition: 
    - For turtlebot0 (first robot): Always succeed immediately
    - For non-turtlebot0 robots: Success only if estimated time from last robot's pushing task was received
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
        
        # Initialize ROS2 if needed
        if not rclpy.ok():
            rclpy.init()
        
        # Create node for subscriptions (only for non-turtlebot0 robots)
        self.node = None
        if not self.is_first_robot:
            self.node = rclpy.create_node(f'wait_pick_{robot_namespace}')
        
        # Estimated time tracking for non-turtlebot0 robots
        self.estimated_time_received = True if self.is_first_robot else False
        self.last_estimated_time = None
        
        # Setup subscriptions
        self.setup_subscriptions()
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'turtlebot0' -> 0, 'turtlebot1' -> 1)"""
        match = re.search(r'turtlebot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def setup_subscriptions(self):
        """Setup ROS2 subscriptions only for non-turtlebot0 robots"""
        if not self.is_first_robot and self.node is not None:
            self.estimated_time_sub = self.node.create_subscription(
                Float64,
                f'/turtlebot{self.last_robot_number}/estimated_time',
                self.estimated_time_callback,
                10
            )
    
    def estimated_time_callback(self, msg):
        """Callback for estimated time from last robot's pushing task"""
        self.last_estimated_time = msg.data
        self.estimated_time_received = True
        print(f"[{self.name}] Received estimated time from tb{self.last_robot_number}: {msg.data:.2f}s")
    
    def check_success_conditions(self):
        """Check if success conditions are met for pick phase"""
        # For turtlebot0 (first robot): always succeed
        if self.is_first_robot:
            return True
        
        # For non-turtlebot0 robots: success only if estimated time was received
        return self.estimated_time_received
    
    def initialise(self):
        """Initialize the behavior when it starts running"""
        self.start_time = time.time()
        self.feedback_message = "Waiting for pick phase"
        print(f"[{self.name}] Starting PICK wait for {self.duration}s...")
        
        if self.is_first_robot:
            print(f"[{self.name}] turtlebot0: Always ready for PICK")
        else:
            print(f"[{self.name}] tb{self.namespace_number}: Waiting for estimated time from tb{self.last_robot_number}")
    
    def update(self) -> py_trees.common.Status:
        """Main update method"""
        # NOTE: Do NOT call rclpy.spin_once() here as we're using MultiThreadedExecutor
        
        elapsed = time.time() - self.start_time
        
        # Check success conditions
        if self.check_success_conditions():
            if self.is_first_robot:
                print(f"[{self.name}] SUCCESS: turtlebot0 ready for PICK!")
            else:
                print(f"[{self.name}] SUCCESS: Estimated time received from tb{self.last_robot_number}, ready for PICK!")
            return py_trees.common.Status.SUCCESS
        
        # Timeout condition - FAILURE if conditions not met
        if elapsed >= self.duration:
            if self.is_first_robot:
                # This should never happen for turtlebot0, but just in case
                print(f"[{self.name}] WARNING: turtlebot0 timeout (should not happen)")
                return py_trees.common.Status.SUCCESS
            else:
                print(f"[{self.name}] TIMEOUT: PICK wait FAILED - estimated time not received from tb{self.last_robot_number}")
                return py_trees.common.Status.FAILURE
        
        # Status update every second
        if elapsed % 1.0 < 0.1:  # Print every second
            if self.is_first_robot:
                print(f"[{self.name}] PICK wait... {elapsed:.1f}/{self.duration}s | turtlebot0 ready")
            else:
                print(f"[{self.name}] PICK wait... {elapsed:.1f}/{self.duration}s | Est. time received: {self.estimated_time_received}")
        
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
        
        # Create initial parcel subscription for index 0
        self.update_parcel_subscription()
        
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
        self.feedback_message = f"Waiting for {self.duration}s and monitoring parcel proximity"
        print(f"[{self.name}] Starting wait for {self.duration}s with parcel monitoring...")
        print(f"[{self.name}] Monitoring robot: tb{self.namespace_number}, relay: Relaypoint{self.relay_number}")
    
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


class ReplanPath(py_trees.behaviour.Behaviour):
    """Path replanning behavior with trajectory optimization"""
    
    def __init__(self, name, duration=1.5, robot_namespace="turtlebot0", case="simple_maze"):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
        self.robot_namespace = robot_namespace
        self.case = case
        
        # Extract robot ID from namespace for file paths
        self.robot_id = self.extract_namespace_number(robot_namespace)
        
        # Setup blackboard access to read pushing_estimated_time
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/pushing_estimated_time", 
            access=py_trees.common.Access.READ
        )
        
        # Replanning state
        self.replanning_complete = False
        self.replanned_successfully = False
        
    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'turtlebot0' -> 0)"""
        match = re.search(r'turtlebot(\d+)', namespace)
        return int(match.group(1)) if match else 0
    
    def initialise(self):
        self.start_time = time.time()
        self.replanning_complete = False
        self.replanned_successfully = False
        self.feedback_message = f"Replanning path for {self.duration}s"
        print(f"[{self.name}] Starting trajectory replanning for robot {self.robot_id}...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        
        # Perform replanning on first update
        if not self.replanning_complete:
            try:
                # Get target time from blackboard
                target_time = getattr(self.blackboard, f'{self.robot_namespace}/pushing_estimated_time', 45.0)
                print(f"[{self.name}] Target time from blackboard: {target_time:.2f}s")
                
                # Perform trajectory replanning
                result = self.replan_trajectory_to_target(
                    case=self.case,
                    target_time=target_time,
                    robot_id=self.robot_id
                )
                
                if result:
                    print(f"[{self.name}] Trajectory replanning successful!")
                    self.replanned_successfully = True
                else:
                    print(f"[{self.name}] Trajectory replanning failed!")
                    self.replanned_successfully = False
                    
                self.replanning_complete = True
                
            except Exception as e:
                print(f"[{self.name}] Error during replanning: {e}")
                self.replanning_complete = True
                self.replanned_successfully = False
        
        # Update progress feedback
        progress = min(elapsed / self.duration, 1.0)
        self.feedback_message = f"Replanning path... {progress*100:.1f}% complete"
        
        # Check if duration has elapsed
        if elapsed >= self.duration:
            if self.replanned_successfully:
                print(f"[{self.name}] Successfully replanned trajectory!")
                return py_trees.common.Status.SUCCESS
            else:
                print(f"[{self.name}] Trajectory replanning failed!")
                return py_trees.common.Status.FAILURE
                
        return py_trees.common.Status.RUNNING
    
    def replan_trajectory_to_target(self, case, target_time, robot_id):
        """
        Load existing trajectory and replan to achieve target time using optimization
        Works with existing tb{id}_Trajectory.json files
        """
        print(f"[{self.name}] Loading trajectory for Robot {robot_id}...")
        
        # Load existing trajectory
        trajectory_data = self.load_existing_trajectory(case, robot_id)
        
        if not trajectory_data:
            print(f"[{self.name}] No trajectory data found for Robot {robot_id}!")
            return False
        
        trajectory_points = trajectory_data.get('Trajectory', [])
        if not trajectory_points:
            print(f"[{self.name}] No trajectory points found for Robot {robot_id}!")
            return False
        
        # Calculate current trajectory duration
        current_duration = len(trajectory_points) * 0.1  # Assuming 0.1s time step
        
        print(f"[{self.name}] Current duration: {current_duration:.3f}s, Target: {target_time:.3f}s")
        print(f"[{self.name}] Original trajectory has {len(trajectory_points)} points")
        
        # Solve optimization problem to find optimal time allocation
        try:
            # Solve optimization problem to replan trajectory
            optimized_trajectory = self.solve_trajectory_optimization(
                trajectory_points, target_time
            )
            
            if not optimized_trajectory:
                print(f"[{self.name}] Failed to solve optimization problem")
                return False
            
            # Save replanned trajectory
            success = self.save_replanned_trajectory_direct(
                optimized_trajectory, case, robot_id, target_time
            )
            
            if success:
                print(f"[{self.name}] Trajectory replanning successful!")
                print(f"[{self.name}] Optimization: {current_duration:.3f}s → {target_time:.3f}s")
                print(f"[{self.name}] New trajectory has {len(optimized_trajectory)} points")
                return True
            else:
                print(f"[{self.name}] Failed to save replanned trajectory")
                return False
            
        except Exception as e:
            print(f"[{self.name}] Trajectory replanning failed: {e}")
            return False
    
    def solve_trajectory_optimization(self, original_trajectory, target_time):
        """
        Solve optimization problem to replan trajectory for target time
        Uses CasADi to solve nonlinear optimization problem
        """
        try:
            print(f"[{self.name}] Setting up optimization problem...")
            
            # Extract trajectory information
            n_points = len(original_trajectory)
            dt_original = 0.1  # Original time step
            current_duration = n_points * dt_original
            
            # Extract positions and velocities from original trajectory
            x_orig = [pt[0] for pt in original_trajectory]
            y_orig = [pt[1] for pt in original_trajectory] 
            theta_orig = [pt[2] for pt in original_trajectory]
            v_orig = [pt[3] for pt in original_trajectory]
            w_orig = [pt[4] for pt in original_trajectory]
            
            # Segment the trajectory for optimization
            n_segments = min(10, n_points // 5)  # Limit number of segments for computational efficiency
            segment_size = n_points // n_segments
            
            print(f"[{self.name}] Dividing trajectory into {n_segments} segments of size ~{segment_size}")
            
            # Setup CasADi optimization problem
            opti = ca.Opti()
            
            # Decision variables: time allocation for each segment
            segment_times = opti.variable(n_segments)
            
            # Constraints: all segment times must be positive and reasonable
            min_segment_time = 0.1  # Minimum time per segment
            max_segment_time = target_time * 0.7  # Maximum time per segment (prevent one segment taking all time)
            
            for i in range(n_segments):
                opti.subject_to(segment_times[i] >= min_segment_time)
                opti.subject_to(segment_times[i] <= max_segment_time)
            
            # Total time constraint
            opti.subject_to(ca.sum1(segment_times) == target_time)
            
            # Smoothness constraints: adjacent segments shouldn't differ too much
            max_time_ratio = 3.0  # Adjacent segments can differ by at most 3x
            for i in range(n_segments - 1):
                opti.subject_to(segment_times[i] <= max_time_ratio * segment_times[i+1])
                opti.subject_to(segment_times[i+1] <= max_time_ratio * segment_times[i])
            
            # Objective: minimize deviation from original velocity profile while maintaining smoothness
            cost = 0
            
            # Cost for deviating from original timing profile (weighted)
            original_segment_times = [current_duration / n_segments] * n_segments
            for i in range(n_segments):
                time_deviation = (segment_times[i] - original_segment_times[i]) ** 2
                cost += 0.5 * time_deviation  # Reduced weight to allow more flexibility
            
            # Smoothness cost: penalize large differences between adjacent segment times
            for i in range(n_segments - 1):
                smoothness_cost = (segment_times[i] - segment_times[i+1]) ** 2
                cost += 0.2 * smoothness_cost  # Increased smoothness weight
            
            # Velocity constraint cost: ensure velocities remain feasible
            max_velocity = 0.8  # m/s - increased maximum velocity
            max_angular_velocity = 1.5  # rad/s - increased maximum angular velocity
            
            for i in range(n_segments):
                # Calculate segment characteristics
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, n_points - 1)
                
                if start_idx < len(x_orig) and end_idx < len(x_orig):
                    # Calculate required velocity for this segment
                    segment_distance = math.sqrt(
                        (x_orig[end_idx] - x_orig[start_idx])**2 + 
                        (y_orig[end_idx] - y_orig[start_idx])**2
                    )
                    
                    if segment_distance > 0:
                        # Required velocity = distance / time
                        required_velocity = segment_distance / segment_times[i]
                        
                        # Soft constraint for velocity limits
                        velocity_violation = ca.fmax(0, required_velocity - max_velocity)
                        cost += 100 * velocity_violation**2  # High penalty for velocity violations
                        
                        # Add preference for reasonable velocities (not too slow either)
                        min_velocity = 0.05  # m/s minimum
                        slow_penalty = ca.fmax(0, min_velocity - required_velocity)
                        cost += 10 * slow_penalty**2
            
            # Set objective
            opti.minimize(cost)
            
            # Initial guess: proportional to original segment distances
            initial_times = []
            total_distance = 0
            segment_distances = []
            
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, n_points - 1)
                
                if start_idx < len(x_orig) and end_idx < len(x_orig):
                    segment_distance = math.sqrt(
                        (x_orig[end_idx] - x_orig[start_idx])**2 + 
                        (y_orig[end_idx] - y_orig[start_idx])**2
                    )
                    segment_distances.append(segment_distance)
                    total_distance += segment_distance
                else:
                    segment_distances.append(1.0)  # Default distance
                    total_distance += 1.0
            
            # Allocate time proportional to distance, but bounded
            for i in range(n_segments):
                if total_distance > 0:
                    proportional_time = (segment_distances[i] / total_distance) * target_time
                    # Bound the initial guess to reasonable values
                    bounded_time = max(min_segment_time, min(proportional_time, max_segment_time))
                    initial_times.append(bounded_time)
                else:
                    initial_times.append(target_time / n_segments)
            
            # Normalize to ensure total equals target_time
            time_sum = sum(initial_times)
            if time_sum > 0:
                initial_times = [t * target_time / time_sum for t in initial_times]
            else:
                initial_times = [target_time / n_segments] * n_segments
            
            opti.set_initial(segment_times, initial_times)
            print(f"[{self.name}] Initial time guess: {[f'{t:.3f}' for t in initial_times]}")
            
            # Configure solver
            opts = {
                'ipopt.max_iter': 500,
                'ipopt.tol': 1e-6,
                'ipopt.print_level': 0,
                'print_time': False,
                'ipopt.acceptable_tol': 1e-4,
                'ipopt.acceptable_obj_change_tol': 1e-6
            }
            opti.solver('ipopt', opts)
            
            print(f"[{self.name}] Solving optimization problem...")
            
            # Solve the optimization problem
            sol = opti.solve()
            
            # Extract optimal segment times
            optimal_times = sol.value(segment_times)
            
            print(f"[{self.name}] Optimization solved successfully!")
            print(f"[{self.name}] Optimal segment times: {[f'{t:.3f}' for t in optimal_times]}")
            
            # Generate new trajectory with optimal timing
            optimized_trajectory = self.generate_trajectory_from_optimal_times(
                original_trajectory, optimal_times, target_time
            )
            
            return optimized_trajectory
            
        except Exception as e:
            print(f"[{self.name}] Optimization failed: {e}")
            # Fallback to simple interpolation if optimization fails
            return self.create_simple_time_interpolated_trajectory(original_trajectory, target_time)
    
    def generate_trajectory_from_optimal_times(self, original_trajectory, optimal_times, target_time):
        """
        Generate new trajectory using optimal time allocation
        """
        try:
            n_points = len(original_trajectory)
            n_segments = len(optimal_times)
            segment_size = n_points // n_segments
            
            # New time step to achieve target duration
            dt_new = 0.1  # Keep same discretization
            n_new_points = int(target_time / dt_new)
            
            # Extract original trajectory data
            x_orig = np.array([pt[0] for pt in original_trajectory])
            y_orig = np.array([pt[1] for pt in original_trajectory])
            theta_orig = np.array([pt[2] for pt in original_trajectory])
            
            # Create cumulative time arrays for original and new trajectories
            original_times = np.arange(0, len(original_trajectory) * 0.1, 0.1)
            
            # Build new time array based on optimal segment times
            new_times = []
            current_time = 0.0
            
            for i, seg_time in enumerate(optimal_times):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, n_points)
                segment_points = end_idx - start_idx
                
                if segment_points > 0:
                    segment_dt = seg_time / segment_points
                    for j in range(segment_points):
                        new_times.append(current_time + j * segment_dt)
                    current_time += seg_time
            
            # Ensure we have the right number of time points
            while len(new_times) < n_points:
                new_times.append(new_times[-1] + dt_new)
            new_times = new_times[:n_points]
            
            # Create uniform output time grid
            output_times = np.arange(0, target_time, dt_new)
            
            # Interpolate trajectory to new timing
            from scipy.interpolate import interp1d
            
            # Create interpolation functions
            if len(new_times) >= 2 and len(original_times) >= 2:
                # Map from original times to new times
                time_mapping = interp1d(original_times[:len(new_times)], new_times, 
                                      kind='linear', fill_value='extrapolate')
                
                # Interpolate positions using the new timing
                interp_x = interp1d(new_times, x_orig[:len(new_times)], 
                                  kind='cubic', fill_value='extrapolate')
                interp_y = interp1d(new_times, y_orig[:len(new_times)], 
                                  kind='cubic', fill_value='extrapolate')
                interp_theta = interp1d(new_times, theta_orig[:len(new_times)], 
                                      kind='linear', fill_value='extrapolate')
                
                # Generate new trajectory
                new_trajectory = []
                for t in output_times:
                    x = float(interp_x(t))
                    y = float(interp_y(t))
                    theta = float(interp_theta(t))
                    
                    # Calculate velocities
                    if len(new_trajectory) > 0:
                        prev_x, prev_y, prev_theta = new_trajectory[-1][:3]
                        dt = dt_new
                        
                        # Linear velocity
                        v = math.sqrt((x - prev_x)**2 + (y - prev_y)**2) / dt
                        
                        # Angular velocity
                        dtheta = theta - prev_theta
                        while dtheta > math.pi:
                            dtheta -= 2 * math.pi
                        while dtheta < -math.pi:
                            dtheta += 2 * math.pi
                        w = dtheta / dt
                    else:
                        v = 0.0
                        w = 0.0
                    
                    new_trajectory.append([x, y, theta, v, w])
                
                print(f"[{self.name}] Generated optimized trajectory with {len(new_trajectory)} points")
                return new_trajectory
            else:
                print(f"[{self.name}] Insufficient points for interpolation")
                return self.create_simple_time_interpolated_trajectory(original_trajectory, target_time)
                
        except Exception as e:
            print(f"[{self.name}] Error generating trajectory from optimal times: {e}")
            return self.create_simple_time_interpolated_trajectory(original_trajectory, target_time)
    
    def create_simple_time_interpolated_trajectory(self, original_trajectory, target_time):
        """
        Fallback method: simple time interpolation of trajectory
        """
        try:
            dt_new = 0.1
            n_new_points = int(target_time / dt_new)
            n_orig_points = len(original_trajectory)
            
            if n_orig_points < 2:
                return original_trajectory
            
            # Create uniform interpolation
            new_trajectory = []
            for i in range(n_new_points):
                # Map new index to original trajectory
                orig_index = (i / (n_new_points - 1)) * (n_orig_points - 1)
                
                # Linear interpolation between adjacent points
                idx_low = int(orig_index)
                idx_high = min(idx_low + 1, n_orig_points - 1)
                alpha = orig_index - idx_low
                
                if idx_low < len(original_trajectory) and idx_high < len(original_trajectory):
                    pt_low = original_trajectory[idx_low]
                    pt_high = original_trajectory[idx_high]
                    
                    # Interpolate all components
                    x = pt_low[0] + alpha * (pt_high[0] - pt_low[0])
                    y = pt_low[1] + alpha * (pt_high[1] - pt_low[1])
                    theta = pt_low[2] + alpha * (pt_high[2] - pt_low[2])
                    v = pt_low[3] + alpha * (pt_high[3] - pt_low[3])
                    w = pt_low[4] + alpha * (pt_high[4] - pt_low[4])
                    
                    new_trajectory.append([x, y, theta, v, w])
            
            print(f"[{self.name}] Created fallback interpolated trajectory with {len(new_trajectory)} points")
            return new_trajectory
            
        except Exception as e:
            print(f"[{self.name}] Error in fallback trajectory creation: {e}")
            return original_trajectory

    def load_trajectory_parameters_individual(self, case, robot_ids):
        """Load trajectory parameters from individual robot trajectory parameter files"""
        data_dir = f'/root/workspace/data/{case}/'
        trajectory_data = {}
        
        for robot_id in robot_ids:
            robot_file = f'{data_dir}robot_{robot_id}_trajectory_parameters_{case}.json'
            
            if os.path.exists(robot_file):
                try:
                    with open(robot_file, 'r') as f:
                        data = json.load(f)
                    trajectory_data[f'robot{robot_id}'] = data
                    print(f"[{self.name}] Loaded trajectory parameters from {robot_file}")
                except Exception as e:
                    print(f"[{self.name}] Error loading {robot_file}: {e}")
            else:
                print(f"[{self.name}] Robot trajectory file not found: {robot_file}")
        
        return trajectory_data
    
    def save_replanned_trajectory(self, replanned_data, case, robot_id):
        """Save replanned trajectory parameters to file"""
        try:
            output_dir = f'/root/workspace/data/{case}/'
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = f'{output_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}.json'
            
            with open(output_file, 'w') as f:
                json.dump(replanned_data, f, indent=2)
            
            print(f"[{self.name}] Replanned trajectory saved: {output_file}")
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error saving replanned trajectory: {e}")
            return False
    
    def generate_discrete_trajectory_from_replanned(self, case, robot_id, dt=0.1):
        """Generate discrete trajectory from replanned trajectory parameters"""
        try:
            # Load replanned trajectory parameters
            data_dir = f'/root/workspace/data/{case}/'
            replanned_file = f'{data_dir}robot_{robot_id}_replanned_trajectory_parameters_{case}.json'
            
            if not os.path.exists(replanned_file):
                print(f"[{self.name}] Replanned trajectory file not found: {replanned_file}")
                return False
            
            with open(replanned_file, 'r') as f:
                replanned_data = json.load(f)
            
            # Extract trajectory components
            waypoints = replanned_data.get('waypoints', [])
            phi = replanned_data.get('phi', [])
            r0 = replanned_data.get('r0', [])
            l = replanned_data.get('l', [])
            phi_new = replanned_data.get('phi_new', [])
            time_segments = replanned_data.get('time_segments', [])
            Flagb = replanned_data.get('Flagb', [])
            
            if not all([waypoints, phi, r0, l, phi_new, time_segments, Flagb]):
                print(f"[{self.name}] Missing trajectory components in replanned data")
                return False
            
            # Generate discrete trajectory points
            trajectory_points = self.discretize_trajectory_from_parameters(
                waypoints, phi, r0, l, phi_new, time_segments, Flagb, dt
            )
            
            if not trajectory_points:
                print(f"[{self.name}] Failed to generate discrete trajectory points")
                return False
            
            # Save discrete trajectory in tb{robot_id}_Trajectory.json format
            discrete_file = f'{data_dir}tb{robot_id}_Trajectory_replanned.json'
            trajectory_data = {
                "Trajectory": trajectory_points
            }
            
            with open(discrete_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            print(f"[{self.name}] Discrete replanned trajectory saved: {discrete_file}")
            print(f"[{self.name}] Generated {len(trajectory_points)} trajectory points")
            
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error generating discrete trajectory: {e}")
            return False
    
    def discretize_trajectory_from_parameters(self, waypoints, phi, r0, l, phi_new, time_segments, Flagb, dt):
        """
        Discretize trajectory from parameters using cubic spline interpolation
        Returns list of trajectory points in format [x, y, theta, v, w]
        """
        try:
            # This is a simplified version - in a full implementation, you would need
            # to load the reeb graph and use the full discretization logic
            # For now, create a basic trajectory based on available parameters
            
            N = len(waypoints) - 1  # Number of segments
            all_times = []
            all_xs = []
            all_ys = []
            current_time = 0.0
            
            # Simple discretization - assuming linear interpolation between waypoints
            # In a full implementation, you would use the actual arc/line segments with proper geometry
            for i in range(N):
                if i < len(time_segments):
                    segment_time = 0.0
                    segment = time_segments[i]
                    
                    if 'arc' in segment and isinstance(segment['arc'], list):
                        segment_time += sum(segment['arc'])
                    if 'line' in segment and isinstance(segment['line'], list):
                        segment_time += sum(segment['line'])
                    
                    # Create time points for this segment
                    num_points = max(int(segment_time / dt), 1)
                    segment_times = np.linspace(0, segment_time, num_points)
                    
                    # Simple linear interpolation between waypoints
                    # In reality, you would use the actual arc/line geometry
                    for j, seg_time in enumerate(segment_times):
                        all_times.append(current_time + seg_time)
                        # Simple interpolation (would need actual waypoint coordinates)
                        all_xs.append(float(i + j / len(segment_times)))
                        all_ys.append(float(i * 0.5 + j * 0.1 / len(segment_times)))
                    
                    current_time += segment_time
            
            # Create uniform time grid
            total_time = all_times[-1] if all_times else 1.0
            t_uniform = np.arange(0, total_time, dt)
            
            if len(all_times) < 2:
                print(f"[{self.name}] Insufficient trajectory points for interpolation")
                return []
            
            # Interpolate positions using cubic spline
            cs_x = CubicSpline(all_times, all_xs)
            cs_y = CubicSpline(all_times, all_ys)
            
            x_uniform = cs_x(t_uniform)
            y_uniform = cs_y(t_uniform)
            
            # Calculate orientations and velocities
            trajectory_points = []
            for i in range(len(t_uniform)):
                if i < len(t_uniform) - 1:
                    # Calculate velocity
                    dx = x_uniform[i+1] - x_uniform[i]
                    dy = y_uniform[i+1] - y_uniform[i]
                    v = math.sqrt(dx*dx + dy*dy) / dt
                    
                    # Calculate orientation
                    theta = math.atan2(dy, dx)
                    
                    # Calculate angular velocity
                    if i > 0:
                        prev_theta = trajectory_points[i-1][2]
                        dtheta = theta - prev_theta
                        # Normalize angle difference
                        while dtheta > math.pi:
                            dtheta -= 2 * math.pi
                        while dtheta < -math.pi:
                            dtheta += 2 * math.pi
                        w = dtheta / dt
                    else:
                        w = 0.0
                else:
                    # Last point
                    v = 0.0
                    w = 0.0
                    if len(trajectory_points) > 0:
                        theta = trajectory_points[-1][2]
                    else:
                        theta = 0.0
                
                trajectory_points.append([
                    float(x_uniform[i]),
                    float(y_uniform[i]),
                    float(theta),
                    float(v),
                    float(w)
                ])
            
            return trajectory_points
            
        except Exception as e:
            print(f"[{self.name}] Error in trajectory discretization: {e}")
            return []
    

    def load_existing_trajectory(self, case, robot_id):
        """Load existing trajectory from tb{id}_Trajectory.json files"""
        try:
            # Try different possible locations for trajectory files
            possible_paths = [
                f'/root/workspace/data/{case}/tb{robot_id}_Trajectory.json'
            ]
            
            for trajectory_file in possible_paths:
                if os.path.exists(trajectory_file):
                    with open(trajectory_file, 'r') as f:
                        data = json.load(f)
                    print(f"[{self.name}] Loaded trajectory from {trajectory_file}")
                    return data
            
            print(f"[{self.name}] No trajectory file found for robot {robot_id} in any location")
            return None
            
        except Exception as e:
            print(f"[{self.name}] Error loading trajectory: {e}")
            return None
    
    
    def save_replanned_trajectory_direct(self, replanned_trajectory, case, robot_id, target_time):
        """Save replanned trajectory directly to JSON file"""
        try:
            # Create output directory if it doesn't exist
            output_dir = f'/root/workspace/data/{case}/'
            os.makedirs(output_dir, exist_ok=True)
            
            # Create replanned trajectory data structure
            trajectory_data = {
                "Trajectory": replanned_trajectory,
                "metadata": {
                    "replanned": True,
                    "target_time": target_time,
                    "original_points": len(replanned_trajectory),
                    "dt": 0.1,
                    "replanning_timestamp": time.time()
                }
            }
            
            # Save to file
            output_file = f'{output_dir}tb{robot_id}_Trajectory_replanned.json'
            with open(output_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            print(f"[{self.name}] Replanned trajectory saved to: {output_file}")
            print(f"[{self.name}] Trajectory contains {len(replanned_trajectory)} points")
            print(f"[{self.name}] Target duration: {target_time:.3f}s")
            
            return True
            
        except Exception as e:
            print(f"[{self.name}] Error saving replanned trajectory: {e}")
            return False


class StopSystem(py_trees.behaviour.Behaviour):
    """System stop behavior"""
    
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
    """Check if robot pair operation is complete"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def update(self):
        print(f"[{self.name}] Checking if pair is complete...")
        return py_trees.common.Status.SUCCESS


class IncrementIndex(py_trees.behaviour.Behaviour):
    """Increment current_parcel_index on blackboard"""
    
    def __init__(self, name, robot_namespace="turtlebot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        
        # Setup blackboard access for namespaced current_parcel_index
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.WRITE
        )
    
    def update(self):
        try:
            current_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
            setattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', current_index + 1)
            print(f"[{self.name}] Incremented {self.robot_namespace}/current_parcel_index to: {current_index + 1}")
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[{self.name}] Error incrementing index: {e}")
            return py_trees.common.Status.FAILURE


class PrintMessage(py_trees.behaviour.Behaviour):
    """Print custom message behavior"""
    
    def __init__(self, name, message):
        super().__init__(name)
        self.message = message
    
    def update(self):
        if callable(self.message):
            blackboard = py_trees.blackboard.Blackboard()
            print(self.message(blackboard))
        else:
            print(self.message)
        return py_trees.common.Status.SUCCESS


class MobileRobotMPC:
    """MPC controller for robot approach - based on State_switch.py implementation"""
    def __init__(self):
        # MPC parameters
        self.N = 10         # Extended prediction horizon for smoother approach
        self.dt = 0.1        # Time step
        self.wx = 2.0        # Increased position error weight for better position convergence
        self.wtheta = 1.5    # Increased orientation error weight for better alignment
        self.wu = 0.08       # Slightly reduced control effort weight for more responsive control
        
        # Control constraints - further reduced max velocity for even slower approach
        self.v_max = 0.02     # m/s (reduced from 0.025)
        self.w_max = 0.6      # rad/s (reduced from 0.8)
        
        # State and control dimensions
        self.nx = 3          # [x, y, theta]
        self.nu = 2          # [v, w]
        
        # Initialize CasADi optimizer
        self.setup_optimizer()
        
    def setup_optimizer(self):
        # Define symbolic variables
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N+1)
        self.U = self.opti.variable(self.nu, self.N)
        
        # Parameters (initial state and reference)
        self.x0 = self.opti.parameter(self.nx)
        self.x_ref = self.opti.parameter(self.nx)
        
        # Cost function
        cost = 0
        for k in range(self.N):
            # Tracking cost with increasing weights as we approach the end
            # Stronger progression factor for more aggressive convergence
            progress_factor = 1.0 + 2.0 * (k + 1) / self.N  # Increases from 1.0 to 3.0
            
            # Position cost - higher weight for xy position tracking
            pos_error = self.X[:2, k] - self.x_ref[:2]
            cost += progress_factor * self.wx * ca.sumsqr(pos_error)
            
            # Orientation cost with angle normalization to handle wraparound
            theta_error = self.X[2, k] - self.x_ref[2]
            # Normalize angle difference to [-pi, pi]
            theta_error = ca.fmod(theta_error + ca.pi, 2*ca.pi) - ca.pi
            cost += progress_factor * self.wtheta * theta_error**2
            
            # Special emphasis on final portion of trajectory
            if k >= self.N - 5:  # Last 5 steps
                # Extra emphasis on final approach
                cost += 1.5 * self.wx * ca.sumsqr(pos_error)
                cost += 2.0 * self.wtheta * theta_error**2
            
            # Control effort cost with smoother transitions
            if k > 0:
                # Penalize control changes for smoother motion
                control_change = self.U[:, k] - self.U[:, k-1]
                cost += 0.1 * ca.sumsqr(control_change)
            
            # Base control effort penalty
            cost += self.wu * ca.sumsqr(self.U[:, k])
        
        # Terminal cost - much stronger to ensure convergence at endpoint
        terminal_pos_error = self.X[:2, self.N] - self.x_ref[:2]
        cost += 10.0 * self.wx * ca.sumsqr(terminal_pos_error)
        
        # Terminal orientation with normalization for angle wraparound
        terminal_theta_error = self.X[2, self.N] - self.x_ref[2]
        terminal_theta_error = ca.fmod(terminal_theta_error + ca.pi, 2*ca.pi) - ca.pi
        cost += 30.0 * self.wtheta * terminal_theta_error**2
        self.opti.minimize(cost)
        
        # Dynamics constraints
        for k in range(self.N):
            x_next = self.X[:, k] + self.robot_model(self.X[:, k], self.U[:, k]) * self.dt
            self.opti.subject_to(self.X[:, k+1] == x_next)
        
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # Control constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max, self.U[0, :], self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.w_max, self.U[1, :], self.w_max))
        
        # Strict terminal constraints to ensure convergence and smooth stopping
        # Terminal velocity constraints - must approach zero at the end
        self.opti.subject_to(self.U[0, -1] <= 0.0005)  # Final linear velocity virtually zero
        self.opti.subject_to(self.U[0, -1] >= 0.0)     # No negative velocity at end
        self.opti.subject_to(self.U[1, -1] <= 0.0005)  # Final angular velocity virtually zero
        self.opti.subject_to(self.U[1, -1] >= -0.0005) # Final angular velocity virtually zero
        
        # Smooth deceleration in last few steps
        for k in range(self.N-3, self.N):
            # Progressive velocity reduction for final steps
            max_vel_factor = (self.N - k) / 4.0  # Ranges from 0.75 to 0.25
            self.opti.subject_to(self.U[0, k] <= self.v_max * max_vel_factor)
        
        # Solver settings with improved convergence parameters
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.tol': 1e-5,           # Even tighter tolerance
            'ipopt.acceptable_tol': 1e-4, # More precise solution
            'ipopt.max_iter': 200,       # More iterations allowed
            'ipopt.warm_start_init_point': 'yes' # Use warm starting for stability
        }
        self.opti.solver('ipopt', opts)
        
    def robot_model(self, x, u):
        # Differential drive kinematics
        dx = ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1]
        )
        return dx
        
    def update_control(self, current_state, target_state):
        # Check how close we are to the target
        dist_to_target = np.sqrt((current_state[0] - target_state[0])**2 + 
                                (current_state[1] - target_state[1])**2)
        
        # Check orientation alignment
        angle_diff = abs((current_state[2] - target_state[2] + np.pi) % (2 * np.pi) - np.pi)
        
        # If we're very close to the target and well-aligned, stop completely
        if dist_to_target < 0.015 and angle_diff < 0.05:  # 1.5cm and ~3 degrees
            return np.array([0.0, 0.0])  # Stop completely
            
        # If we're close but not perfectly aligned, prioritize orientation
        elif dist_to_target < 0.03 and angle_diff > 0.05:
            # Just rotate to align with target, very slowly
            return np.array([0.0, 0.1 * np.sign(target_state[2] - current_state[2])])
        
        # Set initial state and reference
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.x_ref, target_state)
        
        # Solve optimization problem
        try:
            sol = self.opti.solve()
            x_opt = sol.value(self.X)
            u_opt = sol.value(self.U)
            return u_opt[:, 0]  # Return first control input
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None


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
            
            # Create command velocity publisher
            self.cmd_vel_pub = self.ros_node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            
            # Subscribe to robot pose (Odometry)
            self.robot_pose_sub = self.ros_node.create_subscription(
                Odometry, f'/turtlebot{self.namespace_number}/odom_map',
                self.robot_pose_callback, 10)
            
            # Subscribe to current parcel index
            self.current_index_sub = self.ros_node.create_subscription(
                Int32, f'/{self.robot_namespace}/current_parcel_index',
                self.current_index_callback, 10)
            
            # Initial parcel subscription (will be updated based on current index)
            self.update_parcel_subscription()
            
            self.ros_node.get_logger().info(
                f'ApproachObject setup complete for {self.robot_namespace}')
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
        
        # Subscribe to current parcel topic
        parcel_topic = f'/parcel{self.current_parcel_index}/pose'
        self.parcel_pose_sub = self.ros_node.create_subscription(
            PoseStamped, parcel_topic, self.parcel_pose_callback, 10)
        
        self.ros_node.get_logger().info(f'Updated parcel subscription to: {parcel_topic}')

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