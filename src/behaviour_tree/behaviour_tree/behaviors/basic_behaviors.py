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
    """Check if a pair operation is complete behavior"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def update(self):
        print(f"[{self.name}] Checking if pair is complete...")
        return py_trees.common.Status.SUCCESS


class IncrementIndex(py_trees.behaviour.Behaviour):
    """Increment index behavior - increments the current parcel index"""
    
    def __init__(self, name, robot_namespace="turtlebot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        
        # Setup blackboard access
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.WRITE
        )
    
    def update(self):
        try:
            # Get current index
            current_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
            new_index = current_index + 1
            
            # Update the index
            setattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', new_index)
            
            print(f"[{self.name}] Incremented index from {current_index} to {new_index}")
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[{self.name}] Error incrementing index: {e}")
            return py_trees.common.Status.FAILURE


class PrintMessage(py_trees.behaviour.Behaviour):
    """Print message behavior - prints a message or lambda function result"""
    
    def __init__(self, name, message):
        super().__init__(name)
        self.message = message
    
    def update(self):
        try:
            if callable(self.message):
                blackboard = py_trees.blackboard.Blackboard()
                print(self.message(blackboard))
            else:
                print(self.message)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[{self.name}] Error printing message: {e}")
            return py_trees.common.Status.FAILURE