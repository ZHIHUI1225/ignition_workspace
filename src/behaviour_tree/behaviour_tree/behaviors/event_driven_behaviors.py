#!/usr/bin/env python3
"""
Event-driven behavior implementations.
These behaviors are optimized to reduce CPU usage by minimizing polling
and using an event-driven approach.
"""

import py_trees
import math
import time
from rclpy.node import Node
import threading

# Import from existing behaviors
from .basic_behaviors import WaitForPush

# Add missing ROS message imports
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class EventDrivenWaitForPush(WaitForPush):
    """
    Event-driven version of WaitForPush behavior with threading locks and event queues.
    
    This version only checks when the parcel is within range of the relay point.
    Uses event-driven approach with proper thread safety but simplified logic.
    """
    
    def __init__(self, name, duration=60.0, robot_namespace="robot0", 
                 distance_threshold=0.14):
        # Initialize the parent class
        super().__init__(name, duration, robot_namespace, distance_threshold)
        
        # Use RLock to allow recursive locking in nested methods
        self.state_lock = threading.RLock()
        
        # Initialize event-driven state variables
        self._init_event_driven_state()
        
        # Set case name for trajectory files
        self.case_name = "experi"  # Default case name
        
        print(f"[{self.name}] Event-driven WaitForPush created with threading")
        
    def _init_event_driven_state(self):
        """Initialize state variables for event-driven behavior"""
        # Event flags to track condition changes
        self.distance_changed = False
        
        # Cache for distance calculations
        self.prev_parcel_relay_distance = float('inf')
        self.last_distance_to_threshold = float('inf')
        
        # Cached condition checks to avoid recalculation when nothing changed
        self.cached_parcel_in_range = False
        
        # Track last check time for rate limiting
        self.last_check_time = 0.0
        self.min_check_interval = 0.2  # Minimum 200ms between checks to reduce CPU usage
        
        # Store previous success state to detect changes
        self.previous_success_state = False
    
    def parcel_pose_callback(self, msg):
        """Override callback to detect threshold-crossing changes in distance"""
        with self.state_lock:
            # Store original pose first
            previous_pose = self.parcel_pose
            
            # Convert Odometry message to a simple pose for compatibility
            self.parcel_pose = msg  # Store the full Odometry message
            
            # Log first receipt of data
            if previous_pose is None:
                try:
                    print(f"[{self.name}] ✅ Received first parcel pose data: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})")
                except AttributeError as e:
                    print(f"[{self.name}] ✅ Received first parcel pose data (error accessing position: {e})")
                    print(f"[{self.name}] DEBUG: Message type: {type(msg)}, attributes: {dir(msg)}")
            
            # Check if relay pose is available yet
            if self.relay_pose is None:
                print(f"[{self.name}] DEBUG: Relay pose not available yet, skipping distance calculation")
                return
                
            # Calculate distance between parcel and relay point
            try:
                current_distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
                print(f"[{self.name}] DEBUG: Calculated parcel-relay distance: {current_distance:.3f}m")
                
                # Use the unified distance event detection
                if self._check_distance_event(current_distance, "Parcel-relay"):
                    self.distance_changed = True
                    print(f"[{self.name}] DEBUG: Distance change event triggered")
            except Exception as e:
                print(f"[{self.name}] ERROR: Failed to calculate distance in callback: {e}")
                import traceback
                traceback.print_exc()

    def should_check_conditions(self):
        """Determine if conditions should be checked based on event triggers"""
        current_time = time.time()
        time_since_last_check = current_time - self.last_check_time
        
        with self.state_lock:
            # Simple check frequency - check when distance changed or forced check
            forced_check = time_since_last_check > 3.0
            min_interval = 0.3  # 300ms minimum interval
            
            # Check if distance changed and minimum interval passed
            should_check = ((self.distance_changed) and 
                            time_since_last_check > min_interval) or forced_check
            
            # Reset flags if we're going to check
            if should_check:
                self.distance_changed = False
                self.last_check_time = current_time
            
            return should_check
    
    def update(self) -> py_trees.common.Status:
        """
        Event-driven update implementation.
        Only performs full condition check when positions have changed.
        """
        if self._check_timeout():
            return py_trees.common.Status.FAILURE
        
        if not self.should_check_conditions():
            return self._get_cached_status()
            
        return self._perform_full_check()
    
    def _check_timeout(self):
        """Check if the behavior has timed out"""
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            from .tree_builder import report_node_failure
            error_msg = f"WaitForPush timeout after {elapsed:.1f}s - parcel never reached relay point"
            report_node_failure(self.name, error_msg, self.robot_namespace)
            print(f"[{self.name}] FAILURE: {error_msg}")
            return True
        return False
    
    def _format_position(self, pose):
        """Format a pose object for debug output"""
        if pose is None:
            return "Unknown"
        
        try:
            # Handle different message types
            if hasattr(pose, 'pose'):
                # Could be Odometry message
                if hasattr(pose.pose, 'pose'):
                    return f"({pose.pose.pose.position.x:.2f}, {pose.pose.pose.position.y:.2f})"
                # Could be PoseStamped message
                elif hasattr(pose.pose, 'position'):
                    return f"({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})"
            # Direct pose object
            elif hasattr(pose, 'position'):
                return f"({pose.position.x:.2f}, {pose.position.y:.2f})"
            else:
                return f"Unknown pose type: {type(pose)}"
        except AttributeError as e:
            return f"Error: {e}"
        
    def _get_distance_info(self):
        """Get formatted distance information for debug output"""
        if self.parcel_pose is None or self.relay_pose is None:
            return "Distance: Unknown"
        
        distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
        return f"Distance: {distance:.3f}m (threshold: {self.distance_threshold:.3f}m)"
    
    def _get_cached_status(self):
        """Return status based on cached values without full recalculation"""
        with self.state_lock:
            # If no initial distance data yet
            if self.prev_parcel_relay_distance == float('inf'):
                self.feedback_message = f"[{self.robot_namespace}] Waiting for initial distance data..."
                
                # Add diagnostic info every ~5 seconds
                elapsed = time.time() - self.start_time
                if int(elapsed) % 5 == 0:
                    print(f"[{self.name}] Waiting for initial data, elapsed time: {elapsed:.1f}s")
                return py_trees.common.Status.RUNNING
                
            # If we've already determined success, return that
            if self.previous_success_state:
                return py_trees.common.Status.SUCCESS
                
            # Otherwise provide feedback with position information
            parcel_pos = self._format_position(self.parcel_pose)
            relay_pos = self._format_position(self.relay_pose)
            distance_info = self._get_distance_info()
            
            self.feedback_message = (
                f"[{self.robot_namespace}] PUSH wait for parcel{self.current_parcel_index} -> relay{self.relay_number}...\n"
                f"Parcel pos: {parcel_pos}, Relay pos: {relay_pos}, {distance_info}"
            )
        
        return py_trees.common.Status.RUNNING
        
    def _perform_full_check(self):
        """Perform a full condition check when events trigger a reevaluation"""
        with self.state_lock:
            elapsed = time.time() - self.start_time
            
            # Only log condition checks every ~5 seconds to reduce CPU usage
            if int(elapsed) % 5 == 0:
                print(f"[{self.name}] ⚡ Performing event-driven condition check at elapsed: {elapsed:.1f}s")
            
            # Simple check: is parcel within range of relay point?
            if not self._check_parcel_position():
                return py_trees.common.Status.RUNNING
            
            # Success! Parcel is in range
            print(f"[{self.name}] ⚡ Check result: Parcel is within range, returning SUCCESS")
            print(f"[{self.name}] SUCCESS: Ready to proceed with pushing!")
            self.previous_success_state = True
            
            # Immediately start cleanup when success is determined
            self._cleanup_on_success()
            
            return py_trees.common.Status.SUCCESS
        
    def _check_parcel_position(self):
        """Check if parcel is within range of relay point"""
        parcel_in_range = self.check_parcel_in_relay_range()
        self.cached_parcel_in_range = parcel_in_range
        
        # Only log every ~10 seconds to reduce CPU usage
        elapsed = time.time() - self.start_time
        log_now = int(elapsed) % 10 == 0
        
        if not parcel_in_range:
            if log_now:
                print(f"[{self.name}] ⚡ Check result: Still waiting for parcel to move within range of relay point")
            self.previous_success_state = False
            return False
        return True
    
    def _cleanup_on_success(self):
        """Perform immediate cleanup when behavior succeeds to release resources"""
        print(f"[{self.name}] 🧹 Starting immediate cleanup on success...")
        
        # Mark as terminated to prevent further callbacks
        self._terminated = True
        
        # Stop all event processing
        self.distance_changed = False
        
        try:
            # Clean up subscriptions immediately
            if hasattr(self, 'robot_pose_sub') and self.robot_pose_sub:
                self.node.destroy_subscription(self.robot_pose_sub)
                self.robot_pose_sub = None
                print(f"[{self.name}] ✅ Robot pose subscription cleaned up")
                
            if hasattr(self, 'parcel_pose_sub') and self.parcel_pose_sub:
                self.node.destroy_subscription(self.parcel_pose_sub)
                self.parcel_pose_sub = None
                print(f"[{self.name}] ✅ Parcel pose subscription cleaned up")
                
        except Exception as e:
            print(f"[{self.name}] ⚠️ Warning during cleanup: {e}")
            
        print(f"[{self.name}] 🧹 Immediate cleanup completed")
    
    def _check_distance_event(self, current_distance, position_label):
        """
        Unified distance event detection for threshold crossing
        
        Args:
            current_distance (float): The current calculated distance
            position_label (str): Label for the position being checked (for logging)
            
        Returns:
            bool: True if the distance has crossed the threshold, False otherwise
        """
        with self.state_lock:
            # Calculate how far we are from the threshold boundary (positive = outside, negative = inside)
            distance_to_threshold = current_distance - self.distance_threshold
            
            # Check if we've crossed the threshold boundary
            if self.prev_parcel_relay_distance != float('inf'):
                # Check if we crossed the threshold (sign change)
                threshold_crossed = (distance_to_threshold * self.last_distance_to_threshold) <= 0  # Sign change
                
                if threshold_crossed:
                    elapsed = time.time() - self.start_time
                    if int(elapsed) % 5 == 0:  # Only log every 5 seconds
                        print(f"[{self.name}] ⚡ Event: {position_label} distance threshold crossing detected, "
                              f"Distance: {current_distance:.3f}m, Threshold: {self.distance_threshold:.3f}m")
                    
                    # Update cached values
                    self.prev_parcel_relay_distance = current_distance
                    self.last_distance_to_threshold = distance_to_threshold
                    return True
            
            # Update cached values even if no significant change
            self.prev_parcel_relay_distance = current_distance
            self.last_distance_to_threshold = distance_to_threshold
            return False
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        print(f"[{self.name}] 🔧 TERMINATING: Cleaning up subscriptions, status: {new_status}")
        
        with self.state_lock:
            # Mark as terminated to prevent further callbacks
            self._terminated = True
            
            # If we haven't already cleaned up on success, do it now
            if new_status == py_trees.common.Status.SUCCESS:
                print(f"[{self.name}] 🎯 SUCCESS termination - ensuring cleanup is complete")
                
            # Comprehensive cleanup (idempotent - safe to call multiple times)
            try:
                if hasattr(self, 'robot_pose_sub') and self.robot_pose_sub:
                    self.node.destroy_subscription(self.robot_pose_sub)
                    self.robot_pose_sub = None
                    
                if hasattr(self, 'parcel_pose_sub') and self.parcel_pose_sub:
                    self.node.destroy_subscription(self.parcel_pose_sub)
                    self.parcel_pose_sub = None
                    
            except Exception as e:
                print(f"[{self.name}] ⚠️ Warning during termination cleanup: {e}")
        
        # Call parent terminate
        super().terminate(new_status)
        
        print(f"[{self.name}] ✅ Termination complete")

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
            print(f"[{self.name}] ✅ Using robot dedicated callback group: {id(self.callback_group)}")
        else:
            print(f"[{self.name}] ❌ ERROR: No shared callback manager found, cannot use shared callback groups")
            return False
        
        # Setup ROS subscriptions now that we have a node
        self.setup_subscriptions()
        
        # 🔧 NEW: Load relay point from trajectory file immediately during setup
        self.load_relay_point_from_trajectory()
        
        # Call parent setup
        return super().setup(**kwargs)
    
    def load_relay_point_from_trajectory(self):
        """Load relay point from trajectory file"""
        try:
            # Use the function from basic_behaviors
            from .basic_behaviors import load_relay_point_from_trajectory
            
            success, relay_pose_msg = load_relay_point_from_trajectory(
                robot_namespace=self.robot_namespace,
                node=self.node,
                case_name=self.case_name
            )
            
            if success and relay_pose_msg:
                self.relay_pose = relay_pose_msg
                print(f"[{self.name}] ✅ Successfully loaded relay point: ({relay_pose_msg.pose.position.x:.3f}, {relay_pose_msg.pose.position.y:.3f})")
                return True
            else:
                print(f"[{self.name}] ❌ Failed to load relay point from trajectory")
                return False
                
        except Exception as e:
            print(f"[{self.name}] ❌ Error loading relay point: {e}")
            return False
    
    def setup_parcel_subscription(self):
        """Set up parcel subscription when blackboard is ready"""
        if self.node is None:
            print(f"[{self.name}] WARNING: Cannot setup parcel subscription - no ROS node")
            return False
            
        try:
            # Get current parcel index from blackboard (with safe fallback)
            try:
                current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
                print(f"[{self.name}] Retrieved parcel index from blackboard: {current_parcel_index}")
            except Exception as bb_error:
                # Blackboard key doesn't exist yet - use default
                print(f"[{self.name}] INFO: Blackboard key not ready, using default parcel index 0: {bb_error}")
                current_parcel_index = 0
            
            self.current_parcel_index = current_parcel_index
            
            # Clean up existing subscription if it exists
            if self.parcel_pose_sub is not None:
                self.node.destroy_subscription(self.parcel_pose_sub)
                print(f"[{self.name}] Destroyed existing parcel subscription")
                
            # Create new parcel subscription with callback group for thread isolation
            parcel_topic = f'/parcel{current_parcel_index}/odom'
            if self.callback_group is not None:
                self.parcel_pose_sub = self.node.create_subscription(
                    Odometry,  # Now properly imported
                    parcel_topic,
                    self.parcel_pose_callback,
                    10,
                    callback_group=self.callback_group
                )
                print(f"[{self.name}] ✓ Successfully subscribed to {parcel_topic} with callback group")
            else:
                self.parcel_pose_sub = self.node.create_subscription(
                    Odometry,  # Now properly imported
                    parcel_topic,
                    self.parcel_pose_callback,
                    10
                )
                print(f"[{self.name}] ✓ Successfully subscribed to {parcel_topic} without callback group")
                
            # Verify subscription was created successfully
            if self.parcel_pose_sub is None:
                print(f"[{self.name}] ❌ ERROR: Failed to create parcel subscription!")
                return False
                
            print(f"[{self.name}] ✅ Parcel subscription verification: topic={parcel_topic}, sub_obj={self.parcel_pose_sub is not None}")
            return True
            
        except Exception as e:
            print(f"[{self.name}] ERROR: Failed to setup parcel subscription: {e}")
            import traceback
            traceback.print_exc()
            return False

    def initialise(self):
        """Initialize the behavior when it starts running"""
        print(f"[{self.name}] =================== INITIALISE START ===================")
        
        # Reset state variables every time behavior launches
        self.start_time = time.time()
        self._init_event_driven_state()
        
        # 🔧 CRITICAL: Ensure relay point is loaded
        if self.relay_pose is None:
            print(f"[{self.name}] Relay pose not loaded, attempting to load from trajectory...")
            if not self.load_relay_point_from_trajectory():
                print(f"[{self.name}] ❌ CRITICAL: Failed to load relay point from trajectory!")
            else:
                print(f"[{self.name}] ✅ Relay point loaded successfully during initialization")
        
        # 🔧 CRITICAL: Set up parcel subscription now that blackboard should be ready
        if not self.setup_parcel_subscription():
            print(f"[{self.name}] ❌ WARNING: Failed to setup parcel subscription")
        else:
            # Verify the subscription is working
            self._verify_parcel_subscription()
        
        self.feedback_message = f"[{self.robot_namespace}] Event-driven PUSH wait for parcel{self.current_parcel_index} -> relay{self.relay_number}"
        
        print(f"[{self.name}] =================== INITIALISE COMPLETE ===================")
        print(f"[{self.name}] Starting event-driven PUSH wait...")
        print(f"[{self.name}] Monitoring: parcel{self.current_parcel_index} -> relay{self.relay_number}")
        print(f"[{self.name}] Relay point available: {self.relay_pose is not None}")
        print(f"[{self.name}] Parcel subscription active: {self.parcel_pose_sub is not None}")

    def _verify_parcel_subscription(self):
        """Verify that the parcel subscription is properly set up"""
        if not self.node:
            return
            
        parcel_topic = f'/parcel{self.current_parcel_index}/odom'
        
        try:
            # Check if topic exists
            topic_names_and_types = self.node.get_topic_names_and_types()
            available_topics = [name for name, _ in topic_names_and_types]
            
            topic_exists = parcel_topic in available_topics
            pub_count = self.node.count_publishers(parcel_topic) if topic_exists else 0
            sub_count = self.node.count_subscribers(parcel_topic) if topic_exists else 0
            
            print(f"[{self.name}] 🔍 Parcel subscription verification:")
            print(f"[{self.name}]   - Topic: {parcel_topic}")
            print(f"[{self.name}]   - Topic exists: {topic_exists}")
            print(f"[{self.name}]   - Publishers: {pub_count}")
            print(f"[{self.name}]   - Subscribers: {sub_count}")
            print(f"[{self.name}]   - Our subscription object: {self.parcel_pose_sub is not None}")
            
            if not topic_exists:
                print(f"[{self.name}] ❌ ERROR: Parcel topic {parcel_topic} does not exist!")
            elif pub_count == 0:
                print(f"[{self.name}] ❌ ERROR: No publishers for {parcel_topic}!")
            elif self.parcel_pose_sub is None:
                print(f"[{self.name}] ❌ ERROR: Our subscription object is None!")
            else:
                print(f"[{self.name}] ✅ Parcel subscription looks good")
                
        except Exception as e:
            print(f"[{self.name}] ❌ ERROR: Failed to verify parcel subscription: {e}")

def create_event_driven_wait_for_push(name, duration=60.0, robot_namespace="robot0", 
                                     distance_threshold=0.14):
    """
    Factory function to create an EventDrivenWaitForPush node.
    
    Args:
        name (str): Name of the behavior node
        duration (float): Maximum duration to wait before timing out
        robot_namespace (str): Robot namespace (e.g., 'robot0')
        distance_threshold (float): Distance threshold for success condition
        
    Returns:
        py_trees.behaviour.Behaviour: The event-driven wait behavior
    """
    return EventDrivenWaitForPush(
        name=name,
        duration=duration,
        robot_namespace=robot_namespace,
        distance_threshold=distance_threshold
    )