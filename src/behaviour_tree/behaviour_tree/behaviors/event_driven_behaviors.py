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


class EventDrivenWaitForPush(WaitForPush):
    """
    Event-driven version of WaitForPush behavior.
    
    This version only performs full condition checks when:
    1. The distance between parcel and relay point changes relative to the threshold
       (crosses the threshold boundary in either direction)
    2. The pushing_finished status of the previous robot changes
    3. For non-turtlebot0 robots: Last robot's position relative to relay point changes
    
    Instead of checking on every behavior tree tick (2Hz polling),
    it only updates when these critical conditions change.
    """
    
    def __init__(self, name, duration=60.0, robot_namespace="turtlebot0", 
                 distance_threshold=0.14):
        # Initialize the parent class
        super().__init__(name, duration, robot_namespace, distance_threshold)
        
        # Use RLock to allow recursive locking in nested methods
        self.state_lock = threading.RLock()
        
        # Initialize event-driven state variables
        self._init_event_driven_state()
        
        print(f"[{self.name}] Event-driven WaitForPush created")
        
    def _init_event_driven_state(self):
        """Initialize state variables for event-driven behavior"""
        # Event flags to track condition changes
        self.distance_changed = False
        self.status_changed = False
        
        # Cache for distance calculations
        self.prev_parcel_relay_distance = float('inf')
        self.prev_last_robot_relay_distance = float('inf')
        self.last_distance_to_threshold = float('inf')
        
        # Cached condition checks to avoid recalculation when nothing changed
        self.cached_parcel_in_range = False
        self.cached_last_robot_out = True  # Default to True for turtlebot0
        self.cached_previous_finished = True if self.is_first_robot else False
        
        # Track last check time for rate limiting
        self.last_check_time = 0.0
        self.min_check_interval = 0.2  # Minimum 200ms between checks to reduce CPU usage
        
        # Store previous success state to detect changes
        self.previous_success_state = False
    
    def parcel_pose_callback(self, msg):
        """Override callback to detect threshold-crossing changes in distance"""
        with self.state_lock:
            # Store original pose first
            self.parcel_pose = msg
            
            # For non-first robots, skip costly calculations if previous robot hasn't finished pushing
            if not self.is_first_robot and not self.cached_previous_finished:
                return
            
            # Check if relay pose is available yet
            if self.relay_pose is None:
                return
                
            # Calculate distance between parcel and relay point
            current_distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
            
            # Use the unified distance event detection
            if self._check_distance_event(current_distance, "Parcel-relay"):
                self.distance_changed = True
    
    def relay_pose_callback(self, msg):
        """Override callback to detect threshold-crossing changes in distance"""
        with self.state_lock:
            # Store original pose first
            self.relay_pose = msg
            
            # Check if parcel pose is available yet
            if self.parcel_pose is None:
                return
            
            # Calculate distance between parcel and relay point
            current_distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
            
            # Use the unified distance event detection
            if self._check_distance_event(current_distance, "Relay-parcel"):
                self.distance_changed = True
    
    def last_robot_pose_callback(self, msg):
        """Override callback to detect if last robot has moved far enough from parcel"""
        with self.state_lock:
            # Skip if this is turtlebot0 (no previous robot)
            if self.is_first_robot:
                return
                
            # Store original pose first 
            self.last_robot_pose = msg.pose.pose
            
            # Skip costly calculations if previous robot hasn't finished pushing
            if not self.cached_previous_finished:
                return
                
            # Check if parcel pose is available yet
            if self.parcel_pose is None:
                return
            
            # Calculate distance between last robot and parcel
            current_distance = self.calculate_distance(self.last_robot_pose, self.parcel_pose)
            
            # Special handling for last robot (we care about far/near status specifically)
            if self.prev_last_robot_relay_distance != float('inf'):
                was_far = self.prev_last_robot_relay_distance > self.distance_threshold
                is_far = current_distance > self.distance_threshold
                
                # If far/near status changed, flag for check
                if was_far != is_far:
                    status = "FAR from" if is_far else "NEAR"
                    self._trigger_event(f"Last robot moved {status} parcel",
                                      f"Distance: {current_distance:.3f}m, Threshold: {self.distance_threshold:.3f}m")
                    self.distance_changed = True
            
            # Update previous distance (reusing the same variable)
            self.prev_last_robot_relay_distance = current_distance
    
    def previous_robot_pushing_finished_callback(self, msg):
        """Override callback for previous robot's pushing finished status to detect changes"""
        with self.state_lock:
            old_value = self.previous_robot_pushing_finished
            # Update value from parent class method
            super().previous_robot_pushing_finished_callback(msg)
            
            # Detect status change
            if old_value != self.previous_robot_pushing_finished:
                previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
                self._trigger_event(f"{previous_robot_namespace}/pushing_finished changed to {msg.data}")
                self.status_changed = True
    
    def should_check_conditions(self):
        """Determine if conditions should be checked based on event triggers"""
        current_time = time.time()
        time_since_last_check = current_time - self.last_check_time
        
        with self.state_lock:
            # For non-first robots waiting for previous robot, reduce check frequency significantly
            if not self.is_first_robot and not self.cached_previous_finished:
                # Check much less frequently if waiting for previous robot (5 seconds)
                forced_check = time_since_last_check > 5.0
                # Only check when status changes (previous robot finished)
                should_check = (self.status_changed and time_since_last_check > 0.5) or forced_check
            else:
                # Normal check frequency for first robot or when previous robot is finished
                forced_check = time_since_last_check > 3.0
                # Increase minimum interval to reduce checking frequency
                min_interval = 0.3  # Increased to 300ms
                
                # Check if distance relationship or status changed and minimum interval passed
                should_check = ((self.distance_changed or self.status_changed) and 
                                time_since_last_check > min_interval) or forced_check
            
            # Reset flags if we're going to check
            if should_check:
                self.distance_changed = False
                self.status_changed = False
                self.last_check_time = current_time
            
            return should_check
    
    def update(self) -> py_trees.common.Status:
        """
        Event-driven update implementation.
        Only performs full condition check when positions or statuses have changed.
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
            error_msg = f"WaitForPush timeout after {elapsed:.1f}s - previous robot coordination failed"
            report_node_failure(self.name, error_msg, self.robot_namespace)
            print(f"[{self.name}] FAILURE: {error_msg}")
            return True
        return False
    
    def _get_cached_status(self):
        """Return status based on cached values without full recalculation"""
        # If no initial distance data yet
        if self.prev_parcel_relay_distance == float('inf'):
            self.feedback_message = f"[{self.robot_namespace}] Waiting for initial distance data..."
            print(f"[{self.name}] Waiting for initial data...")
            return py_trees.common.Status.RUNNING
            
        # Use cached results for feedback
        previous_finished = self.cached_previous_finished
        last_robot_far_enough = self.cached_last_robot_out  # Same variable, new meaning
        
        # If we've already determined success, return that
        if self.previous_success_state:
            return py_trees.common.Status.SUCCESS
            
        # Otherwise provide appropriate feedback message based on what we're waiting for
        if not previous_finished:
            previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
            self.feedback_message = f"[{self.robot_namespace}] Waiting for {previous_robot_namespace} to finish pushing..."
        elif not self.is_first_robot and not last_robot_far_enough:
            self.feedback_message = f"[{self.robot_namespace}] Waiting for tb{self.last_robot_number} to move away from parcel..."
        else:
            self.feedback_message = f"[{self.robot_namespace}] PUSH wait for parcel{self.current_parcel_index} -> relay{self.relay_number}..."
        
        return py_trees.common.Status.RUNNING
        
    def _perform_full_check(self):
        """Perform a full condition check when events trigger a reevaluation"""
        elapsed = time.time() - self.start_time
        
        # Only log condition checks every ~5 seconds to reduce CPU usage
        if int(elapsed) % 5 == 0:
            print(f"[{self.name}] ⚡ Performing event-driven condition check at elapsed: {elapsed:.1f}s")
        
        # Add periodic debug information much less frequently (every 60 seconds)
        if int(elapsed) % 60 == 0:
            self.debug_coordination_status()
        
        # Check conditions in order of priority
        
        # 1. Check if previous robot has finished pushing (highest priority)
        # For non-first robots, this is the gatekeeper check - don't proceed to position checks until this is true
        if not self._check_previous_robot_status():
            # If we've been waiting for more than 60 seconds, print debug info
            if elapsed > 60 and int(elapsed) % 30 == 0:  # Every 30 seconds after first minute
                print(f"[{self.name}] ⚠️ Still waiting for previous robot after {elapsed:.1f}s, printing debug info:")
                self.debug_coordination_status()
            return py_trees.common.Status.RUNNING
        
        if self.is_first_robot:
            # For first robot (turtlebot0), just check if parcel is within range of relay point
            if not self._check_parcel_position():
                return py_trees.common.Status.RUNNING
        else:
            # For non-turtlebot0 robots, only check these positions AFTER previous robot has finished pushing
            # This avoids unnecessary calculations when waiting for previous robot
            
            # Check if last robot is far enough from parcel
            if not self._check_last_robot_position():
                return py_trees.common.Status.RUNNING
                
            # And also check if parcel is within range of relay point
            if not self._check_parcel_position():
                return py_trees.common.Status.RUNNING
        
        # All conditions met - success!
        print(f"[{self.name}] ⚡ Check result: All conditions satisfied, returning SUCCESS")
        print(f"[{self.name}] SUCCESS: Ready to proceed with pushing!")
        self.previous_success_state = True
        return py_trees.common.Status.SUCCESS
        
    def _check_previous_robot_status(self):
        """Check if previous robot has finished pushing"""
        previous_finished = self.check_previous_robot_finished()
        self.cached_previous_finished = previous_finished
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace)
        
        if not previous_finished:
            # Only log every ~10 seconds to reduce CPU usage
            elapsed = time.time() - self.start_time
            if int(elapsed) % 10 == 0:
                print(f"[{self.name}] ⚡ Check result: Still waiting for {previous_robot_namespace} to finish pushing")
            self.feedback_message = f"[{self.robot_namespace}] Waiting for {previous_robot_namespace} to finish pushing..."
            self.previous_success_state = False
            return False
        return True
        
    def _check_last_robot_position(self):
        """Check if last robot is far enough from parcel"""
        # Use our dedicated method for this check
        last_robot_far_enough = self.check_last_robot_far_from_parcel()
        self.cached_last_robot_out = last_robot_far_enough  # Reuse the same cache variable
        
        # Only log every ~10 seconds to reduce CPU usage
        elapsed = time.time() - self.start_time
        log_now = int(elapsed) % 10 == 0
        
        if not last_robot_far_enough:
            if log_now:
                print(f"[{self.name}] ⚡ Check result: Still waiting for last robot (tb{self.last_robot_number}) to move away from parcel")
            self.feedback_message = f"[{self.robot_namespace}] Waiting for tb{self.last_robot_number} to move away from parcel..."
            self.previous_success_state = False
            return False
        elif log_now:
            print(f"[{self.name}] ⚡ Check result: Last robot (tb{self.last_robot_number}) is far enough from parcel")
        return True
        
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
            self.feedback_message = f"[{self.robot_namespace}] Waiting for parcel to move within range of relay point..."
            self.previous_success_state = False
            return False
        return True
    
    def _trigger_event(self, event_description, additional_info=""):
        """
        Unified method for triggering events with consistent logging
        
        Args:
            event_description (str): Description of the event that occurred
            additional_info (str): Any additional information to include in the log
        """
        # Only log events every ~5 seconds to reduce CPU usage
        elapsed = time.time() - self.start_time
        if int(elapsed) % 5 == 0:
            if additional_info:
                print(f"[{self.name}] ⚡ Event: {event_description}. {additional_info}")
            else:
                print(f"[{self.name}] ⚡ Event: {event_description}")
    
    def check_last_robot_far_from_parcel(self):
        """Check if last robot is far enough from parcel"""
        # For turtlebot0 (first robot), this condition is always satisfied
        if self.is_first_robot:
            return True
        
        # For other robots, check if last robot is far enough from parcel
        if self.last_robot_pose is None or self.parcel_pose is None:
            return False  # If we can't determine, assume not satisfied
        
        distance = self.calculate_distance(self.last_robot_pose, self.parcel_pose)
        is_far_enough = distance > self.distance_threshold
        return is_far_enough
    
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
                    self._trigger_event(
                        f"{position_label} distance threshold crossing detected",
                        f"Distance: {current_distance:.3f}m, Threshold: {self.distance_threshold:.3f}m"
                    )
                    
                    # Update cached values
                    self.prev_parcel_relay_distance = current_distance
                    self.last_distance_to_threshold = distance_to_threshold
                    return True
            
            # Update cached values even if no significant change
            self.prev_parcel_relay_distance = current_distance
            self.last_distance_to_threshold = distance_to_threshold
            return False
            
    # Keeping this for backward compatibility, now just calls the unified method
    def _check_distance_threshold(self, current_distance, position_label):
        """
        Backward compatibility wrapper for _check_distance_event
        """
        return self._check_distance_event(current_distance, position_label)
    
    def debug_coordination_status(self):
        """Debug method to check the current status of robot coordination"""
        # Get previous robot namespace if it exists
        previous_robot_namespace = self.get_previous_robot_namespace(self.robot_namespace) if not self.is_first_robot else None
        
        print(f"\n[{self.name}] 🔍 COORDINATION DEBUG INFO:")
        print(f"[{self.name}] Robot namespace: {self.robot_namespace}")
        print(f"[{self.name}] Is first robot: {self.is_first_robot}")
        print(f"[{self.name}] Previous robot namespace: {previous_robot_namespace}")
        print(f"[{self.name}] Previous robot finished status: {self.previous_robot_pushing_finished}")
        
        # Check if we have subscribers
        if hasattr(self, 'pushing_finished_sub') and self.pushing_finished_sub:
            print(f"[{self.name}] ✅ Subscribed to {previous_robot_namespace}/pushing_finished topic")
        else:
            print(f"[{self.name}] ❌ No subscription to previous robot's pushing_finished topic")
        
        # Check publisher
        if hasattr(self, 'pushing_finished_pub') and self.pushing_finished_pub:
            print(f"[{self.name}] ✅ Publishing to /{self.robot_namespace}/pushing_finished topic")
            
            # Send a test message
            from std_msgs.msg import Bool
            test_msg = Bool()
            test_msg.data = False  # Just a test, don't interfere with actual state
            self.pushing_finished_pub.publish(test_msg)
            print(f"[{self.name}] 📤 Sent test message to /{self.robot_namespace}/pushing_finished")
        else:
            print(f"[{self.name}] ❌ No publisher for this robot's pushing_finished topic")
        
        # Print topic communication layout
        if not self.is_first_robot:
            print(f"[{self.name}] Expected communication:")
            print(f"[{self.name}] {previous_robot_namespace} --[/{previous_robot_namespace}/pushing_finished]--> {self.robot_namespace}")
        
        print(f"[{self.name}] Current cached values:")
        print(f"[{self.name}]  - Previous robot finished: {self.cached_previous_finished}")
        print(f"[{self.name}]  - Last robot far from parcel: {self.cached_last_robot_out}")
        print(f"[{self.name}]  - Parcel in relay range: {self.cached_parcel_in_range}")
        print(f"[{self.name}] 🔍 END DEBUG INFO\n")
    

def create_event_driven_wait_for_push(name, duration=60.0, robot_namespace="turtlebot0", 
                                     distance_threshold=0.14):
    """
    Factory function to create an EventDrivenWaitForPush node.
    
    Args:
        name (str): Name of the behavior node
        duration (float): Maximum duration to wait before timing out
        robot_namespace (str): Robot namespace (e.g., 'turtlebot0')
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