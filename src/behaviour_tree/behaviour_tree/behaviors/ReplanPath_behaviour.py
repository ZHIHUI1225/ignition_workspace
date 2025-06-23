#!/usr/bin/env python3
"""
ReplanPath behavior for trajectory replanning in behavior trees.
Uses the replanning functions from Replan_behaviors.py to optimize robot trajectories.
"""

import py_trees
import time
import re
import rclpy
from std_msgs.msg import Float64
from .Replan_behaviors import replan_trajectory_parameters_to_target


class ReplanPath(py_trees.behaviour.Behaviour):
    """
    Behavior that replans robot trajectory to achieve target time from blackboard.
    
    This behavior reads the target time from the blackboard variable 'pushing_estimated_time'
    and uses the optimization functions to replan the robot's trajectory accordingly.
    """
    
    def __init__(self, name, duration=20.0, robot_namespace="turtlebot0", case="simple_maze"):
        """
        Initialize the ReplanPath behavior.
        
        Args:
            name (str): Name of the behavior
            duration (float): Maximum duration for the replanning process (seconds)
            robot_namespace (str): Robot namespace for ROS topics
            case (str): Case name for trajectory data (e.g., "simple_maze")
        """
        super().__init__(name)
        self.duration = duration
        self.robot_namespace = robot_namespace
        self.case = case
        self.start_time = None
        self.replanning_completed = False
        self.replanning_successful = False
        
        # ROS2 components
        self.node = None
        self.pushing_estimated_time_sub = None
        self.previous_robot_pushing_estimated_time = 45.0  # Default value
        
        # Extract robot number for determining previous robot
        namespace_match = re.search(r'turtlebot(\d+)', self.robot_namespace)
        self.namespace_number = int(namespace_match.group(1)) if namespace_match else 0
        self.previous_robot_namespace = f"turtlebot{self.namespace_number - 1}" if self.namespace_number > 0 else None
        
    
    def initialise(self):
        """Initialize the behavior when it starts running."""
        self.start_time = time.time()
        self.replanning_completed = False
        self.replanning_successful = False
        
        print(f"[{self.name}] Starting trajectory replanning for case '{self.case}'...")
        
        # Get target time from previous robot's pushing_estimated_time via ROS topic
        if self.namespace_number == 0:
            # turtlebot0 uses default 45s
            raw_target_time = 45.0
            print(f"[{self.name}] Using default target time for turtlebot0: {raw_target_time:.2f}s")
        else:
            # turtlebotN gets pushing_estimated_time from turtlebot(N-1) via ROS topic
            raw_target_time = self.previous_robot_pushing_estimated_time
            print(f"[{self.name}] Getting target time from {self.previous_robot_namespace} via ROS topic: {raw_target_time:.2f}s")

        # Check if pushing_estimated_time is too small (< 20 seconds)
        if raw_target_time < 20.0:
            from .tree_builder import report_node_failure
            error_msg = f"Pushing estimated time ({raw_target_time:.2f}s) is too small (< 20s) and fallback copy failed"
            report_node_failure(self.name, error_msg, self.robot_namespace)
            print(f"[{self.name}] FAILURE: {error_msg}")
            
            # Set system failure flag directly via blackboard
            try:
                blackboard_client = py_trees.blackboard.Client(name="replan_failure_reporter")
                blackboard_client.register_key(
                    key=f"{self.robot_namespace}/system_failed",
                    access=py_trees.common.Access.WRITE
                )
                blackboard_client.set(f"{self.robot_namespace}/system_failed", True)
                print(f"[{self.name}] System failure flag set to True due to insufficient pushing time and failed fallback")
            except Exception as bb_error:
                print(f"[{self.name}] Warning: Could not set system failure flag: {bb_error}")
            
            # Mark as failed to be caught in update()
            self.replanning_completed = True
            self.replanning_successful = False
            return

        target_time = raw_target_time
        print(f"[{self.name}] Target time for replanning: {target_time:.2f}s")
        
        # Use robot ID from namespace
        robot_id = self.namespace_number
        print(f"[{self.name}] Using robot ID: {robot_id}")
        
        # Store parameters for the update method
        self.target_time = target_time
        self.robot_id = robot_id
        
        self.feedback_message = f"[{self.robot_namespace}] Initializing replanning for Robot {robot_id}..."
    
    def update(self):
        """Update the behavior state."""
        if self.start_time is None:
            self.initialise()
            return py_trees.common.Status.RUNNING
        
        elapsed = time.time() - self.start_time
        
        # Check for timeout
        if elapsed >= self.duration:
            print(f"[{self.name}] Replanning timeout after {self.duration}s")
            return py_trees.common.Status.FAILURE
        
        # Check if replanning failed during initialization (e.g., pushing time too small)
        if self.replanning_completed and not self.replanning_successful:
            print(f"[{self.name}] Replanning failed during initialization")
            return py_trees.common.Status.FAILURE
        
        # If replanning hasn't been started yet, start it
        if not self.replanning_completed:
            print(f"[{self.name}] Executing trajectory replanning...")
            
            try:
                # Call the replanning function from Replan_behaviors.py
                result = replan_trajectory_parameters_to_target(
                    case=self.case,
                    target_time=self.target_time,
                    robot_id=self.robot_id,
                    save_results=True
                )
                
                if result is not None:
                    self.replanning_successful = True
                    self.replanning_completed = True
                    
                    print(f"[{self.name}] Replanning completed successfully!")
                    print(f"[{self.name}] Original time: {result['optimization_results']['original_total_time']:.3f}s")
                    print(f"[{self.name}] Optimized time: {result['optimization_results']['optimized_total_time']:.3f}s")
                    print(f"[{self.name}] Target time: {result['optimization_results']['target_time']:.3f}s")
                    print(f"[{self.name}] Deviation: {result['optimization_results']['deviation']:.3f}s")
                    
                    return py_trees.common.Status.SUCCESS
                else:
                    # Replanning failed - copy original data directly as fallback
                    print(f"[{self.name}] Replanning failed - copying original trajectory data as fallback")
                    
                    try:
                        # Import the copy function to handle fallback
                        from .Replan_behaviors import copy_original_trajectory_as_fallback
                        
                        # Copy original trajectory data
                        copy_result = copy_original_trajectory_as_fallback(
                            case=self.case,
                            robot_id=self.robot_id
                        )
                        
                        if copy_result:
                            self.replanning_completed = True
                            self.replanning_successful = True
                            
                            print(f"[{self.name}] Successfully copied original trajectory data as fallback")
                            print(f"[{self.name}] Robot {self.robot_id} will use original trajectory")
                            
                            return py_trees.common.Status.SUCCESS
                        else:
                            print(f"[{self.name}] Failed to copy original trajectory data")
                            
                    except ImportError:
                        print(f"[{self.name}] Fallback copy function not available - using manual copy")
                        
                        # Manual fallback - copy original trajectory file
                        if self._copy_original_trajectory_as_fallback():
                            self.replanning_completed = True
                            self.replanning_successful = True
                            
                            print(f"[{self.name}] Robot {self.robot_id} will use original trajectory")
                            return py_trees.common.Status.SUCCESS
                    
                    # If all fallback attempts failed, report failure but don't stop the system
                    self.replanning_completed = True
                    self.replanning_successful = False
                    
                    from .tree_builder import report_node_failure
                    error_msg = "Replanning failed and fallback copy also failed - optimization returned None"
                    report_node_failure(self.name, error_msg, self.robot_namespace)
                    print(f"[{self.name}] WARNING: Both replanning and fallback failed")
                
                    return py_trees.common.Status.FAILURE
                    
            except Exception as e:
                # Replanning failed with exception - try to copy original data as fallback
                print(f"[{self.name}] Replanning failed with exception: {str(e)}")
                print(f"[{self.name}] Attempting to copy original trajectory data as fallback...")
                
                try:
                    # Manual fallback - copy original trajectory file
                    if self._copy_original_trajectory_as_fallback():
                        self.replanning_completed = True
                        self.replanning_successful = True
                        
                        print(f"[{self.name}] Successfully copied original trajectory as fallback after exception")
                        print(f"[{self.name}] Robot {self.robot_id} will use original trajectory")
                        
                        return py_trees.common.Status.SUCCESS
                        
                except Exception as copy_error:
                    print(f"[{self.name}] Failed to copy original trajectory after exception: {copy_error}")
                
                # If fallback copy also failed, report the failure
                self.replanning_completed = True
                self.replanning_successful = False
                
                from .tree_builder import report_node_failure
                error_msg = f"Replanning failed with exception and fallback copy also failed: {str(e)}"
                report_node_failure(self.name, error_msg, self.robot_namespace)
                print(f"[{self.name}] WARNING: Both replanning and fallback failed due to exception")
                
                return py_trees.common.Status.FAILURE
        
        # If we get here, replanning was completed successfully
        return py_trees.common.Status.SUCCESS
    
    def terminate(self, new_status):
        """Clean up when the behavior terminates."""
        # Clean up ROS subscriptions
        if hasattr(self, 'pushing_estimated_time_sub') and self.pushing_estimated_time_sub:
            try:
                self.node.destroy_subscription(self.pushing_estimated_time_sub)
                self.pushing_estimated_time_sub = None
            except:
                pass
        
        if new_status == py_trees.common.Status.SUCCESS:
            print(f"[{self.name}] Replanning behavior completed successfully")
        elif new_status == py_trees.common.Status.FAILURE:
            print(f"[{self.name}] Replanning behavior failed")
        else:
            print(f"[{self.name}] Replanning behavior terminated with status: {new_status}")
        
        # Reset state for next execution
        self.start_time = None
        self.replanning_completed = False
        self.replanning_successful = False
    
    def setup(self, **kwargs):
        """Setup ROS2 components"""
        try:
            # Get the shared ROS node from kwargs
            if 'node' in kwargs:
                self.node = kwargs['node']
            else:
                print(f"[{self.name}] No ROS node provided")
                return False
            
            # Subscribe to previous robot's pushing_estimated_time topic
            if self.previous_robot_namespace:
                self.pushing_estimated_time_sub = self.node.create_subscription(
                    Float64,
                    f'/{self.previous_robot_namespace}/pushing_estimated_time',
                    self.pushing_estimated_time_callback,
                    10
                )
                # print(f"[{self.name}] DEBUG: Subscribed to {self.previous_robot_namespace}/pushing_estimated_time topic")
            else:
                print(f"[{self.name}] DEBUG: No previous robot (this is turtlebot0)")
            
            return True
            
        except Exception as e:
            print(f"[{self.name}] Setup failed: {e}")
            return False
    
    def pushing_estimated_time_callback(self, msg):
        """Callback for previous robot's pushing estimated time"""
        self.previous_robot_pushing_estimated_time = msg.data
        # print(f"[{self.name}] DEBUG: Received {self.previous_robot_namespace}/pushing_estimated_time = {msg.data:.2f}s")
    
    def _copy_original_trajectory_as_fallback(self):
        """
        Copy the original trajectory file as a fallback when replanning fails.
        
        Returns:
            bool: True if copy was successful, False otherwise
        """
        try:
            import os
            import shutil
            
            # Define file paths
            original_file = f'/root/workspace/data/{self.case}/tb{self.robot_id}_Trajectory.json'
            replanned_file = f'/root/workspace/data/{self.case}/tb{self.robot_id}_Trajectory_replanned.json'
            
            if os.path.exists(original_file):
                shutil.copy2(original_file, replanned_file)
                print(f"[{self.name}] Successfully copied original trajectory: {original_file} -> {replanned_file}")
                return True
            else:
                print(f"[{self.name}] Original trajectory file not found: {original_file}")
                return False
                
        except Exception as e:
            print(f"[{self.name}] Failed to copy original trajectory: {e}")
            return False
