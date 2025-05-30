#!/usr/bin/env python3
"""
Manipulation behavior classes for the behavior tree system.
Contains object manipulation behaviors like pushing and picking.
"""

import py_trees
import rclpy
from std_srvs.srv import Trigger
import time
import threading
from geometry_msgs.msg import Twist
from .mpc_controller import MPCControllerNode


class PushObject(py_trees.behaviour.Behaviour):
    """Push object behavior using general MPC controller for trajectory following"""
    
    def __init__(self, name):
        super().__init__(name)
        self.start_time = None
        self.pushing_active = False
        self.ros_node = None
        self.mpc_controller_node = None
        self.pushing_complete = False
        self.robot_namespace = "tb0"  # Default, will be updated from parameters
        self.case = "simple_maze"  # Default case
        
    def setup(self, **kwargs):
        """Setup ROS connections and MPC controller"""
        if 'node' in kwargs:
            self.ros_node = kwargs['node']
            # Get robot namespace and case from ROS parameters
            try:
                self.robot_namespace = self.ros_node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "tb0"
            
            try:
                self.case = self.ros_node.get_parameter('case').get_parameter_value().string_value
            except:
                self.case = "simple_maze"
            
            print(f"[{self.name}] Setting up MPC push controller for {self.robot_namespace}, case: {self.case}")
    
    def initialise(self):
        """Initialize the pushing behavior with MPC controller"""
        self.start_time = time.time()
        self.pushing_active = False
        self.pushing_complete = False
        self.feedback_message = "Starting push operation with MPC..."
        print(f"[{self.name}] Starting to push object using MPC trajectory following...")
        
        # Create MPC controller node for pushing
        if self.mpc_controller_node is None:
            self.mpc_controller_node = MPCControllerNode(
                namespace=self.robot_namespace,
                case=self.case,
                controller_type="push"
            )
        
        # Start MPC control
        if self.mpc_controller_node.start_control():
            self.pushing_active = True
            print(f"[{self.name}] MPC push controller started successfully")
        else:
            print(f"[{self.name}] Failed to start MPC push controller")
    
    def update(self):
        """Update pushing behavior status"""
        if self.start_time is None:
            self.start_time = time.time()
        
        if not self.pushing_active:
            return py_trees.common.Status.FAILURE
        
        elapsed = time.time() - self.start_time
        self.feedback_message = f"MPC pushing object... {elapsed:.1f}s elapsed"
        
        # Check if MPC control is complete
        if self.mpc_controller_node and self.mpc_controller_node.is_control_complete():
            self.pushing_complete = True
            self.mpc_controller_node.stop_control()
            print(f"[{self.name}] Successfully pushed object using MPC!")
            return py_trees.common.Status.SUCCESS
        
        # Timeout check
        if elapsed >= 30.0:  # Increased timeout for MPC operations
            print(f"[{self.name}] Push operation timed out")
            if self.mpc_controller_node:
                self.mpc_controller_node.stop_control()
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        if self.mpc_controller_node:
            self.mpc_controller_node.stop_control()
        print(f"[{self.name}] Push behavior terminated with status: {new_status}")


class PickObject(py_trees.behaviour.Behaviour):
    """Pick object behavior using general MPC controller for trajectory following"""
    
    def __init__(self, name):
        super().__init__(name)
        self.start_time = None
        self.picking_active = False
        self.ros_node = None
        self.mpc_controller_node = None
        self.spawn_parcel_client = None
        self.picking_complete = False
        self.robot_namespace = "tb0"  # Default, will be updated from parameters
        self.case = "simple_maze"  # Default case
        
    def setup(self, **kwargs):
        """Setup ROS connections and MPC controller"""
        if 'node' in kwargs:
            self.ros_node = kwargs['node']
            # Get robot namespace and case from ROS parameters
            try:
                self.robot_namespace = self.ros_node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "tb0"
            
            try:
                self.case = self.ros_node.get_parameter('case').get_parameter_value().string_value
            except:
                self.case = "simple_maze"
            
            # Service client for spawning next parcel
            self.spawn_parcel_client = self.ros_node.create_client(
                Trigger, '/spawn_next_parcel_service')
            
            print(f"[{self.name}] Setting up MPC pick controller for {self.robot_namespace}, case: {self.case}")
    
    def initialise(self):
        """Initialize the picking behavior with MPC controller"""
        self.start_time = time.time()
        self.picking_active = False
        self.picking_complete = False
        self.feedback_message = "Starting pick operation with MPC..."
        print(f"[{self.name}] Starting to pick object using MPC trajectory following...")
        
        # Create MPC controller node for picking
        if self.mpc_controller_node is None:
            self.mpc_controller_node = MPCControllerNode(
                namespace=self.robot_namespace,
                case=self.case,
                controller_type="pick"
            )
        
        # Start MPC control
        if self.mpc_controller_node.start_control():
            self.picking_active = True
            print(f"[{self.name}] MPC pick controller started successfully")
        else:
            print(f"[{self.name}] Failed to start MPC pick controller")
    
    def update(self):
        """Update picking behavior status"""
        if self.start_time is None:
            self.start_time = time.time()
        
        if not self.picking_active:
            return py_trees.common.Status.FAILURE
        
        elapsed = time.time() - self.start_time
        self.feedback_message = f"MPC picking object... {elapsed:.1f}s elapsed"
        
        # Check if MPC control is complete
        if self.mpc_controller_node and self.mpc_controller_node.is_control_complete():
            
            # Spawn next parcel
            self._spawn_next_parcel()
            
            # Update blackboard with new parcel index
            self._update_parcel_index()
            
            self.picking_complete = True
            self.mpc_controller_node.stop_control()
            print(f"[{self.name}] Successfully picked object using MPC!")
            return py_trees.common.Status.SUCCESS
        
        # Timeout check
        if elapsed >= 45.0:  # Increased timeout for MPC operations
            print(f"[{self.name}] Pick operation timed out")
            if self.mpc_controller_node:
                self.mpc_controller_node.stop_control()
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING
    
    def _spawn_next_parcel(self):
        """Spawn the next parcel using service call"""
        if self.spawn_parcel_client and self.spawn_parcel_client.service_is_ready():
            spawn_request = Trigger.Request()
            spawn_future = self.spawn_parcel_client.call_async(spawn_request)
            rclpy.spin_until_future_complete(self.ros_node, spawn_future, timeout_sec=2.0)
            if spawn_future.done() and spawn_future.result():
                spawn_response = spawn_future.result()
                if spawn_response.success:
                    print(f"[{self.name}] Next parcel spawned successfully")
                else:
                    print(f"[{self.name}] Failed to spawn next parcel: {spawn_response.message}")
            else:
                print(f"[{self.name}] Spawn parcel service call timed out")
        else:
            print(f"[{self.name}] Spawn parcel service not available")
    
    def _update_parcel_index(self):
        """Update the current_parcel_index in blackboard"""
        try:
            blackboard = self.attach_blackboard_client(name=self.name)
            blackboard.register_key(key="current_parcel_index", access=py_trees.common.Access.WRITE)
            current_index = getattr(blackboard, 'current_parcel_index', 0)
            blackboard.current_parcel_index = current_index + 1
            print(f"[{self.name}] Updated current_parcel_index to {blackboard.current_parcel_index}")
        except Exception as e:
            print(f"[{self.name}] Failed to update current_parcel_index: {e}")
    
    def terminate(self, new_status):
        """Clean up when behavior terminates"""
        if self.mpc_controller_node:
            self.mpc_controller_node.stop_control()
        print(f"[{self.name}] Pick behavior terminated with status: {new_status}")
