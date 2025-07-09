#!/usr/bin/env python3
"""
Movement behavior classes for the behavior tree system.
Contains robot movement and navigation behaviors with PI-based control.
"""

import py_trees
import rclpy
import re
import traceback
from queue import Queue, Empty
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import math
import time
import threading  # Keep full threading module for compatibility
from threading import Lock, Event  # Also import specific components for clarity
import tf_transformations as tf
import numpy as np


class RobotMotionPIController:
    """PI controller for robot approach - uses proportional-integral control for stable motion"""
    def __init__(self):
        # Control constraints
        self.vx_max = 0.05   # m/s max velocity in x direction
        self.vy_max = 0.05   # m/s max velocity in y direction
        
        # PI controller parameters
        self.kp = 0.8        # Proportional gain
        self.ki = 0.2        # Integral gain
        self.dt = 0.5        # Control timestep (2Hz)
        
        # PI controller state variables
        self.error_integral = np.array([0.0, 0.0])  # Integral of position error [x, y]
        self.last_error = np.array([0.0, 0.0])      # Previous error for derivative (if needed later)
        self.max_integral = 0.5  # Anti-windup limit for integral term
        
        # Reset flag to clear integral on new approach
        self.reset_pi_state = True
        
    def update_control(self, current_state, target_state, position_achieved=False):
        # SEQUENTIAL APPROACH: Position first, then orientation
        # Improved for stability: smooth transition, reduced overshoot, slower approach near target
        dist_to_target = np.sqrt((current_state[0] - target_state[0])**2 + (current_state[1] - target_state[1])**2)
        angle_diff = abs((current_state[2] - target_state[2] + np.pi) % (2 * np.pi) - np.pi)

        # PHASE 1: Position control with PI controller
        if not position_achieved and dist_to_target >= 0.015:
            # Reset PI state on new approach or significant target change
            if self.reset_pi_state:
                self.error_integral = np.array([0.0, 0.0])
                self.last_error = np.array([0.0, 0.0])
                self.reset_pi_state = False
                self._last_target = target_state.copy()  # Store target for change detection
                print(f"[RobotMotionPIController] PI controller state reset (distance: {dist_to_target:.3f}m)")
            elif hasattr(self, '_last_target'):
                # Check if target has changed significantly
                target_change = np.linalg.norm(target_state[:2] - self._last_target[:2])
                if target_change > 0.1:  # 10cm target change threshold
                    self.error_integral = np.array([0.0, 0.0])
                    self.last_error = np.array([0.0, 0.0])
                    self._last_target = target_state.copy()
                    print(f"[RobotMotionPIController] PI controller reset due to target change: {target_change:.3f}m")
            
            # Calculate position error in global frame
            current_pos = np.array([current_state[0], current_state[1]])
            target_pos = np.array([target_state[0], target_state[1]])
            error_global = target_pos - current_pos
            
            # Transform error to robot body frame
            robot_theta = current_state[2]
            cos_theta = np.cos(robot_theta)
            sin_theta = np.sin(robot_theta)
            
            # Rotation matrix from global to robot body frame
            # [x_robot]   [cos(θ)  sin(θ)] [x_global]
            # [y_robot] = [-sin(θ) cos(θ)] [y_global]
            error_body = np.array([
                error_global[0] * cos_theta + error_global[1] * sin_theta,    # forward/backward error
                -error_global[0] * sin_theta + error_global[1] * cos_theta   # left/right error
            ])
            
            # Update integral term with anti-windup (in body frame)
            self.error_integral += error_body * self.dt
            # Apply anti-windup: clamp integral to prevent excessive buildup
            self.error_integral = np.clip(self.error_integral, -self.max_integral, self.max_integral)
            
            # PI control calculation in body frame
            proportional_term = self.kp * error_body
            integral_term = self.ki * self.error_integral
            
            # Distance-based gain scaling for smooth approach
            distance_scale = min(1.0, dist_to_target / 0.08)  # Scale down as we get closer
            
            # Combine PI terms with distance scaling (in body frame)
            cmd_vel_body = (proportional_term + integral_term) * distance_scale
            cmd_vel_body[0] = np.clip(cmd_vel_body[0], -0.1, 0.1)  # forward/backward constraint
            cmd_vel_body[1] = np.clip(cmd_vel_body[1], -0.1, 0.1)  # left/right constraint
            
            # Apply speed scaling for smooth deceleration
            speed_scale = min(1.0, dist_to_target / 0.08)  # Slow down within 8cm
            
            # ROBOT BODY FRAME CONTROL: velocities in robot's local coordinate system
            linear_x_vel = cmd_vel_body[0] * speed_scale  # Forward/backward
            linear_y_vel = cmd_vel_body[1] * speed_scale  # Left/right
            
            # Clamp speeds for stability
            linear_x_vel = np.clip(linear_x_vel, -0.08, 0.08)
            linear_y_vel = np.clip(linear_y_vel, -0.08, 0.08)
            
            # For orientation control, align robot with target direction
            if dist_to_target > 0.02:  # Only adjust orientation when moving
                # Calculate desired heading toward target
                target_heading = np.arctan2(error_global[1], error_global[0])
                angular_error = target_heading - current_state[2]
                # Normalize angle difference
                while angular_error > np.pi:
                    angular_error -= 2 * np.pi
                while angular_error < -np.pi:
                    angular_error += 2 * np.pi
                angular_vel = 0.3 * angular_error  # Gentle orientation adjustment
                angular_vel = np.clip(angular_vel, -0.4, 0.4)
            else:
                angular_vel = 0.0
            
            # Store current error for next iteration
            self.last_error = error_body.copy()
            
            # Debug output (reduced frequency for performance - only when there are issues)
            if not hasattr(self, '_pi_debug_counter'):
                self._pi_debug_counter = 0
            self._pi_debug_counter += 1
            # Only print when there are control issues or every 100th iteration (every 10 seconds at 10Hz)
            should_debug = (self._pi_debug_counter % 100 == 1 or 
                          dist_to_target > 0.08 or  # Large distance error
                          abs(angular_error) > 0.3)  # Large angular error
            if should_debug:
                print(f"[RobotMotionPIController] PI Control - Error (body frame): [{error_body[0]:.3f}, {error_body[1]:.3f}], "
                      f"Integral: [{self.error_integral[0]:.3f}, {self.error_integral[1]:.3f}], "
                      f"Output: [linear.x={linear_x_vel:.3f}, linear.y={linear_y_vel:.3f}, ω={angular_vel:.3f}], Distance: {dist_to_target:.3f}m")
            
            return np.array([linear_x_vel, linear_y_vel, angular_vel])
            
        # PHASE 2: Orientation alignment (after position achieved)
        elif position_achieved:
            angular_error = target_state[2] - current_state[2]
            while angular_error > np.pi:
                angular_error -= 2 * np.pi
            while angular_error < -np.pi:
                angular_error += 2 * np.pi
            # Proportional rotation control with increased speed for faster orientation
            angular_vel = 0.6 * angular_error  # Increased gain for faster response
            angular_vel = np.clip(angular_vel, -0.4, 0.4)  # Higher max angular velocity
            return np.array([0.0, 0.0, angular_vel])  # No linear movement during orientation
        
        # Fallback: stop if in between phases
        else:
            return np.array([0.0, 0.0, 0.0])  # Full stop
    
    def reset_pi_controller(self):
        """Reset PI controller state - call when starting a new approach"""
        self.error_integral = np.array([0.0, 0.0])
        self.last_error = np.array([0.0, 0.0])
        self.reset_pi_state = True
        print(f"[RobotMotionPIController] PI controller state manually reset")
        
    def get_pi_state_info(self):
        """Get current PI controller state for debugging"""
        return {
            'error_integral': self.error_integral.copy(),
            'last_error': self.last_error.copy(),
            'kp': self.kp,
            'ki': self.ki,
            'max_integral': self.max_integral
        }


class EventDrivenApproachObject(py_trees.behaviour.Behaviour):
    """
    Event-driven approach object behavior - uses sequential position and orientation control.
    Uses PI controller to make the robot approach the target with separate position and orientation phases.
    Uses event queue for pose updates to eliminate lock contention.
    """

    def __init__(self, name="EventDrivenApproachObject", robot_namespace="turtlebot0", approach_distance=0.14):
        """
        Initialize the EventDrivenApproachObject behavior.
        
        Args:
            name: Name of the behavior node
            robot_namespace: The robot namespace (e.g., 'turtlebot0', 'turtlebot1')
            approach_distance: Distance to maintain from the parcel
        """
        super(EventDrivenApproachObject, self).__init__(name)
        self.robot_namespace = robot_namespace
        self.approach_distance = approach_distance
        
        # Extract namespace number for topic subscriptions
        self.namespace_number = self.extract_namespace_number(robot_namespace)
        
        # Setup blackboard access for current_parcel_index with namespace
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index", 
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/pushing_estimated_time", 
            access=py_trees.common.Access.WRITE
        )
        
        # ROS2 components (will be initialized in setup)
        self.node = None  # Changed from ros_node to node for consistency
        self.callback_group = None  # Will be initialized in setup
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        self.cmd_vel_pub = None
        self.pushing_estimated_time_pub = None
        
        # Subscription destruction flags to prevent race conditions
        self._robot_sub_destroying = False
        self._parcel_sub_destroying = False
        
        # Event Queue for thread-safe event processing (replaces locks)
        self._event_queue = Queue()
        self._processing_events = False
        self._last_processed_event_time = 0
        
        # Pose storage
        self.robot_pose = None
        self.parcel_pose = None
        self.current_parcel_index = 0
        
        # PI controller (will be initialized in initialise() method)
        self.controller = None
        
        # Control loop timer period for high-frequency control
        self.dt = 0.5  # Control timestep (2Hz)
        
        # Replace threading with ROS timer for control
        self.control_timer = None
        self.control_active = False
        
        # Thread-safe subscription lock (only used for subscription management)
        self._subscription_lock = Lock()  # Using Lock directly for subscription management only
        
    # Event queue methods
    def _enqueue_event(self, event_type, data=None):
        """Add an event to the queue with timestamp"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        self._event_queue.put(event)
        
    def _process_events(self):
        """Process all pending events in the queue"""
        if self._processing_events:
            return  # Prevent recursive calls
            
        self._processing_events = True
        events_processed = 0
        
        try:
            # Process all available events
            while not self._event_queue.empty():
                try:
                    event = self._event_queue.get_nowait()
                    self._handle_event(event)
                    self._event_queue.task_done()
                    events_processed += 1
                except Empty:
                    break
                    
            # Update timestamp of last processing
            if events_processed > 0:
                self._last_processed_event_time = time.time()
                
        finally:
            self._processing_events = False
            
    def _handle_event(self, event):
        """Handle a specific event based on its type"""
        event_type = event['type']
        data = event['data']
        
        if event_type == 'robot_pose_update':
            self._handle_robot_pose_update(data)
        elif event_type == 'parcel_pose_update':
            self._handle_parcel_pose_update(data)
        elif event_type == 'parcel_index_change':
            self._handle_parcel_index_change(data)
        else:
            print(f"[{self.name}][{self.robot_namespace}] 警告: 未知事件类型: {event_type}")
            
    def _handle_robot_pose_update(self, msg):
        """Handle robot pose update event"""
        if msg is None:
            return
            
        self.robot_pose = msg.pose.pose
        
        # Update current state - lock-free using event queue
        x = self.robot_pose.position.x
        y = self.robot_pose.position.y
        theta = self.quaternion_to_yaw(self.robot_pose.orientation)
        self.current_state = np.array([x, y, theta])
        
    def _handle_parcel_pose_update(self, msg):
        """Handle parcel pose update event"""
        if msg is None:
            return
            
        self.parcel_pose = msg.pose
        
    def _handle_parcel_index_change(self, new_index):
        """Handle parcel index change event"""
        if new_index == self.current_parcel_index:
            return
            
        old_index = self.current_parcel_index
        self.current_parcel_index = new_index
        print(f"[{self.name}][{self.robot_namespace}] ✓ 包裹索引已更新: {old_index} -> {new_index}")
        
        # Update subscription after index change
        self.setup_parcel_subscription()

# Keep the original class for backward compatibility but make it inherit from the new one
class ApproachObject(EventDrivenApproachObject):
    """
    Approach Object behavior - uses sequential position and orientation control.
    Uses PI controller to make the robot approach the target with separate position and orientation phases.
    
    This class now inherits from EventDrivenApproachObject for backward compatibility.
    """

    def __init__(self, name="ApproachObject", robot_namespace="turtlebot0", approach_distance=0.14):
        """
        Initialize the ApproachObject behavior.
        
        Args:
            name: Name of the behavior node
            robot_namespace: The robot namespace (e.g., 'turtlebot0', 'turtlebot1')
            approach_distance: Distance to maintain from the parcel
        """
        super(ApproachObject, self).__init__(name=name, robot_namespace=robot_namespace, approach_distance=approach_distance)
        print(f"[{name}] 注意: 使用事件驱动的ApproachObject实现以提高性能和稳定性")
        
        # For backward compatibility (if any code directly accesses this attribute)
        self.lock = None
        
    def _stop_robot(self):
        """Helper method to stop the robot safely"""
        try:
            if self.cmd_vel_pub:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
                # Debug output for stop commands - only occasionally
                if not hasattr(self, '_stop_debug_counter'):
                    self._stop_debug_counter = 0
                self._stop_debug_counter += 1
                # Only print every 100th stop command to avoid spam
                if self._stop_debug_counter % 100 == 1:
                    print(f"[{self.name}][{self.robot_namespace}] 发布停止命令 #{self._stop_debug_counter}: 线速度=0.0, 角速度=0.0")
        except Exception as e:
            print(f"[{self.name}][{self.robot_namespace}] 警告: 停止机器人时出错: {e}")

    def extract_namespace_number(self, namespace):
        """Extract number from namespace (e.g., 'turtlebot0' -> 0, 'turtlebot1' -> 1)"""
        match = re.search(r'turtlebot(\d+)', namespace)
        return int(match.group(1)) if match else 0
        
    def setup(self, **kwargs):
        """设置ROS节点和通信组件（非阻塞优化版）
        
        功能包括：
        1. 使用共享回调组管理器避免线程增殖
        2. 创建发布者（cmd_vel, pushing_estimated_time）
        3. 订阅移至initialise避免竞态条件
        """
        if 'node' in kwargs:
            self.node = kwargs['node']
            
            # 🔧 CRITICAL FIX: Use shared callback groups to prevent proliferation
            if hasattr(self.node, 'shared_callback_manager'):
                self.callback_group = self.node.shared_callback_manager.get_group('sensor')
                self.control_callback_group = self.node.shared_callback_manager.get_group('control')
                print(f"[{self.name}] ✅ Using shared callback groups: sensor={id(self.callback_group)}, control={id(self.control_callback_group)}")
            elif hasattr(self.node, 'robot_dedicated_callback_group'):
                self.callback_group = self.node.robot_dedicated_callback_group
                self.control_callback_group = self.node.robot_dedicated_callback_group
                print(f"[{self.name}] ✅ 使用机器人专用回调组: {id(self.callback_group)}")
            else:
                print(f"[{self.name}] ❌ 错误：没有找到shared_callback_manager，无法使用共享回调组")
                return False
            
            # Initialize state variables early to prevent callback race conditions
            self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
            self.target_state = np.array([0.0, 0.0, 0.0])   # [x, y, theta]
            
            # 创建ROS发布者
            self.cmd_vel_pub = self.node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            
            self.pushing_estimated_time_pub = self.node.create_publisher(
                Float64, f'/{self.robot_namespace}/pushing_estimated_time', 10)
            
            print(f"[{self.name}] 回调组已创建，支持并行回调")
            print(f"[{self.name}] {self.robot_namespace} 设置完成，订阅将在initialise中创建避免竞态条件")
            return True
        
        return False

    def setup_robot_subscription(self):
        """设置机器人姿态订阅（使用非阻塞回调组避免回调阻塞）"""
        if self.node is None:
            print(f"[{self.name}] 警告: 无法设置机器人订阅 - 缺少ROS节点")
            return False
            
        with self._subscription_lock:
            try:
                # Mark for safe destruction if exists
                if self.robot_pose_sub is not None:
                    self._robot_sub_destroying = True
                    time.sleep(0.01)  # Give callbacks time to exit
                    try:
                        self.node.destroy_subscription(self.robot_pose_sub)
                    except Exception as e:
                        print(f"[{self.name}] 警告: 销毁机器人订阅时出错: {e}")
                    finally:
                        self.robot_pose_sub = None
                        self._robot_sub_destroying = False
                
                # 使用回调组创建机器人里程计订阅，避免回调阻塞
                robot_odom_topic = f'/turtlebot{self.namespace_number}/odom_map'
                if self.callback_group is not None:
                    self.robot_pose_sub = self.node.create_subscription(
                        Odometry, robot_odom_topic, self.robot_pose_callback, 10,
                        callback_group=self.callback_group)
                    print(f"[{self.name}] ✓ 机器人订阅设置完成: {robot_odom_topic} (使用非阻塞回调组)")
                else:
                    self.robot_pose_sub = self.node.create_subscription(
                        Odometry, robot_odom_topic, self.robot_pose_callback, 10)
                    print(f"[{self.name}] ✓ 机器人订阅设置完成: {robot_odom_topic} (使用默认回调组)")
                return True
                
            except Exception as e:
                print(f"[{self.name}] 错误: 机器人订阅设置失败: {e}")
                self._robot_sub_destroying = False
                return False

    def setup_parcel_subscription(self):
        """设置包裹订阅（黑板就绪时使用回调组隔离）- 线程安全"""
        if self.node is None:
            print(f"[{self.name}] 警告: 无法设置包裹订阅 - 缺少ROS节点")
            return False
            
        with self._subscription_lock:
            try:
                # 从黑板获取当前包裹索引（安全回退）- 修复blackboard访问
                try:
                    current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
                    print(f"[{self.name}] 调试: 从黑板获取包裹索引: {current_parcel_index}")
                except Exception as bb_error:
                    # 黑板键不存在时使用默认值
                    print(f"[{self.name}] 信息: 黑板键尚未就绪，使用默认包裹索引0: {bb_error}")
                    current_parcel_index = 0
                
                self.current_parcel_index = current_parcel_index
                
                # Mark for safe destruction if exists
                if self.parcel_pose_sub is not None:
                    self._parcel_sub_destroying = True
                    time.sleep(0.01)  # Give callbacks time to exit
                    self.node.destroy_subscription(self.parcel_pose_sub)
                    self.parcel_pose_sub = None
                    self._parcel_sub_destroying = False
                    print(f"[{self.name}] 调试: 已安全销毁现有包裹订阅")
                    
                # 使用回调组创建新的包裹订阅
                parcel_topic = f'/parcel{current_parcel_index}/pose'
                if self.callback_group is not None:
                    self.parcel_pose_sub = self.node.create_subscription(
                        PoseStamped, parcel_topic, self.parcel_pose_callback, 10,
                        callback_group=self.callback_group)
                    print(f"[{self.name}] ✓ 成功订阅 {parcel_topic} (使用回调组)")
                else:
                    self.parcel_pose_sub = self.node.create_subscription(
                        PoseStamped, parcel_topic, self.parcel_pose_callback, 10)
                    print(f"[{self.name}] ✓ 成功订阅 {parcel_topic} (无回调组)")
                print(f"[{self.name}] 调试: 包裹订阅对象: {self.parcel_pose_sub}")
                print(f"[{self.name}] 调试: 节点名称: {self.node.get_name()}")
                return True
                
            except Exception as e:
                print(f"[{self.name}] 错误: 包裹订阅设置失败: {e}")
                traceback.print_exc()
                self._parcel_sub_destroying = False
                return False

    def update_parcel_subscription(self, new_parcel_index=None):
        """更新包裹订阅到正确话题（基于当前索引，始终使用黑板）"""
        if self.node is None:
            print(f"[{self.name}] 警告: 无法更新包裹订阅 - 缺少ROS节点")
            return False
            
        # 始终从黑板获取当前包裹索引（忽略new_parcel_index参数）
        try:
            parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
            print(f"[{self.name}] 调试: 从黑板检索包裹索引: {parcel_index}")
        except Exception as bb_error:
            # 黑板键尚不存在 - 在早期初始化时可能发生
            print(f"[{self.name}] 信息: 黑板键尚未就绪，使用默认包裹索引0: {bb_error}")
            parcel_index = 0
        
        # If the parcel index has changed, enqueue an event and update subscription
        old_index = getattr(self, 'current_parcel_index', 'none')
        if parcel_index != old_index:
            # Enqueue a parcel index change event
            self._enqueue_event('parcel_index_change', parcel_index)
        
        try:
            # 始终清理现有订阅（如果存在）
            if self.parcel_pose_sub is not None:
                self.node.destroy_subscription(self.parcel_pose_sub)
                self.parcel_pose_sub = None
                print(f"[{self.name}] 调试: 已销毁现有包裹订阅")
            
            # 始终使用当前黑板索引和回调组创建新订阅
            parcel_topic = f'/parcel{parcel_index}/pose'
            if self.callback_group is not None:
                self.parcel_pose_sub = self.node.create_subscription(
                    PoseStamped, parcel_topic, self.parcel_pose_callback, 10,
                    callback_group=self.callback_group)
                print(f"[{self.name}] ✓ 包裹订阅已更新: parcel{old_index} -> parcel{parcel_index} (话题: {parcel_topic}) 使用回调组")
            else:
                self.parcel_pose_sub = self.node.create_subscription(
                    PoseStamped, parcel_topic, self.parcel_pose_callback, 10)
                print(f"[{self.name}] ✓ 包裹订阅已更新: parcel{old_index} -> parcel{parcel_index} (话题: {parcel_topic}) 无回调组")
            
            # Update current_parcel_index after subscription is created successfully
            self.current_parcel_index = parcel_index
            return True
            
        except Exception as e:
            print(f"[{self.name}] 错误: 包裹订阅更新失败: {e}")
            return False

    def robot_pose_callback(self, msg):
        """Callback for robot pose updates (Odometry message) - non-blocking and event-driven"""
        # Early exit if subscription is being destroyed
        if self._robot_sub_destroying:
            return
            
        try:
            # Enqueue event with the message data
            self._enqueue_event('robot_pose_update', msg)
        except Exception as e:
            # Silently handle exceptions during shutdown
            if not self._robot_sub_destroying:
                print(f"[{self.name}][{self.robot_namespace}] 警告: 机器人位姿回调异常: {e}")

    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates (PoseStamped message) - non-blocking and event-driven"""
        # Early exit if subscription is being destroyed
        if self._parcel_sub_destroying:
            return
            
        try:
            # Enqueue event with the message data
            self._enqueue_event('parcel_pose_update', msg)
        except Exception as e:
            # Silently handle exceptions during shutdown
            if not self._parcel_sub_destroying:
                print(f"[{self.name}][{self.robot_namespace}] 警告: 包裹位姿回调异常: {e}")

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        quat_list = [x, y, z, w]
        euler = tf.euler_from_quaternion(quat_list)
        return euler[2]

    def control_loop_callback(self):
        """Control loop callback for ROS timer - replaces thread worker"""
        try:
            # Process any pending events first to ensure we have the latest state
            self._process_events()
            
            # Early exit if behavior is inactive
            if not self.control_active:
                return
                
            # Check that required data is available
            if self.robot_pose is None:
                return
                
            # Check that parcel data is available
            if self.parcel_pose is None:
                return
                
            # Run the control loop logic directly - no lock needed as we process events in the same thread
            self.control_loop()
                
        except Exception as e:
            print(f"[{self.name}][{self.robot_namespace}] 控制循环错误: {e}")
            traceback.print_exc()
            try:
                self._stop_robot()
            except:
                pass

    def control_loop(self):  
        """Control loop for the approaching behavior - now event-driven without locks"""
        # Process any pending events to ensure we have the latest state
        self._process_events()
            
        # Validate critical resources before proceeding
        if not hasattr(self, 'robot_pose') or not hasattr(self, 'parcel_pose'):
            return
            
        # Check if we have the necessary pose data
        if self.robot_pose is None or self.parcel_pose is None:
            return
        
        # Additional safety check: ensure we're still in control mode
        if not self.control_active:
            return
        
        # Calculate target state and update instance target_state
        target_state, distance_to_target_state = self.calculate_target_state()
        if target_state is None:
            return
        self.target_state = target_state
        
        # Calculate position and orientation errors
        pos_dist = np.sqrt((self.current_state[0] - self.target_state[0])**2 + 
                          (self.current_state[1] - self.target_state[1])**2)
        angle_diff = abs((self.current_state[2] - self.target_state[2] + np.pi) % (2 * np.pi) - np.pi)
        
        # Update control flags
        position_threshold = 0.05  # 5cm for position
        orientation_threshold = 0.05  # ~3 degrees for orientation
        
        if pos_dist < position_threshold:
            self.position_control_achieved = True
        
        if self.position_control_achieved and angle_diff < orientation_threshold:
            self.orientation_control_achieved = True
        
        # Check if both position and orientation control are achieved
        if self.position_control_achieved and self.orientation_control_achieved:
            self._stop_robot()
            self.control_active = False
            print(f"[{self.name}][{self.robot_namespace}] Both position and orientation control achieved! pos: {pos_dist:.3f}m, angle: {angle_diff:.3f}rad")
        else:
            # Generate and apply control using PI controller
            if self.controller is not None:
                try:
                    u = self.controller.update_control(self.current_state, self.target_state, self.position_control_achieved)
                    if u is not None and self.cmd_vel_pub:
                        cmd = Twist()
                        # Handle both 2D and 3D velocity commands
                        if len(u) == 3:  # True 2D movement: [vx, vy, angular_z]
                            cmd.linear.x = float(u[0])
                            cmd.linear.y = float(u[1])
                            cmd.angular.z = float(u[2])
                        else:  # Differential drive: [linear_x, angular_z]
                            cmd.linear.x = float(u[0])
                            cmd.angular.z = float(u[1])
                        self.cmd_vel_pub.publish(cmd)
                except Exception as e:
                    print(f"[{self.name}][{self.robot_namespace}] 错误: PI控制失败: {e}")
                    self._stop_robot()

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

    def calculate_target_state(self):
        """Calculate the target state for the robot based on parcel pose and optimal approach direction."""
        if self.robot_pose is None or self.parcel_pose is None:
            return None, float('inf')
        
        # Compute target state following State_switch.py logic
        target_state = np.array([
            self.parcel_pose.position.x,
            self.parcel_pose.position.y,
            self.quaternion_to_yaw(self.parcel_pose.orientation)
        ])
        
        # Get optimal direction and apply offset
        optimal_direction = self.get_direction(self.current_state[2], target_state[2])
        target_state[2] = optimal_direction
        target_state[0] = target_state[0] - (self.approach_distance) * math.cos(optimal_direction)
        target_state[1] = target_state[1] - (self.approach_distance) * math.sin(optimal_direction)
        
        # Calculate distance to target state
        distance_to_target_state = math.sqrt(
            (self.current_state[0] - target_state[0])**2 + 
            (self.current_state[1] - target_state[1])**2
        )
        
        return target_state, distance_to_target_state

    def publish_pushing_estimated_time(self):
        """Publish the pushing estimated time via ROS topic"""
        if self.pushing_estimated_time_pub:
            # Get the current pushing_estimated_time from blackboard, default to 45.0
            estimated_time = 50.0 # Default estimated time for turtlebot0
            msg = Float64()
            msg.data = estimated_time
            self.pushing_estimated_time_pub.publish(msg)

    def initialise(self):
        """Initialize the behavior when it starts running"""
        print(f"[{self.name}][{self.robot_namespace}] =================== INITIALISE START ===================")
        
        # Reset state variables every time behavior launches
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target_state = np.array([0.0, 0.0, 0.0])   # [x, y, theta]
        self.control_active = False
        
        # Control flags
        self.position_control_achieved = False
        self.orientation_control_achieved = False
        
        # Timeout tracking
        self.start_time = time.time()
        self.timeout_duration = 30.0  # 30 second timeout
        
        # Reset pose data
        self.robot_pose = None
        self.parcel_pose = None
        
        # Clear event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except Empty:
                break
        
        # Reset event processing state
        self._processing_events = False
        self._last_processed_event_time = time.time()
        
        self.feedback_message = f"[{self.robot_namespace}] 初始化事件驱动型接近行为"
        
        # Set default pushing estimated time
        setattr(self.blackboard, f"{self.robot_namespace}/pushing_estimated_time", 45.0)
        
        # Create new PI controller
        self.controller = RobotMotionPIController()
        
        # Reset PI controller state for new approach
        self.controller.reset_pi_controller()
        
        # 如果节点可用则设置ROS组件
        if self.node:
            # 停止现有控制线程
            self.stop_control_thread()
            
            # 🔧 修复：优雅地销毁现有订阅并等待完成
            print(f"[{self.name}][{self.robot_namespace}] 第一步：清理现有订阅...")
            if self.robot_pose_sub is not None:
                print(f"[{self.name}][{self.robot_namespace}] 销毁现有机器人订阅...")
                try:
                    self.node.destroy_subscription(self.robot_pose_sub)
                    print(f"[{self.name}][{self.robot_namespace}] ✓ 机器人订阅已销毁")
                except Exception as e:
                    print(f"[{self.name}][{self.robot_namespace}] 警告: 销毁机器人订阅时出错: {e}")
                self.robot_pose_sub = None
                time.sleep(0.1)  # 等待销毁完成
                
            if self.parcel_pose_sub is not None:
                print(f"[{self.name}][{self.robot_namespace}] 销毁现有包裹订阅...")
                try:
                    self.node.destroy_subscription(self.parcel_pose_sub)
                    print(f"[{self.name}][{self.robot_namespace}] ✓ 包裹订阅已销毁")
                except Exception as e:
                    print(f"[{self.name}][{self.robot_namespace}] 警告: 销毁包裹订阅时出错: {e}")
                self.parcel_pose_sub = None
                time.sleep(0.1)  # 等待销毁完成
            
            # 🔧 修复：验证订阅设置成功
            print(f"[{self.name}][{self.robot_namespace}] 第二步：创建新订阅...")
            success_robot = self.setup_robot_subscription()
            success_parcel = self.setup_parcel_subscription()
            
            print(f"[{self.name}][{self.robot_namespace}] 订阅创建结果: robot={success_robot}, parcel={success_parcel}")
            
            if not success_robot:
                print(f"[{self.name}][{self.robot_namespace}] ❌ 机器人订阅设置失败")
            if not success_parcel:
                print(f"[{self.name}][{self.robot_namespace}] ❌ 包裹订阅设置失败")
            
            # 🔧 修复：给ROS更多时间建立订阅连接
            print(f"[{self.name}][{self.robot_namespace}] 第三步：等待订阅建立连接...")
            time.sleep(0.8)  # 增加到800ms让订阅完全建立
            
            # 🔧 新增：验证话题连通性
            self.verify_topic_connectivity()
            
            # 🔧 新增：再次等待数据开始到达
            print(f"[{self.name}][{self.robot_namespace}] 第四步：等待数据开始到达...")
            time.sleep(0.5)  # 再等待500ms让数据开始流动
            
            # 🔧 新增：检查初始数据状态
            robot_has_data = self.robot_pose is not None
            parcel_has_data = self.parcel_pose is not None
            print(f"[{self.name}][{self.robot_namespace}] 初始数据检查: robot_data={robot_has_data}, parcel_data={parcel_has_data}")
            
        # 启动专用控制线程而不是ROS定时器
        print(f"[{self.name}][{self.robot_namespace}] 第五步：启动专用10Hz控制线程...")
        self.start_control_thread()
        
        print(f"[{self.name}][{self.robot_namespace}] =================== INITIALISE COMPLETE ===================")
        print(f"[{self.name}][{self.robot_namespace}] 初始化完成，开始等待话题数据...")
    
    def verify_topic_connectivity(self):
        """验证话题连通性和发布者状态"""
        if not self.node:
            return
            
        robot_topic = f'/turtlebot{self.namespace_number}/odom_map'
        parcel_topic = f'/parcel{self.current_parcel_index}/pose'
        
        print(f"[{self.name}][{self.robot_namespace}] 🔍 话题连通性验证:")
        
        try:
            # 检查话题是否存在
            topic_names_and_types = self.node.get_topic_names_and_types()
            available_topics = [name for name, _ in topic_names_and_types]
            
            robot_topic_exists = robot_topic in available_topics
            parcel_topic_exists = parcel_topic in available_topics
            
            # 检查发布者数量
            robot_pub_count = self.node.count_publishers(robot_topic)
            parcel_pub_count = self.node.count_publishers(parcel_topic)
            
            # 检查订阅者数量
            robot_sub_count = self.node.count_subscribers(robot_topic)
            parcel_sub_count = self.node.count_subscribers(parcel_topic)
            
            # 诊断问题
            if not robot_topic_exists:
                print(f"   ❌ 机器人话题不存在！检查Gazebo仿真和机器人spawning")
            elif robot_pub_count == 0:
                print(f"   ⚠️ 机器人话题无发布者！检查机器人节点是否运行")
            else:
                print(f"   ✅ 机器人话题连通性正常")
                
            if not parcel_topic_exists:
                print(f"   ❌ 包裹话题不存在！检查包裹spawning")
            elif parcel_pub_count == 0:
                print(f"   ⚠️ 包裹话题无发布者！检查包裹pose发布节点")
            else:
                print(f"   ✅ 包裹话题连通性正常")
                
        except Exception as e:
            print(f"   ❌ 话题连通性检查失败: {e}")
            traceback.print_exc()

    def update(self):
        """Main update method - behavior tree logic only, control runs via timer"""
        # Process any pending events first to ensure we have the latest state
        self._process_events()
        
        # Check for parcel index changes and update subscription if needed
        current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
        
        # Check if parcel index has changed
        if current_parcel_index != self.current_parcel_index:
            self._enqueue_event('parcel_index_change', current_parcel_index)
            self.update_parcel_subscription()
        
        # Periodically publish pushing_estimated_time while approaching
        if hasattr(self, '_last_estimated_time_publish'):
            if time.time() - self._last_estimated_time_publish > 1.0:  # Publish every 1 second
                self.publish_pushing_estimated_time()
                self._last_estimated_time_publish = time.time()
        else:
            # First time - publish and set timer
            self.publish_pushing_estimated_time()
            self._last_estimated_time_publish = time.time()
        
        # Check timeout
        elapsed = time.time() - self.start_time
        if elapsed >= self.timeout_duration:
            from .tree_builder import report_node_failure
            error_msg = f"ApproachObject timeout after {elapsed:.1f}s - failed to reach parcel"
            report_node_failure(self.name, error_msg, self.robot_namespace)
            print(f"[{self.name}][{self.robot_namespace}] FAILURE: Approach timeout after {elapsed:.1f}s")
            return py_trees.common.Status.FAILURE
        
        # Check if we have pose data and calculate target state
        robot_pose_available = self.robot_pose is not None
        parcel_pose_available = self.parcel_pose is not None
        
        if not robot_pose_available or not parcel_pose_available:
            # Check if we need to recreate subscriptions
            if hasattr(self.node, 'count_publishers'):
                robot_topic = f'/turtlebot{self.namespace_number}/odom_map'
                parcel_topic = f'/parcel{self.current_parcel_index}/pose'
                
                robot_pub_count = self.node.count_publishers(robot_topic)
                parcel_pub_count = self.node.count_publishers(parcel_topic)
                
                # If publishers exist but we have no data, try recreating subscriptions
                if not robot_pose_available and robot_pub_count > 0:
                    self.setup_robot_subscription()
                    
                if not parcel_pose_available and parcel_pub_count > 0:
                    self.setup_parcel_subscription()
            
            self.feedback_message = f"[{self.robot_namespace}] 等待话题数据... (机器人: {robot_pose_available}, 包裹: {parcel_pose_available})"
            return py_trees.common.Status.RUNNING

        target_state, distance_to_target_state = self.calculate_target_state()
        if target_state is None:
            self.feedback_message = f"[{self.robot_namespace}] Waiting for pose data..."
            return py_trees.common.Status.RUNNING
        
        self.target_state = target_state
        
        # Check if approach is complete
        if (self.position_control_achieved and self.orientation_control_achieved and not self.control_active):
            self._stop_robot()
            self.feedback_message = f"[{self.robot_namespace}] Both position and orientation control achieved, approach complete"
            print(f"[{self.name}][{self.robot_namespace}] Approach complete! Both control flags achieved. Distance to target state: {distance_to_target_state:.3f}m")
            return py_trees.common.Status.SUCCESS
        else:
            # Continue approaching the target state
            self.control_active = True
            
            self.feedback_message = (
                f"[{self.robot_namespace}] Approaching parcel{current_parcel_index} - "
                f"Distance: {distance_to_target_state:.3f}m, "
                f"target: ({self.target_state[0]:.2f}, {self.target_state[1]:.2f}, θ={self.target_state[2]:.2f}), "
                f"robot{self.namespace_number}: ({self.current_state[0]:.2f}, {self.current_state[1]:.2f}, θ={self.current_state[2]:.2f}), "
                f"position_flag: {self.position_control_achieved}, orientation_flag: {self.orientation_control_achieved}"
            )
            
            return py_trees.common.Status.RUNNING
                
    def terminate(self, new_status):
        """Clean up when behavior terminates - with safe subscription cleanup"""
        print(f"[{self.name}][{self.robot_namespace}] 开始终止行为，状态: {new_status}")
        
        # Step 1: Stop control and mark as inactive FIRST
        self.control_active = False
        
        # Step 2: Reset control flags
        self.position_control_achieved = False
        self.orientation_control_achieved = False
        
        # Step 3: Stop the robot immediately
        self._stop_robot()
        print(f"[{self.name}][{self.robot_namespace}] 调试: 机器人已停止")
        
        # Step 4: Stop the dedicated control thread
        self.stop_control_thread()
        
        # Step 5: Process any remaining events in the queue
        self._process_events()
        
        # Step 6: Clear the event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except Empty:
                break
                
        # Give a moment for events to be fully processed
        time.sleep(0.05)  # 50ms delay to allow thread to complete
        
        # Step 7: Clean up subscriptions with safe destruction
        with self._subscription_lock:
            if hasattr(self, 'robot_pose_sub') and self.robot_pose_sub is not None:
                try:
                    print(f"[{self.name}][{self.robot_namespace}] 调试: 开始安全销毁机器人订阅...")
                    self._robot_sub_destroying = True
                    time.sleep(0.02)  # Give callbacks time to exit gracefully
                    
                    if self.node:  # Simple node existence check
                        self.node.destroy_subscription(self.robot_pose_sub)
                        print(f"[{self.name}][{self.robot_namespace}] 调试: 机器人姿态订阅已安全销毁")
                    self.robot_pose_sub = None
                    self._robot_sub_destroying = False
                except Exception as e:
                    print(f"[{self.name}][{self.robot_namespace}] 警告: 机器人订阅销毁错误: {e}")
                    self.robot_pose_sub = None
                    self._robot_sub_destroying = False
            
            if hasattr(self, 'parcel_pose_sub') and self.parcel_pose_sub is not None:
                try:
                    print(f"[{self.name}][{self.robot_namespace}] 调试: 开始安全销毁包裹订阅...")
                    self._parcel_sub_destroying = True
                    time.sleep(0.02)  # Give callbacks time to exit gracefully
                    
                    if self.node:  # Simple node existence check
                        self.node.destroy_subscription(self.parcel_pose_sub)
                        print(f"[{self.name}][{self.robot_namespace}] 调试: 包裹姿态订阅已安全销毁")
                    self.parcel_pose_sub = None
                    self._parcel_sub_destroying = False
                except Exception as e:
                    print(f"[{self.name}][{self.robot_namespace}] 警告: 包裹订阅销毁错误: {e}")
                    self.parcel_pose_sub = None
                    self._parcel_sub_destroying = False
        
        # Step 8: Clear pose data
        self.robot_pose = None
        self.parcel_pose = None
        
        self.feedback_message = f"[{self.robot_namespace}] {self.name} 已终止，状态: {new_status}"
        print(f"[{self.name}][{self.robot_namespace}] {self.name} 终止完成，状态: {new_status}")

    def start_control_thread(self):
        """Start control timer (replacing thread with ROS timer) - uses shared callback group"""
        try:
            # Stop any existing timer first
            self.stop_control_thread()
            
            # Create timer using shared callback group for unified execution
            if hasattr(self, 'control_callback_group') and self.control_callback_group is not None:
                self.control_timer = self.node.create_timer(
                    self.dt,  # 0.1s timer period for 10Hz control
                    self.control_loop_callback,
                    callback_group=self.control_callback_group  # Use shared callback group
                )
                print(f"[{self.name}][{self.robot_namespace}] ✅ 控制定时器已启动 (周期: {self.dt}s，使用共享回调组)")
            else:
                # Fallback: use default callback group
                self.control_timer = self.node.create_timer(
                    self.dt,  # 0.1s timer period
                    self.control_loop_callback
                )
                print(f"[{self.name}][{self.robot_namespace}] ✅ 控制定时器已启动 (周期: {self.dt}s，使用默认回调组)")
                
            self.control_active = True
            return True
            
        except Exception as e:
            print(f"[{self.name}][{self.robot_namespace}] 错误: 启动控制定时器失败: {e}")
            return False

    def stop_control_thread(self):
        """Stop control timer (replacing thread cleanup with timer cleanup) - thread-safe"""
        try:
            # Mark control as inactive first
            self.control_active = False
            
            # Destroy timer if it exists
            if hasattr(self, 'control_timer') and self.control_timer is not None:
                try:
                    self.node.destroy_timer(self.control_timer)
                    self.control_timer = None
                    print(f"[{self.name}][{self.robot_namespace}] ✅ 控制定时器已停止")
                except Exception as e:
                    print(f"[{self.name}][{self.robot_namespace}] 警告: 停止控制定时器时出错: {e}")
                    self.control_timer = None
            
            # Stop robot immediately
            self._stop_robot()
            return True
            
        except Exception as e:
            print(f"[{self.name}][{self.robot_namespace}] 错误: 停止控制定时器失败: {e}")
            return False

class MoveBackward(py_trees.behaviour.Behaviour):
    """Move backward behavior - using event-driven velocity control"""
    
    def __init__(self, name, distance=0.2):
        super().__init__(name)
        self.distance = distance  # meters to move backward
        self.start_time = None
        self.ros_node = None
        self.cmd_vel_pub = None
        self.robot_pose_sub = None
        self.start_pose = None
        self.current_pose = None
        self.move_speed = -0.1  # negative for backward movement
        self.robot_namespace = "turtlebot0"  # Default, will be updated from parameters
        
        # Event queue for thread-safe pose updates
        self._event_queue = Queue()
        self._processing_events = False
        self._sub_destroying = False
        
    def _process_events(self):
        """Process all events in the queue - non-blocking"""
        if self._processing_events:
            return  # Prevent recursive processing
            
        self._processing_events = True
        events_processed = 0
        
        try:
            # Process all available events
            while not self._event_queue.empty():
                try:
                    event_type, data = self._event_queue.get_nowait()
                    if event_type == 'pose_update':
                        self.current_pose = data  # Thread-safe update in main thread
                    self._event_queue.task_done()
                    events_processed += 1
                except Empty:
                    break
        finally:
            self._processing_events = False
            
    def setup(self, **kwargs):
        """Setup ROS connections"""
        if 'node' in kwargs:
            self.ros_node = kwargs['node']
            # Get robot namespace from ROS parameters
            try:
                self.robot_namespace = self.ros_node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "turtlebot0"
            
            # Publisher for cmd_vel
            self.cmd_vel_pub = self.ros_node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
                
            # Create callback group if available for non-blocking callbacks
            callback_group = None
            if hasattr(self.ros_node, 'shared_callback_manager'):
                callback_group = self.ros_node.shared_callback_manager.get_group('sensor')
            
            # Subscriber for robot pose - with callback group if available
            topic = f'/turtlebot{self.robot_namespace[-1]}/odom_map'
            if callback_group:
                self.robot_pose_sub = self.ros_node.create_subscription(
                    Odometry, topic, self.robot_pose_callback, 10,
                    callback_group=callback_group)
            else:
                self.robot_pose_sub = self.ros_node.create_subscription(
                    Odometry, topic, self.robot_pose_callback, 10)
            
            print(f"[{self.name}] Setup complete: subscribed to {topic}")
    
    def robot_pose_callback(self, msg):
        """Thread-safe callback for robot pose updates using event queue"""
        # Early exit if subscription is being destroyed
        if hasattr(self, '_sub_destroying') and self._sub_destroying:
            return
            
        try:
            # Enqueue pose update event
            self._event_queue.put(('pose_update', msg.pose.pose))
        except Exception as e:
            if not self._sub_destroying:
                print(f"[{self.name}] Warning: Robot pose callback error: {e}")
        
    def calculate_distance_moved(self):
        if self.start_pose is None or self.current_pose is None:
            return 0.0
        dx = self.current_pose.position.x - self.start_pose.position.x
        dy = self.current_pose.position.y - self.start_pose.position.y
        return math.sqrt(dx*dx + dy*dy)
    
    def initialise(self):
        self.start_time = time.time()
        # Clear event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except Empty:
                break
        
        # Reset state
        self._processing_events = False
        self._sub_destroying = False
        
        # Save current pose as starting position
        self.start_pose = self.current_pose
        self.feedback_message = f"[{self.robot_namespace}] Moving backward {self.distance}m"
        print(f"[{self.name}] Starting to move backward...")
    
    def update(self):
        # Process any pending events first
        self._process_events()
        
        if self.start_time is None:
            self.start_time = time.time()
        
        # Wait for pose data
        if self.current_pose is None:
            self.feedback_message = f"[{self.robot_namespace}] Waiting for pose data..."
            return py_trees.common.Status.RUNNING
            
        # Set start_pose when we first get valid position data
        if self.start_pose is None and self.current_pose is not None:
            self.start_pose = self.current_pose
            print(f"[{self.name}] Got initial pose at: ({self.start_pose.position.x:.2f}, {self.start_pose.position.y:.2f})")
        
        # Calculate how far we've moved
        distance_moved = self.calculate_distance_moved()
        self.feedback_message = f"[{self.robot_namespace}] Moving backward... {distance_moved:.2f}/{self.distance:.2f}m"
        
        # Debug logging every few iterations
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 20 == 1:  # Print every 20th iteration to avoid spam
            print(f"[{self.name}] Debug: current=({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), " +
                  f"start=({self.start_pose.position.x:.2f}, {self.start_pose.position.y:.2f}), " +
                  f"dist={distance_moved:.3f}/{self.distance:.2f}")
        
        # Check if we've moved far enough
        if distance_moved >= self.distance:
            # Stop the robot
            if self.cmd_vel_pub:
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
            print(f"[{self.name}] Successfully moved to safe distance! Moved: {distance_moved:.2f}m")
            return py_trees.common.Status.SUCCESS
        
        # Continue moving backward
        if self.cmd_vel_pub:
            cmd_vel = Twist()
            cmd_vel.linear.x = self.move_speed
            cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_vel)
        
        # Safety timeout
        elapsed = time.time() - self.start_time
        if elapsed >= 15.0:  # 15 second timeout
            if self.cmd_vel_pub:
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
            from .tree_builder import report_node_failure
            error_msg = f"MoveBackward timeout after {elapsed:.1f}s - failed to reach target distance"
            report_node_failure(self.name, error_msg, "turtlebot0")  # MoveBackward doesn't have robot_namespace
            print(f"[{self.name}] Move backward timed out")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """Clean up when behavior terminates - ensure robot stops and resources are released"""
        # Stop the robot immediately
        if self.cmd_vel_pub:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_cmd)
            
        # Clean up subscription
        if hasattr(self, 'robot_pose_sub') and self.robot_pose_sub is not None:
            try:
                self._sub_destroying = True
                if hasattr(self, 'ros_node') and self.ros_node:
                    self.ros_node.destroy_subscription(self.robot_pose_sub)
                self.robot_pose_sub = None
            except Exception as e:
                print(f"[{self.name}] Warning: Error destroying subscription: {e}")
            finally:
                self._sub_destroying = False
                
        # Clear event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except Empty:
                break
                
        self.feedback_message = f"[{self.robot_namespace}] MoveBackward terminated with status: {new_status}"
        print(f"[{self.name}] MoveBackward terminated with status: {new_status} - robot stopped")