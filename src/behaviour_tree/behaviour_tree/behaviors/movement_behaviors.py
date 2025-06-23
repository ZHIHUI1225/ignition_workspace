#!/usr/bin/env python3
"""
Movement behavior classes for the behavior tree system.
Contains robot movement and navigation behaviors with MPC-based control.
"""

import py_trees
import rclpy
import re
import traceback
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import math
import time
import threading
import tf_transformations as tf
import numpy as np


class MobileRobotMPC:
    """MPC controller for robot approach - now using proportional control for better performance"""
    def __init__(self):
        # Control constraints - these are the only parameters actually used
        self.vx_max = 0.05   # m/s max velocity in x direction
        self.vy_max = 0.05   # m/s max velocity in y direction
        
    def update_control(self, current_state, target_state, position_achieved=False):
        # SEQUENTIAL APPROACH: Position first, then orientation
        # Improved for stability: smooth transition, reduced overshoot, slower approach near target
        dist_to_target = np.sqrt((current_state[0] - target_state[0])**2 + (current_state[1] - target_state[1])**2)
        angle_diff = abs((current_state[2] - target_state[2] + np.pi) % (2 * np.pi) - np.pi)

        # PHASE 1: Position control (with smooth deceleration)
        if not position_achieved and dist_to_target >= 0.015:
            # Use fast proportional control instead of MPC for better performance
            # This avoids the 0.8s MPC computation time issue
            current_pos = np.array([current_state[0], current_state[1]])
            target_pos = np.array([target_state[0], target_state[1]])
            error = target_pos - current_pos
            
            # Proportional control with distance-based gain scaling
            base_gain = 0.8
            distance_scale = min(1.0, dist_to_target / 0.08)  # Scale down as we get closer
            gain = base_gain * distance_scale
            
            cmd_vel_2d = gain * error
            cmd_vel_2d[0] = np.clip(cmd_vel_2d[0], -0.06, 0.06)  # vx_max constraint
            cmd_vel_2d[1] = np.clip(cmd_vel_2d[1], -0.06, 0.06)  # vy_max constraint
            
            # Convert 2D velocity command to [linear_x, angular_z] format
            vx_global = cmd_vel_2d[0]
            vy_global = cmd_vel_2d[1]
            theta = current_state[2]
            vx_robot = vx_global * np.cos(theta) + vy_global * np.sin(theta)
            vy_robot = -vx_global * np.sin(theta) + vy_global * np.cos(theta)
            
            # Apply speed scaling for smooth deceleration
            speed_scale = min(1.0, dist_to_target / 0.08)  # Slow down within 8cm
            linear_vel = vx_robot * speed_scale
            angular_vel = (vy_robot / 0.18) * speed_scale  # Larger divisor for gentler turns
            
            # Clamp speeds for stability
            linear_vel = np.clip(linear_vel, -0.06, 0.06)
            angular_vel = np.clip(angular_vel, -0.15, 0.15)
            return np.array([linear_vel, angular_vel])
            
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
            return np.array([0.0, angular_vel])
        # Fallback: stop if in between phases
        else:
            return np.array([0.0, 0.0])


class ApproachObject(py_trees.behaviour.Behaviour):
    """
    Approach Object behavior - uses sequential position and orientation control.
    Uses MPC controller to make the robot approach the target with separate position and orientation phases.
    """

    def __init__(self, name="ApproachObject", robot_namespace="turtlebot0", approach_distance=0.14):
        """
        Initialize the ApproachObject behavior.
        
        Args:
            name: Name of the behavior node
            robot_namespace: The robot namespace (e.g., 'turtlebot0', 'turtlebot1')
            approach_distance: Distance to maintain from the parcel
        """
        super(ApproachObject, self).__init__(name)
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
        
        # Subscription lock for thread-safe access
        self._subscription_lock = threading.Lock()
        
        # Pose storage
        self.robot_pose = None
        self.parcel_pose = None
        self.current_parcel_index = 0
        
        # MPC controller (will be initialized in initialise() method)
        self.mpc = None
        
        # Control loop timer period for high-frequency control (10Hz)
        self.dt = 0.1
        
        # Threading components for high-frequency control
        self.control_thread = None
        self.control_thread_active = False
        self.control_thread_stop_event = threading.Event()
        
        # Threading lock for state protection
        self.lock = threading.Lock()
        
    def _stop_robot(self):
        """Helper method to stop the robot safely"""
        try:
            if self.cmd_vel_pub:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
                # Debug output for stop commands
                if not hasattr(self, '_stop_debug_counter'):
                    self._stop_debug_counter = 0
                self._stop_debug_counter += 1
                # Print every 20th stop command to avoid spam
                if self._stop_debug_counter % 20 == 1:
                    print(f"[{self.name}][{self.robot_namespace}] 发布停止命令 #{self._stop_debug_counter}: 线速度=0.0, 角速度=0.0")
        except Exception as e:
            print(f"[{self.name}][{self.robot_namespace}] 警告: 停止机器人时出错: {e}")

    def extract_namespace_number(self, namespace):
        """Extract numerical index from namespace string using regex"""
        match = re.search(r'\d+', namespace)
        return int(match.group()) if match else 0

    def setup(self, **kwargs):
        """设置ROS节点和通信组件（非阻塞优化版）
        
        功能包括：
        1. 创建ReentrantCallbackGroup支持并行回调
        2. 创建发布者（cmd_vel, pushing_estimated_time）
        3. 订阅移至initialise避免竞态条件
        """
        if 'node' in kwargs:
            self.node = kwargs['node']
            
            # 创建回调组支持并行执行
            self.callback_group = ReentrantCallbackGroup()
            
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
                # 从黑板获取当前包裹索引（安全回退）
                current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
                self.current_parcel_index = current_parcel_index
                print(f"[{self.name}] 调试: 从黑板获取包裹索引: {current_parcel_index}")
                
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
        
        # 始终从黑板更新self.current_parcel_index并（重新）创建订阅
        # 这确保我们始终订阅正确的话题，即使在节点重启后也是如此
        old_index = getattr(self, 'current_parcel_index', 'none')
        self.current_parcel_index = parcel_index
        
        try:
            # 始终清理现有订阅（如果存在）
            if self.parcel_pose_sub is not None:
                self.node.destroy_subscription(self.parcel_pose_sub)
                self.parcel_pose_sub = None
                print(f"[{self.name}] 调试: 已销毁现有包裹订阅")
            
            # 始终使用当前黑板索引和回调组创建新订阅
            parcel_topic = f'/parcel{self.current_parcel_index}/pose'
            if self.callback_group is not None:
                self.parcel_pose_sub = self.node.create_subscription(
                    PoseStamped, parcel_topic, self.parcel_pose_callback, 10,
                    callback_group=self.callback_group)
                print(f"[{self.name}] ✓ 包裹订阅已更新: parcel{old_index} -> parcel{self.current_parcel_index} (话题: {parcel_topic}) 使用回调组")
            else:
                self.parcel_pose_sub = self.node.create_subscription(
                    PoseStamped, parcel_topic, self.parcel_pose_callback, 10)
                print(f"[{self.name}] ✓ 包裹订阅已更新: parcel{old_index} -> parcel{self.current_parcel_index} (话题: {parcel_topic}) 无回调组")
            return True
            
        except Exception as e:
            print(f"[{self.name}] 错误: 包裹订阅更新失败: {e}")
            return False

    def robot_pose_callback(self, msg):
        """Callback for robot pose updates (Odometry message) - non-blocking and optimized"""
        # Early exit if subscription is being destroyed
        if self._robot_sub_destroying:
            return
            
        try:
            # Use minimal lock holding time and non-blocking approach
            if self.lock.acquire(blocking=False):
                try:
                    self.robot_pose = msg.pose.pose
                    # Update current state for MPC - local calculation to minimize lock time
                    x = self.robot_pose.position.x
                    y = self.robot_pose.position.y
                    theta = self.quaternion_to_yaw(self.robot_pose.orientation)
                    self.current_state = np.array([x, y, theta])
                finally:
                    self.lock.release()
            # If lock can't be acquired, skip this update (non-blocking)
        except Exception as e:
            # Silently handle exceptions during shutdown
            if not self._robot_sub_destroying:
                print(f"[{self.name}][{self.robot_namespace}] 警告: 机器人位姿回调异常: {e}")

    def parcel_pose_callback(self, msg):
        """Callback for parcel pose updates (PoseStamped message) - non-blocking and optimized"""
        # Early exit if subscription is being destroyed
        if self._parcel_sub_destroying:
            return
            
        try:
            # Use minimal lock holding time and non-blocking approach
            if self.lock.acquire(blocking=False):
                try:
                    self.parcel_pose = msg.pose
                finally:
                    self.lock.release()
            # If lock can't be acquired, skip this update (non-blocking)
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

    def control_thread_worker(self):
        """Dedicated control thread worker - runs at precise 10Hz"""
        print(f"[{self.name}][{self.robot_namespace}] 控制线程已启动 - 目标频率: 10Hz")
        
        # Debug counters for thread performance
        thread_debug_counter = 0
        thread_start_time = time.time()
        
        while not self.control_thread_stop_event.is_set():
            try:
                loop_start_time = time.time()
                
                # Only execute control if active and not both control flags achieved
                if (hasattr(self, 'control_active') and self.control_active and 
                    hasattr(self, 'position_control_achieved') and hasattr(self, 'orientation_control_achieved') and
                    not (self.position_control_achieved and self.orientation_control_achieved)):
                    
                    # Ensure we have necessary resources
                    if (hasattr(self, 'robot_pose_sub') and hasattr(self, 'parcel_pose_sub') and 
                        hasattr(self, 'cmd_vel_pub') and self.cmd_vel_pub):
                        
                        # Add timeout protection for control loop
                        control_start_time = time.time()
                        try:
                            self.control_loop()
                            control_duration = time.time() - control_start_time
                            
                            # Warn if control loop takes too long (>60ms for 10Hz operation)
                            if control_duration > 0.06:
                                print(f"[{self.name}][{self.robot_namespace}] ⚠️ 控制循环执行时间过长: {control_duration:.3f}s (建议: <0.060s)")
                        except Exception as control_error:
                            print(f"[{self.name}][{self.robot_namespace}] 错误: 控制循环失败: {control_error}")
                            self._stop_robot()
                    else:
                        # Stop robot if resources not available
                        self._stop_robot()
                else:
                    # Stop robot when not in control mode
                    self._stop_robot()
                
                # Debug frequency tracking
                thread_debug_counter += 1
                if thread_debug_counter % 50 == 1:  # Every 5 seconds at 10Hz
                    elapsed_time = time.time() - thread_start_time
                    actual_frequency = thread_debug_counter / elapsed_time if elapsed_time > 0 else 0
                    print(f"[{self.name}][{self.robot_namespace}] 控制线程统计: 调用#{thread_debug_counter}, 实际频率={actual_frequency:.1f}Hz (目标:10Hz)")
                
                # Sleep to maintain 10Hz frequency
                loop_execution_time = time.time() - loop_start_time
                sleep_time = self.dt - loop_execution_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif loop_execution_time > self.dt * 1.2:  # Warn if loop takes >120% of target time
                    print(f"[{self.name}][{self.robot_namespace}] ⚠️ 线程循环执行时间过长: {loop_execution_time:.3f}s (目标: {self.dt:.3f}s)")
                    
            except Exception as e:
                print(f"[{self.name}][{self.robot_namespace}] 错误: 控制线程异常: {e}")
                print(f"[{self.name}][{self.robot_namespace}] 异常详情: {traceback.format_exc()}")
                # Emergency stop on error
                try:
                    self._stop_robot()
                except:
                    pass
                time.sleep(0.1)  # Prevent tight error loop
        
        print(f"[{self.name}][{self.robot_namespace}] 控制线程已停止")

    def start_control_thread(self):
        """Start the dedicated control thread"""
        if self.control_thread is None or not self.control_thread.is_alive():
            self.control_thread_stop_event.clear()
            self.control_thread_active = True
            self.control_thread = threading.Thread(
                target=self.control_thread_worker,
                name=f"ControlThread_{self.robot_namespace}",
                daemon=True
            )
            self.control_thread.start()
            print(f"[{self.name}][{self.robot_namespace}] 专用控制线程已启动")
        else:
            print(f"[{self.name}][{self.robot_namespace}] 控制线程已在运行")

    def stop_control_thread(self):
        """Stop the dedicated control thread safely"""
        if self.control_thread and self.control_thread.is_alive():
            print(f"[{self.name}][{self.robot_namespace}] 正在停止控制线程...")
            self.control_thread_stop_event.set()
            self.control_thread_active = False
            
            # Wait for thread to finish with timeout
            self.control_thread.join(timeout=0.5)
            if self.control_thread.is_alive():
                print(f"[{self.name}][{self.robot_namespace}] 警告: 控制线程未在超时内停止")
            else:
                print(f"[{self.name}][{self.robot_namespace}] 控制线程已成功停止")
            self.control_thread = None


    def control_loop(self):  
        """Control loop for the approaching behavior - with non-blocking lock acquisition"""
        # Use non-blocking lock acquisition to prevent callback blocking
        if not self.lock.acquire(blocking=False):
            return  # Skip this control cycle if lock is busy
            
        try:
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
            position_threshold = 0.04  # 2cm for position
            orientation_threshold = 0.03  # ~3 degrees for orientation
            
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
                # Generate and apply control using MPC
                if self.mpc is not None:
                    try:
                        u = self.mpc.update_control(self.current_state, self.target_state, self.position_control_achieved)
                        if u is not None and self.cmd_vel_pub:
                            cmd = Twist()
                            cmd.linear.x = float(u[0])
                            cmd.angular.z = float(u[1])
                            self.cmd_vel_pub.publish(cmd)
                            # Debug output for published commands
                            if not hasattr(self, '_cmd_debug_counter'):
                                self._cmd_debug_counter = 0
                            self._cmd_debug_counter += 1
                            # Print every 10th command (once per second at 10Hz)
                            if self._cmd_debug_counter % 10 == 1:
                                print(f"[{self.name}][{self.robot_namespace}] 发布控制命令 #{self._cmd_debug_counter}: linear.x={cmd.linear.x:.3f} m/s, angular.z={cmd.angular.z:.3f} rad/s [频率: 10Hz]")
                                print(f"[{self.name}][{self.robot_namespace}] 控制状态: 位置误差={pos_dist:.3f}m, 角度误差={angle_diff:.3f}rad, 位置达成={self.position_control_achieved}, 方向达成={self.orientation_control_achieved}")
                    except Exception as e:
                        print(f"[{self.name}][{self.robot_namespace}] 错误: MPC控制失败: {e}")
                        self._stop_robot()
        finally:
            self.lock.release()

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
        # Reset state variables every time behavior launches
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target_state = np.array([0.0, 0.0, 0.0])   # [x, y, theta]
        self.control_active = False
        
        # 添加位置和方向控制的单独标志
        self.position_control_achieved = False
        self.orientation_control_achieved = False
        
        # 添加超时跟踪
        self.start_time = time.time()
        self.timeout_duration = 30.0  # 30秒接近超时
        
        # 重置姿态存储
        self.robot_pose = None
        self.parcel_pose = None
        
        self.feedback_message = f"[{self.robot_namespace}] 初始化接近行为"
        
        # 每次行为开始时设置默认推送预估时间（45秒）
        setattr(self.blackboard, f"{self.robot_namespace}/pushing_estimated_time", 45.0)
        
        # 每次节点启动时重置并创建新的MPC控制器
        print(f"[{self.name}][{self.robot_namespace}] 创建新的MPC控制器实例")
        self.mpc = MobileRobotMPC()
        
        # 如果节点可用则设置ROS组件
        if self.node:
            # 停止现有控制线程
            self.stop_control_thread()
            
            if self.robot_pose_sub is not None:
                self.node.destroy_subscription(self.robot_pose_sub)
                self.robot_pose_sub = None
            if self.parcel_pose_sub is not None:
                self.node.destroy_subscription(self.parcel_pose_sub)
                self.parcel_pose_sub = None
        
            success_robot = self.setup_robot_subscription()
            success_parcel = self.setup_parcel_subscription()
            
        # 启动专用控制线程而不是ROS定时器
        print(f"[{self.name}][{self.robot_namespace}] 启动专用10Hz控制线程...")
        self.start_control_thread()
        
        # 给ROS一点时间建立订阅再开始
        print(f"[{self.name}][{self.robot_namespace}] 调试: 允许时间建立ROS订阅...")

    def update(self):
        """Main update method - behavior tree logic only, control runs via timer"""
        # Check for parcel index changes and update subscription if needed
        current_parcel_index = getattr(self.blackboard, f'{self.robot_namespace}/current_parcel_index', 0)
        
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
            # Add debug info about subscription status
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = 0
            self._debug_counter += 1
            
            # Print debug info every 50 update cycles (about every 5 seconds)
            if self._debug_counter % 50 == 1:
                robot_topic = f'/turtlebot{self.namespace_number}/odom_map'
                parcel_topic = f'/parcel{self.current_parcel_index}/pose'
                print(f"[{self.name}][{self.robot_namespace}] DEBUG: Still waiting for pose data after {self._debug_counter} update cycles")
                print(f"[{self.name}][{self.robot_namespace}] DEBUG: Robot subscription: {self.robot_pose_sub is not None} (topic: {robot_topic})")
                print(f"[{self.name}][{self.robot_namespace}] DEBUG: Parcel subscription: {self.parcel_pose_sub is not None} (topic: {parcel_topic})")
                print(f"[{self.name}][{self.robot_namespace}] DEBUG: Robot pose received: {robot_pose_available}, Parcel pose received: {parcel_pose_available}")
            
            self.feedback_message = f"[{self.robot_namespace}] Waiting for pose data... (robot: {robot_pose_available}, parcel: {parcel_pose_available})"
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
        
        # Give a moment for thread to fully stop
        time.sleep(0.05)  # 50ms delay to allow thread to complete
        
        # Step 5: Clean up subscriptions with safe destruction
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
        
        # Step 6: Clear pose data
        self.robot_pose = None
        self.parcel_pose = None
        
        self.feedback_message = f"[{self.robot_namespace}] ApproachObject 已终止，状态: {new_status}"
        print(f"[{self.name}][{self.robot_namespace}] ApproachObject 终止完成，状态: {new_status}")


class MoveBackward(py_trees.behaviour.Behaviour):
    """Move backward behavior - using direct velocity control"""
    
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
            # Subscriber for robot pose
            self.robot_pose_sub = self.ros_node.create_subscription(
                Odometry, f'/turtlebot{self.robot_namespace[-1]}/odom_map', 
                self.robot_pose_callback, 10)
    
    def robot_pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        
    def calculate_distance_moved(self):
        if self.start_pose is None or self.current_pose is None:
            return 0.0
        dx = self.current_pose.position.x - self.start_pose.position.x
        dy = self.current_pose.position.y - self.start_pose.position.y
        return math.sqrt(dx*dx + dy*dy)
    
    def initialise(self):
        self.start_time = time.time()
        self.start_pose = self.current_pose
        self.feedback_message = f"[{self.robot_namespace}] Moving backward {self.distance}m"
        print(f"[{self.name}] Starting to move backward...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # Wait for pose data
        if self.current_pose is None:
            self.feedback_message = f"[{self.robot_namespace}] Waiting for pose data..."
            return py_trees.common.Status.RUNNING
        
        # Calculate how far we've moved
        distance_moved = self.calculate_distance_moved()
        self.feedback_message = f"[{self.robot_namespace}] Moving backward... {distance_moved:.2f}/{self.distance:.2f}m"
        
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
        """Clean up when behavior terminates - ensure robot stops"""
        # Stop the robot immediately
        if self.cmd_vel_pub:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_cmd)
        
        self.feedback_message = f"[{self.robot_namespace}] MoveBackward terminated with status: {new_status}"
        print(f"[{self.name}] MoveBackward terminated with status: {new_status} - robot stopped")