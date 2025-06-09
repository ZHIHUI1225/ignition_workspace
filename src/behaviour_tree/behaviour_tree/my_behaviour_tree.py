#!/usr/bin/env python3
import py_trees
import py_trees.console as console
import rclpy
from rclpy.node import Node
import time
import threading
import traceback
import py_trees_ros.trees
import py_trees_ros.blackboard
import py_trees_ros.utilities
import py_trees_ros.visitors
import py_trees.display
from py_trees_ros_interfaces.srv import OpenSnapshotStream, CloseSnapshotStream

# Import all behaviors from the modular structure
from .behaviors import create_root

class ApproachObject(py_trees.behaviour.Behaviour):
    """接近物体节点 - 集成State_switch逻辑"""
    def __init__(self, name):
        super().__init__(name)
        self.start_time = None
        self.proximity_reached = False
        self.ros_node = None
        self.cmd_vel_pub = None
        self.robot_pose_sub = None
        self.parcel_pose_sub = None
        self.current_robot_pose = None
        self.current_parcel_pose = None
        self.proximity_threshold = 0.5  # meters
        self.robot_namespace = "turtlebot0"  # Default, will be updated from blackboard
        
    def setup(self, **kwargs):
        """Setup ROS connections"""
        if 'node' in kwargs:
            self.ros_node = kwargs['node']
            # Get robot namespace from ROS parameters
            try:
                self.robot_namespace = self.ros_node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "tb0"
            
            # Publishers and subscribers for approach behavior
            self.cmd_vel_pub = self.ros_node.create_publisher(
                Twist, f'/{self.robot_namespace}/cmd_vel', 10)
            self.robot_pose_sub = self.ros_node.create_subscription(
                Odometry, f'/turtlebot{self.robot_namespace[-1]}/odom_map', 
                self.robot_pose_callback, 10)
            
            # Subscribe to current parcel pose - get index from blackboard
            blackboard = self.attach_blackboard_client(name=self.name)
            blackboard.register_key(key="current_parcel_index", access=py_trees.common.Access.READ)
            parcel_index = getattr(blackboard, 'current_parcel_index', 0)
            self.parcel_pose_sub = self.ros_node.create_subscription(
                PoseStamped, f'/parcel{parcel_index}/pose', 
                self.parcel_pose_callback, 10)
    
    def robot_pose_callback(self, msg):
        self.current_robot_pose = msg.pose.pose
        
    def parcel_pose_callback(self, msg):
        self.current_parcel_pose = msg.pose
    
    def calculate_distance(self, pose1, pose2):
        if pose1 is None or pose2 is None:
            return float('inf')
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx*dx + dy*dy)
    
    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def initialise(self):
        self.start_time = time.time()
        self.proximity_reached = False
        self.feedback_message = "Approaching object..."
        print(f"[{self.name}] Starting to approach object...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # Check if we have valid pose data
        if self.current_robot_pose is None or self.current_parcel_pose is None:
            self.feedback_message = "Waiting for pose data..."
            return py_trees.common.Status.RUNNING
        
        # Calculate distance to parcel
        distance = self.calculate_distance(self.current_robot_pose, self.current_parcel_pose)
        self.feedback_message = f"Distance to object: {distance:.2f}m"
        
        # Check if close enough
        if distance <= self.proximity_threshold:
            # Stop the robot
            if self.cmd_vel_pub:
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
            print(f"[{self.name}] Successfully reached object! Distance: {distance:.2f}m")
            return py_trees.common.Status.SUCCESS
        
        # Continue approaching - basic proportional control
        if self.cmd_vel_pub and self.current_robot_pose and self.current_parcel_pose:
            cmd_vel = Twist()
            # Simple proportional control
            max_speed = 0.1  # m/s
            cmd_vel.linear.x = min(max_speed, distance * 0.2)
            
            # Calculate heading error
            dx = self.current_parcel_pose.position.x - self.current_robot_pose.position.x
            dy = self.current_parcel_pose.position.y - self.current_robot_pose.position.y
            target_yaw = math.atan2(dy, dx)
            
            # Get current yaw from quaternion
            current_yaw = self.quaternion_to_yaw(self.current_robot_pose.orientation)
            yaw_error = target_yaw - current_yaw
            
            # Normalize angle
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi
            
            cmd_vel.angular.z = yaw_error * 0.5  # Proportional control
            self.cmd_vel_pub.publish(cmd_vel)
        
        return py_trees.common.Status.RUNNING

class PushObject(py_trees.behaviour.Behaviour):
    """推动物体节点 - 集成Follow_controller逻辑"""
    def __init__(self, name):
        super().__init__(name)
        self.start_time = None
        self.pushing_active = False
        self.ros_node = None
        self.start_pushing_client = None
        self.pushing_finished_client = None
        self.pushing_complete = False
        self.robot_namespace = "tb0"  # Default, will be updated from parameters
        
    def setup(self, **kwargs):
        """Setup ROS connections"""
        if 'node' in kwargs:
            self.ros_node = kwargs['node']
            # Get robot namespace from ROS parameters
            try:
                self.robot_namespace = self.ros_node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "tb0"
            
            # Service clients for pushing operations
            self.start_pushing_client = self.ros_node.create_client(
                Trigger, f'/{self.robot_namespace}/start_pushing')
            self.pushing_finished_client = self.ros_node.create_client(
                Trigger, f'/{self.robot_namespace}/pushing_finished')
            
            # Wait for services (non-blocking)
            print(f"[{self.name}] Setting up push services for {self.robot_namespace}")
    
    def initialise(self):
        self.start_time = time.time()
        self.pushing_active = False
        self.pushing_complete = False
        self.feedback_message = "Starting push operation..."
        print(f"[{self.name}] Starting to push object...")
        
        # Call start pushing service
        if self.start_pushing_client and self.start_pushing_client.service_is_ready():
            request = Trigger.Request()
            future = self.start_pushing_client.call_async(request)
            # Use a timeout for the service call
            rclpy.spin_until_future_complete(self.ros_node, future, timeout_sec=1.0)
            if future.done() and future.result():
                response = future.result()
                if response.success:
                    self.pushing_active = True
                    print(f"[{self.name}] Push service started successfully")
                else:
                    print(f"[{self.name}] Failed to start push service: {response.message}")
            else:
                print(f"[{self.name}] Push service call timed out or failed")
        else:
            # Simulate push start if service not available
            self.pushing_active = True
            print(f"[{self.name}] Push service not available, simulating push start")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        if not self.pushing_active:
            return py_trees.common.Status.FAILURE
        
        elapsed = time.time() - self.start_time
        self.feedback_message = f"Pushing object... {elapsed:.1f}s elapsed"
        
        # Check if pushing is complete - simulate completion after 5 seconds
        if elapsed >= 5.0 and not self.pushing_complete:
            # Call pushing finished service if available
            if self.pushing_finished_client and self.pushing_finished_client.service_is_ready():
                request = Trigger.Request()
                future = self.pushing_finished_client.call_async(request)
                rclpy.spin_until_future_complete(self.ros_node, future, timeout_sec=1.0)
                if future.done() and future.result():
                    response = future.result()
                    if response.success:
                        self.pushing_complete = True
                        print(f"[{self.name}] Successfully pushed object!")
                        return py_trees.common.Status.SUCCESS
                    else:
                        print(f"[{self.name}] Push completion failed: {response.message}")
                        return py_trees.common.Status.FAILURE
            else:
                # Simulate successful completion
                self.pushing_complete = True
                print(f"[{self.name}] Successfully pushed object! (simulated)")
                return py_trees.common.Status.SUCCESS
        
        # Timeout check
        if elapsed >= 10.0:
            print(f"[{self.name}] Push operation timed out")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING

class MoveBackward(py_trees.behaviour.Behaviour):
    """后退节点 - 使用直接速度控制"""
    def __init__(self, name):
        super().__init__(name)
        self.distance = 0.3  # meters to move backward
        self.start_time = None
        self.ros_node = None
        self.cmd_vel_pub = None
        self.robot_pose_sub = None
        self.start_pose = None
        self.current_pose = None
        self.move_speed = -0.1  # negative for backward movement
        self.robot_namespace = "tb0"  # Default, will be updated from parameters
        
    def setup(self, **kwargs):
        """Setup ROS connections"""
        if 'node' in kwargs:
            self.ros_node = kwargs['node']
            # Get robot namespace from ROS parameters
            try:
                self.robot_namespace = self.ros_node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "tb0"
            
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
        self.feedback_message = f"Moving backward {self.distance}m"
        print(f"[{self.name}] Starting to move backward...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # Wait for pose data
        if self.current_pose is None:
            self.feedback_message = "Waiting for pose data..."
            return py_trees.common.Status.RUNNING
        
        # Calculate how far we've moved
        distance_moved = self.calculate_distance_moved()
        self.feedback_message = f"Moving backward... {distance_moved:.2f}/{self.distance:.2f}m"
        
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
        if elapsed >= 10.0:  # 10 second timeout
            if self.cmd_vel_pub:
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
            print(f"[{self.name}] Move backward timed out")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING

class ReplanPath(py_trees.behaviour.Behaviour):
    def __init__(self, name, duration=1.5):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Replanning path for {self.duration}s"
        print(f"[{self.name}] Starting to replan path...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.duration, 1.0)
        self.feedback_message = f"Replanning path... {progress*100:.1f}% complete"
        
        if elapsed >= self.duration:
            print(f"[{self.name}] Successfully replanned path!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class PickObject(py_trees.behaviour.Behaviour):
    """拾取物体节点 - 集成PickUp_controller逻辑"""
    def __init__(self, name):
        super().__init__(name)
        self.start_time = None
        self.picking_active = False
        self.ros_node = None
        self.start_picking_client = None
        self.picking_finished_client = None
        self.spawn_parcel_client = None
        self.picking_complete = False
        self.robot_namespace = "tb0"  # Default, will be updated from parameters
        
    def setup(self, **kwargs):
        """Setup ROS connections"""
        if 'node' in kwargs:
            self.ros_node = kwargs['node']
            # Get robot namespace from ROS parameters
            try:
                self.robot_namespace = self.ros_node.get_parameter('robot_namespace').get_parameter_value().string_value
            except:
                self.robot_namespace = "tb0"
            
            # Service clients for picking operations
            self.start_picking_client = self.ros_node.create_client(
                Trigger, f'/{self.robot_namespace}/start_picking')
            self.picking_finished_client = self.ros_node.create_client(
                Trigger, f'/{self.robot_namespace}/picking_finished')
            self.spawn_parcel_client = self.ros_node.create_client(
                Trigger, '/spawn_next_parcel_service')
            
            print(f"[{self.name}] Setting up pick services for {self.robot_namespace}")
    
    def initialise(self):
        self.start_time = time.time()
        self.picking_active = False
        self.picking_complete = False
        self.feedback_message = "Starting pick operation..."
        print(f"[{self.name}] Starting to pick object...")
        
        # Call start picking service
        if self.start_picking_client and self.start_picking_client.service_is_ready():
            request = Trigger.Request()
            future = self.start_picking_client.call_async(request)
            rclpy.spin_until_future_complete(self.ros_node, future, timeout_sec=1.0)
            if future.done() and future.result():
                response = future.result()
                if response.success:
                    self.picking_active = True
                    print(f"[{self.name}] Pick service started successfully")
                else:
                    print(f"[{self.name}] Failed to start pick service: {response.message}")
            else:
                print(f"[{self.name}] Pick service call timed out or failed")
        else:
            # Simulate pick start if service not available
            self.picking_active = True
            print(f"[{self.name}] Pick service not available, simulating pick start")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        if not self.picking_active:
            return py_trees.common.Status.FAILURE
        
        elapsed = time.time() - self.start_time
        self.feedback_message = f"Picking object... {elapsed:.1f}s elapsed"
        
        # Check if picking is complete - simulate completion after 8 seconds
        if elapsed >= 8.0 and not self.picking_complete:
            # Call picking finished service if available
            if self.picking_finished_client and self.picking_finished_client.service_is_ready():
                request = Trigger.Request()
                future = self.picking_finished_client.call_async(request)
                rclpy.spin_until_future_complete(self.ros_node, future, timeout_sec=1.0)
                if future.done() and future.result():
                    response = future.result()
                    if response.success:
                        print(f"[{self.name}] Pick operation completed")
                    else:
                        print(f"[{self.name}] Pick completion failed: {response.message}")
            
            # Spawn next parcel
            if self.spawn_parcel_client and self.spawn_parcel_client.service_is_ready():
                spawn_request = Trigger.Request()
                spawn_future = self.spawn_parcel_client.call_async(spawn_request)
                rclpy.spin_until_future_complete(self.ros_node, spawn_future, timeout_sec=1.0)
                if spawn_future.done() and spawn_future.result():
                    spawn_response = spawn_future.result()
                    if spawn_response.success:
                        print(f"[{self.name}] Next parcel spawned successfully")
                    else:
                        print(f"[{self.name}] Failed to spawn next parcel: {spawn_response.message}")
            
            # Update blackboard with new parcel index
            blackboard = self.attach_blackboard_client(name=self.name)
            blackboard.register_key(key="current_parcel_index", access=py_trees.common.Access.WRITE)
            current_index = getattr(blackboard, 'current_parcel_index', 0)
            blackboard.current_parcel_index = current_index + 1
            
            self.picking_complete = True
            print(f"[{self.name}] Successfully picked object!")
            return py_trees.common.Status.SUCCESS
        
        # Timeout check
        if elapsed >= 15.0:
            print(f"[{self.name}] Pick operation timed out")
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.RUNNING

class StopSystem(py_trees.behaviour.Behaviour):
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
    def __init__(self, name):
        super().__init__(name)
    def update(self):
        print(f"[{self.name}] Checking if pair is complete...")
        return py_trees.common.Status.SUCCESS

class IncrementIndex(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
    def update(self):
        print(f"[{self.name}] Incrementing index...")
        return py_trees.common.Status.SUCCESS

class PrintMessage(py_trees.behaviour.Behaviour):
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

def create_root(robot_namespace="tb0"):
    """创建行为树根节点（网页6结构优化版）"""
    root = py_trees.composites.Sequence(name="MainSequence", memory=True)
    
    # 初始化黑板变量（网页8权限规范）
    init_bb = py_trees.composites.Sequence(name="InitSequence", memory=True)
    init_bb.add_children([
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize index",
            variable_name="index",
            variable_value=0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize parcel index",
            variable_name="current_parcel_index",
            variable_value=0,
            overwrite=True
        ),
        ResetFlags("ResetFlags")
    ])
    
    # 主循环结构（修复网页7的Repeat装饰器错误）
    pair_sequence = py_trees.composites.Sequence(name="PairSequence", memory=True)
    
    # 动作执行并行节点（网页3策略优化）
    # 修改为Selector来测试，避免并行执行问题
    action_execution = py_trees.composites.Selector(
        name="ActionExecution",
        memory=True
    )
    
    # Pushing序列（网页5 XML结构转代码）- 直接调用类
    pushing_sequence = py_trees.composites.Sequence(name="PushingSequence", memory=True)
    pushing_sequence.add_children([
        WaitAction("WaitingPush", 3.0, robot_namespace),
        ApproachObject("ApproachingPush", robot_namespace),
        PushObject("Pushing"),
        MoveBackward("BackwardToSafeDistance")
    ])
    
    # PickingUp序列（网页3巡逻任务参考）- 直接调用类
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitAction("WaitingPick", 2.0, robot_namespace),
        ReplanPath("Replanning", 2.0),
        PickObject("PickingUp"),
        StopSystem("Stop", 1.5)
    ])
    
    # 完成检查序列（网页8黑板变量重置规范）
    completion_sequence = py_trees.composites.Sequence(name="CompletionSequence", memory=True)
    completion_sequence.add_children([
        CheckPairComplete("CheckPairComplete"),
        ResetFlags("ResetFlags"),
        IncrementIndex("IncrementIndex"),
        PrintMessage(
            name="PrintCompletedPair",
            message=lambda blackboard: f"Completed pair, index: {getattr(blackboard, 'index', 0)}"
        )
    ])
    
    # 组合结构（修复网页7的节点连接错误）
    action_execution.add_children([pushing_sequence, picking_up_sequence])
    pair_sequence.add_children([action_execution, completion_sequence])
    
    # 主循环装饰器（网页6装饰器正确用法）
    main_loop = py_trees.decorators.Repeat(
        name="MainLoop",
        child=pair_sequence,
        num_success=2  # Run only 2 iterations for testing
    )
    
    root.add_children([init_bb, main_loop])
    return root


def setup_snapshot_streams(node, robot_namespace=""):
    """设置快照流，使PyTrees Viewer可以连接到快照流"""
    # 设置默认快照流参数，这样PyTrees Viewer可以连接
    try:
        node.declare_parameter('default_snapshot_stream', True)
        node.declare_parameter('default_snapshot_blackboard_data', True) 
        node.declare_parameter('default_snapshot_blackboard_activity', True)
    except:
        # 参数可能已经声明过了
        pass
    
    # 构建快照流主题名称
    if robot_namespace:
        snapshot_topic = f"/{robot_namespace}/tree/snapshot_streams"
    else:
        snapshot_topic = "/tree/snapshot_streams"
    
    node.get_logger().info("Snapshot streams configured for PyTrees Viewer")
    node.get_logger().info(f"Connect PyTrees Viewer to: {snapshot_topic}")
    
    return snapshot_topic


def main():
    """主函数 - 启动带有PyTrees Viewer支持的行为树"""
    
    # 初始化ROS
    rclpy.init()
    
    try:
        # 创建临时节点获取参数
        temp_node = rclpy.create_node("param_reader")
        
        # 声明参数并获取
        try:
            temp_node.declare_parameter('robot_id', 0)
            temp_node.declare_parameter('robot_namespace', 'turtlebot0')
            robot_id = temp_node.get_parameter('robot_id').get_parameter_value().integer_value
            robot_namespace = temp_node.get_parameter('robot_namespace').get_parameter_value().string_value
        except Exception as e:
            print(f"Warning: Could not get robot parameters: {e}")
            robot_id = 0
            robot_namespace = "turtlebot0"
        
        # 销毁临时节点
        temp_node.destroy_node()
        
        # 创建ROS节点，用于执行器和快照发布 (使用"tree"作为节点名以支持快照流)
        ros_node = rclpy.create_node("tree")
        
        # 声明robot参数到主节点
        ros_node.declare_parameter('robot_id', robot_id)
        ros_node.declare_parameter('robot_namespace', robot_namespace)
        
        print(f"="*80)
        print(f"BEHAVIOR TREE FOR ROBOT {robot_id} ({robot_namespace})")
        print(f"="*80)
        
        # 创建行为树
        # Convert robot_namespace to the format expected by WaitAction (e.g., "turtlebot0" -> "tb0")
        import re
        match = re.search(r'turtlebot(\d+)', robot_namespace)
        tb_namespace = f"tb{match.group(1)}" if match else "tb0"
        root = create_root(tb_namespace)
        
        # 打印行为树结构
        print("BEHAVIOR TREE STRUCTURE:")
        print("="*40)
        print(py_trees.display.ascii_tree(root))
        
        print(f"Tree is running for {robot_namespace}... (Ctrl+C to stop)")
        print("ROS topics will be published for PyTrees Viewer:")
        print(f"  - /{robot_namespace}/tree_log")
        print(f"  - /{robot_namespace}/tree_snapshot")  
        print(f"  - /{robot_namespace}/tree_updates")
        print("="*80)
        
        # 使用py_trees_ros.trees.BehaviourTree创建ROS集成的行为树
        tree = py_trees_ros.trees.BehaviourTree(
            root=root,
            unicode_tree_debug=True
        )
        
        # 设置行为树 - 传入ROS节点以启用快照流
        tree.setup(timeout=15.0, node=ros_node)
        
        # 设置快照流供PyTrees Viewer使用
        snapshot_topic = setup_snapshot_streams(ros_node, robot_namespace)
        
        # 添加TreeToMsgVisitor用于PyTrees Viewer
        tree_to_msg_visitor = py_trees_ros.visitors.TreeToMsgVisitor()
        tree.add_visitor(tree_to_msg_visitor)
        
        # 创建ROS执行器用于后台处理ROS话题
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(ros_node)
        
        # 使用手动ticking循环，保证ROS话题处理
        iteration_count = 0
        
        # 创建后台线程用于ROS执行器
        def spin_ros():
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
        
        # 启动ROS执行器线程
        ros_thread = threading.Thread(target=spin_ros)
        ros_thread.daemon = True  # 守护线程，主线程结束时自动结束
        ros_thread.start()
        
        while rclpy.ok():
            tree.tick()
            
            # 每50次tick打印一次状态
            if iteration_count % 50 == 0:
                tree_status = tree.root.status
                print(f"--- Tick {iteration_count + 1} ---")
                print(f"Tree status: {tree_status}")
                
                # 显示树状态
                print(py_trees.display.ascii_tree(tree.root, show_status=True))
            
            # 如果树完成，重置并继续
            if tree.root.status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
                tree.root.stop(py_trees.common.Status.INVALID)
                
            iteration_count += 1
            time.sleep(0.5)  # 2Hz rate
        
        print("Behavior tree execution completed.")
        
    except KeyboardInterrupt:
        print("\nShutting down behavior tree...")
        if 'tree' in locals():
            tree.shutdown()
    except Exception as e:
        print(f"Error running behavior tree: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理ROS资源
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()