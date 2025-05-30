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


class ResetFlags(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
    def update(self):
        print(f"[{self.name}] Resetting flags...")
        return py_trees.common.Status.SUCCESS

class WaitAction(py_trees.behaviour.Behaviour):
    """等待动作节点（网页3示例改进版）"""
    def __init__(self, name, duration):
        super().__init__(name)
        self.duration = duration
        self.start_time = 0
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Waiting for {self.duration}s"
        print(f"[{self.name}] Starting wait for {self.duration}s...")
    
    def update(self) -> py_trees.common.Status:  # 添加类型注解（网页8建议）
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            print(f"[{self.name}] Wait completed!")
            return py_trees.common.Status.SUCCESS
        else:
            print(f"[{self.name}] Waiting... {elapsed:.1f}/{self.duration}s")
            return py_trees.common.Status.RUNNING

class ApproachObject(py_trees.behaviour.Behaviour):
    """接近物体节点 - 模拟实际的接近过程"""
    def __init__(self, name, duration=3.0):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Approaching object for {self.duration}s"
        print(f"[{self.name}] Starting to approach object...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.duration, 1.0)
        self.feedback_message = f"Approaching object... {progress*100:.1f}% complete"
        
        if elapsed >= self.duration:
            print(f"[{self.name}] Successfully reached object!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class PushObject(py_trees.behaviour.Behaviour):
    def __init__(self, name, duration=2.0):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Pushing object for {self.duration}s"
        print(f"[{self.name}] Starting to push object...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.duration, 1.0)
        self.feedback_message = f"Pushing object... {progress*100:.1f}% complete"
        
        if elapsed >= self.duration:
            print(f"[{self.name}] Successfully pushed object!")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class MoveBackward(py_trees.behaviour.Behaviour):
    def __init__(self, name, duration=1.5):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Moving backward for {self.duration}s"
        print(f"[{self.name}] Starting to move backward...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.duration, 1.0)
        self.feedback_message = f"Moving backward... {progress*100:.1f}% complete"
        
        if elapsed >= self.duration:
            print(f"[{self.name}] Successfully moved to safe distance!")
            return py_trees.common.Status.SUCCESS
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
    def __init__(self, name, duration=2.5):
        super().__init__(name)
        self.duration = duration
        self.start_time = None
    
    def initialise(self):
        self.start_time = time.time()
        self.feedback_message = f"Picking object for {self.duration}s"
        print(f"[{self.name}] Starting to pick object...")
    
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.duration, 1.0)
        self.feedback_message = f"Picking object... {progress*100:.1f}% complete"
        
        if elapsed >= self.duration:
            print(f"[{self.name}] Successfully picked object!")
            return py_trees.common.Status.SUCCESS
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

def create_root():
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
    
    # Pushing序列（网页5 XML结构转代码）
    pushing_sequence = py_trees.composites.Sequence(name="PushingSequence", memory=True)
    pushing_sequence.add_children([
        WaitAction("WaitingPush", 3.0),
        ApproachObject("ApproachingPush", 4.0),
        PushObject("Pushing", 3.0),
        MoveBackward("BackwardToSafeDistance", 2.0)
    ])
    
    # PickingUp序列（网页3巡逻任务参考）
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitAction("WaitingPick", 2.0),
        ReplanPath("Replanning", 2.0),
        PickObject("PickingUp", 3.0),
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
        root = create_root()
        
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