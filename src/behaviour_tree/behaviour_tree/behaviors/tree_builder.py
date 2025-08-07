#!/usr/bin/env python3
"""
Behavior tree structure and creation functions.
Contains the main tree building sequences and modular functions.
"""

import py_trees
from .basic_behaviors import (
   WaitAction, WaitForPush, WaitForPick, StopSystem, 
    CheckPairComplete, IncrementIndex
)
from .event_driven_behaviors import EventDrivenWaitForPush
from .ReplanPath_behaviour import ReplanPath
from .movement_behaviors import MoveBackward, ApproachObject
from .manipulation_behaviors import PushObject
from .Pickup import PickObject

# Add config import
from behaviour_tree.config_loader import config



def report_node_failure(node_name, error_info, robot_namespace, blackboard_client=None):
    """Utility function to report node failure to blackboard"""
    try:
        if blackboard_client is None:
            blackboard_client = py_trees.blackboard.Client(name="failure_reporter")
            blackboard_client.register_key(
                key=f"{robot_namespace}/failure_context",
                access=py_trees.common.Access.WRITE
            )
        
        failure_context = {
            "failed_node": node_name,
            "error_info": error_info,
            "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
        
        blackboard_client.set(f"{robot_namespace}/failure_context", failure_context)
        print(f"[FAILURE][{robot_namespace}] {node_name}: {error_info}")
        
    except Exception as e:
        print(f"[ERROR][{robot_namespace}] Failed to report failure for {node_name}: {e}")


class LoopCondition(py_trees.behaviour.Behaviour):
    """强化循环条件检查器 - 精确失败检测与快速响应"""
    
    def __init__(self, name, robot_namespace="robot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        
        # 注册所有相关的黑板键
        self.blackboard.register_key(
            key=f"{robot_namespace}/system_failed",
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/loop_iteration_count",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/current_parcel_index",
            access=py_trees.common.Access.READ
        )
        
        self.iteration_count = 0
    
    def setup(self, **kwargs):
        """初始化设置"""
        self.iteration_count = 0
        return True
    
    def update(self):
        """强化的循环条件检查逻辑"""
        # 增加迭代计数
        self.iteration_count += 1
        self.blackboard.set(f"{self.robot_namespace}/loop_iteration_count", self.iteration_count)
        
        # 获取当前状态信息
        try:
            system_failed = self.blackboard.get(f"{self.robot_namespace}/system_failed")
        except KeyError:
            print(f"[{self.name}][{self.robot_namespace}] ⚠️ system_failed 键不存在 - 默认为 False")
            system_failed = False
        
        try:
            parcel_index = self.blackboard.get(f"{self.robot_namespace}/current_parcel_index")
        except KeyError:
            parcel_index = "Unknown"
        
        print(f"[{self.name}][{self.robot_namespace}] 🔄 循环检查 #{self.iteration_count}: "
              f"system_failed={system_failed}, parcel_index={parcel_index}")
        
        # 快速失败检测 - 立即响应系统失败
        if system_failed:
            print(f"[{self.name}][{self.robot_namespace}] 🚨 CRITICAL: 系统失败检测到 - 立即终止循环")
            print(f"[{self.name}][{self.robot_namespace}] 📊 终止上下文: 迭代#{self.iteration_count}, 包裹#{parcel_index}")
            return py_trees.common.Status.FAILURE  # 立即终止循环
        
        # 系统正常 - 继续循环
        print(f"[{self.name}][{self.robot_namespace}] ✅ 系统正常 - 执行第 {self.iteration_count} 次迭代")
        return py_trees.common.Status.SUCCESS


class GlobalExceptionHandler(py_trees.behaviour.Behaviour):
    """强化全局异常处理器 - 统一失败传播与资源清理"""
    
    def __init__(self, name, robot_namespace="robot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        
        # 注册完整的黑板键集合
        required_keys = [
            "system_failed", "failure_context", "loop_iteration_count", 
            "current_parcel_index", "emergency_stop_requested"
        ]
        
        for key in required_keys:
            self.blackboard.register_key(
                key=f"{robot_namespace}/{key}",
                access=py_trees.common.Access.WRITE if key in ["system_failed", "emergency_stop_requested"] 
                       else py_trees.common.Access.READ
            )
    
    def update(self):
        """强化的全局失败处理逻辑"""
        print(f"[{self.name}][{self.robot_namespace}] 🚨 全局异常处理器激活 - 执行系统级清理")
        
        # 收集完整的失败上下文
        context_info = {}
        try:
            context_info["iteration_count"] = self.blackboard.get(f"{self.robot_namespace}/loop_iteration_count")
            context_info["parcel_index"] = self.blackboard.get(f"{self.robot_namespace}/current_parcel_index")
            context_info["failure_context"] = self.blackboard.get(f"{self.robot_namespace}/failure_context")
            context_info["timestamp"] = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        except KeyError as e:
            context_info["error"] = f"Context collection failed: {e}"
        
        print(f"[{self.name}][{self.robot_namespace}] 📊 完整失败上下文: {context_info}")
        
        # 确保系统失败标志被正确设置 (幂等操作)
        self.blackboard.set(f"{self.robot_namespace}/system_failed", True)
        
        # 设置紧急停止标志（可选，用于其他组件）
        try:
            self.blackboard.set(f"{self.robot_namespace}/emergency_stop_requested", True)
        except:
            pass  # 可选标志，失败不影响主要逻辑
        
        # 输出统一的失败报告
        print(f"[{self.name}][{self.robot_namespace}] � SYSTEM FAILURE CONFIRMED")
        print(f"[{self.name}][{self.robot_namespace}] 📋 Final Context: {context_info}")
        print(f"[{self.name}][{self.robot_namespace}] ⚡ 返回 FAILURE - 确保失败传播到根节点")
        
        # 返回 FAILURE 以确保整个系统停止
        return py_trees.common.Status.FAILURE



def create_root(robot_namespace="robot0", case="experi", control_dt=None):
    """Create behavior tree root node with optimized Selector+LoopCondition control and parallel blackboard init"""
    # Use provided control_dt or fall back to global default
    # Load discrete_dt from config
    discrete_dt = getattr(config, "discrete_dt", 0.2)

    root = py_trees.composites.Sequence(name="MainSequence", memory=True)
    
    # OPTIMIZATION: Parallel blackboard initialization for faster startup
    init_blackboard_parallel = py_trees.composites.Parallel(
        name="InitBlackboardParallel",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    
    init_blackboard_parallel.add_children([
        py_trees.behaviours.SetBlackboardVariable(
            name="InitParcelIndex",
            variable_name=f"{robot_namespace}/current_parcel_index",
            variable_value=0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="InitSystemFailedFlag",
            variable_name=f"{robot_namespace}/system_failed",
            variable_value=False,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="InitLoopIterationCount",
            variable_name=f"{robot_namespace}/loop_iteration_count",
            variable_value=0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="InitFailureContext",
            variable_name=f"{robot_namespace}/failure_context",
            variable_value=None,
            overwrite=True
        ),
        # Set discrete_dt in blackboard
        py_trees.behaviours.SetBlackboardVariable(
            name="DiscreteDT",
            variable_name="discrete_dt",
            variable_value=discrete_dt,
            overwrite=True
        )
    ])
    
    # Main pair sequence - core business logic
    pair_sequence = py_trees.composites.Sequence(name="PairSequence", memory=True)
    
    # Pushing sequence
    pushing_sequence = py_trees.composites.Sequence(name="PushingSequence", memory=True)
    pushing_sequence.add_children([
        # EventDrivenWaitForPush("WaitingPush", 3000.0, robot_namespace, distance_threshold=0.14),
        WaitForPush("WaitingPush", 3000.0, robot_namespace, distance_threshold=0.14),
        ApproachObject("ApproachingPush", robot_namespace),
        PushObject("Pushing", robot_namespace, distance_threshold=0.07, case=case)
    ])
    
    # Parallel execution for move + replan
    parallel_move_replan = py_trees.composites.Parallel(
        name="ParallelMoveReplan",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    parallel_move_replan.add_children([
        MoveBackward("BackwardToSafeDistance", distance=0.2),
        ReplanPath("Replanning", 20.0, robot_namespace, case, dt=control_dt)
    ])
    
    # Picking sequence
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitForPick("WaitingPick", 2.0, robot_namespace),
        PickObject("PickingUp", robot_namespace, timeout=100.0, dt=control_dt, case=case),
        StopSystem("Stop", 1.5)
    ])
    
    # Completion check sequence
    completion_sequence = py_trees.composites.Sequence(name="CompletionSequence", memory=True)
    completion_sequence.add_children([
        CheckPairComplete("CheckPairComplete", robot_namespace),
        IncrementIndex("IncrementIndex", robot_namespace),
    ])
    
    # Build the main sequence: if ANY step fails, the entire sequence fails
    pair_sequence.add_children([
        pushing_sequence, 
        parallel_move_replan, 
        picking_up_sequence,
        completion_sequence
    ])
    
    # Create loop body with condition check FIRST
    loop_iteration_body = py_trees.composites.Sequence(name="LoopIterationBody", memory=True)
    loop_iteration_body.add_children([
        LoopCondition("CheckLoopCondition", robot_namespace),  # FIRST: Check if loop should continue
        pair_sequence  # SECOND: Execute main logic if condition passes
    ])
    
    # MAIN LOOP CONTROL: Selector + LoopCondition (replaces Repeat decorator)
    # This selector keeps trying loop_iteration_body until LoopCondition returns FAILURE
    main_loop_selector = py_trees.composites.Selector(
        name="MainLoopSelector",
        memory=False  # No memory - always try loop_iteration_body first
    )
    
    # The selector tries the loop body repeatedly until LoopCondition fails
    # When LoopCondition returns FAILURE (system_failed=True), the selector moves to next child
    main_loop_selector.add_children([
        py_trees.decorators.Repeat(
            name="LoopBodyRepeater",
            child=loop_iteration_body,
            num_success=-1  # Repeat indefinitely until child fails (LoopCondition fails)
        ),
        GlobalExceptionHandler("GlobalFailureHandler", robot_namespace)  # Activated when loop terminates
    ])
    
    # Final root structure: Initialize → Loop → Global failure handling
    root.add_children([
        init_blackboard_parallel,  # Parallel initialization for speed
        main_loop_selector  # Selector-based loop control with global failure handling
    ])
    
    return root
    
    return root
    return root
    
    return root
