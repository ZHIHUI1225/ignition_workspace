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
from .ReplanPath_behaviour import ReplanPath
from .movement_behaviors import ApproachObject, MoveBackward
from .manipulation_behaviors import PushObject
from .Pickup import PickObject


class FailureHandler(py_trees.behaviour.Behaviour):
    """Behavior to handle failures and set blackboard variables"""
    
    def __init__(self, name, robot_namespace="turtlebot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key=f"{robot_namespace}/system_failed",
            access=py_trees.common.Access.WRITE
        )
    
    def update(self):
        print(f"[{self.name}] FAILURE DETECTED - Setting system_failed flag")
        self.blackboard.set(f"{self.robot_namespace}/system_failed", True)
        return py_trees.common.Status.SUCCESS


class LoopCondition(py_trees.behaviour.Behaviour):
    """Check if loop should continue based on failure state"""
    
    def __init__(self, name, robot_namespace="turtlebot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key=f"{robot_namespace}/system_failed",
            access=py_trees.common.Access.READ
        )
    
    def update(self):
        try:
            system_failed = self.blackboard.get(f"{self.robot_namespace}/system_failed")
            if system_failed:
                print(f"[{self.name}] System failure detected - stopping loop")
                return py_trees.common.Status.FAILURE
            else:
                return py_trees.common.Status.SUCCESS
        except KeyError:
            # If key doesn't exist, system hasn't failed
            return py_trees.common.Status.SUCCESS


class FailureLogger(py_trees.behaviour.Behaviour):
    """Log failures and provide detailed information"""
    
    def __init__(self, name, robot_namespace="turtlebot0", component_name="Unknown"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.component_name = component_name
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key=f"{robot_namespace}/system_failed",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/failure_reason",
            access=py_trees.common.Access.WRITE
        )
    
    def update(self):
        failure_message = f"CRITICAL FAILURE in {self.component_name} - System stopping"
        print(f"[{self.name}] {failure_message}")
        
        # Set failure flags on blackboard
        self.blackboard.set(f"{self.robot_namespace}/system_failed", True)
        self.blackboard.set(f"{self.robot_namespace}/failure_reason", failure_message)
        
        # Log to file for debugging
        try:
            with open(f"/root/workspace/failure_log_{self.robot_namespace}.txt", "a") as f:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {failure_message}\n")
        except Exception as e:
            print(f"[{self.name}] Failed to write to log file: {e}")
        
        return py_trees.common.Status.SUCCESS


class SystemMonitor(py_trees.behaviour.Behaviour):
    """Monitor system state and provide status updates"""
    
    def __init__(self, name, robot_namespace="turtlebot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key=f"{robot_namespace}/system_failed",
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/failure_reason",
            access=py_trees.common.Access.READ
        )
    
    def update(self):
        try:
            system_failed = self.blackboard.get(f"{self.robot_namespace}/system_failed")
            if system_failed:
                try:
                    failure_reason = self.blackboard.get(f"{self.robot_namespace}/failure_reason")
                    print(f"[{self.name}] System is in FAILED state: {failure_reason}")
                except KeyError:
                    print(f"[{self.name}] System is in FAILED state: Unknown reason")
                return py_trees.common.Status.FAILURE
            else:
                return py_trees.common.Status.SUCCESS
        except KeyError:
            # No failure flag set
            return py_trees.common.Status.SUCCESS


def create_root(robot_namespace="turtlebot0"):
    """Create behavior tree root node with failure handling and loop control"""
    root = py_trees.composites.Sequence(name="MainSequence", memory=True)
    
    # Initialize blackboard variables with proper permissions  
    init_bb = py_trees.composites.Sequence(name="InitSequence", memory=True)
    init_bb.add_children([
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize parcel index",
            variable_name=f"{robot_namespace}/current_parcel_index",
            variable_value=0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize pushing estimated time",
            variable_name=f"{robot_namespace}/pushing_estimated_time",
            variable_value=45.0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize system failure flag",
            variable_name=f"{robot_namespace}/system_failed",
            variable_value=False,
            overwrite=True
        )
    ])
    
    # Main loop condition check
    loop_condition = LoopCondition("CheckLoopCondition", robot_namespace)
    
    # Main pair sequence with failure handling
    pair_sequence_with_fallback = py_trees.composites.Selector(
        name="PairSequenceWithFallback", 
        memory=True
    )
    
    # Main pair sequence
    pair_sequence = py_trees.composites.Sequence(name="PairSequence", memory=True)
    
    # Action execution sequence with failure handling
    action_execution_with_fallback = py_trees.composites.Selector(
        name="ActionExecutionWithFallback",
        memory=True
    )
    
    # Main action execution sequence
    action_execution = py_trees.composites.Sequence(
        name="ActionExecution",
        memory=True
    )
    
    # Pushing sequence with failure handling
    pushing_sequence_with_fallback = py_trees.composites.Selector(
        name="PushingSequenceWithFallback", 
        memory=True
    )
    
    pushing_sequence = py_trees.composites.Sequence(name="PushingSequence", memory=True)
    pushing_sequence.add_children([
        WaitForPush("WaitingPush", 3000.0, robot_namespace, distance_threshold=0.14),
        ApproachObject("ApproachingPush", robot_namespace),
        PushObject("Pushing", robot_namespace, distance_threshold=0.14)
    ])
    
    # Add pushing sequence and its failure handler
    pushing_sequence_with_fallback.add_children([
        pushing_sequence,
        FailureHandler("PushingFailureHandler", robot_namespace)
    ])
    
    # Parallel execution with failure handling
    parallel_move_replan_with_fallback = py_trees.composites.Selector(
        name="ParallelMoveReplanWithFallback",
        memory=True
    )
    
    parallel_move_replan = py_trees.composites.Parallel(
        name="ParallelMoveReplan",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()  # Both must succeed
    )
    parallel_move_replan.add_children([
        MoveBackward("BackwardToSafeDistance", distance=0.2),
        ReplanPath("Replanning", 20.0, robot_namespace, "simple_maze")
    ])
    
    # Add parallel sequence and its failure handler
    parallel_move_replan_with_fallback.add_children([
        parallel_move_replan,
        FailureHandler("ParallelFailureHandler", robot_namespace)
    ])
    
    # Picking sequence with failure handling
    picking_up_sequence_with_fallback = py_trees.composites.Selector(
        name="PickingUpSequenceWithFallback",
        memory=True
    )
    
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitForPick("WaitingPick", 2.0, robot_namespace),
        PickObject("PickingUp", robot_namespace),
        StopSystem("Stop", 1.5)
    ])
    
    # Add picking sequence and its failure handler
    picking_up_sequence_with_fallback.add_children([
        picking_up_sequence,
        FailureHandler("PickingFailureHandler", robot_namespace)
    ])
    
    # Completion check sequence with failure handling
    completion_sequence_with_fallback = py_trees.composites.Selector(
        name="CompletionSequenceWithFallback",
        memory=True
    )
    
    completion_sequence = py_trees.composites.Sequence(name="CompletionSequence", memory=True)
    completion_sequence.add_children([
        CheckPairComplete("CheckPairComplete", robot_namespace),
        IncrementIndex("IncrementIndex", robot_namespace),
    ])
    
    # Add completion sequence and its failure handler
    completion_sequence_with_fallback.add_children([
        completion_sequence,
        FailureHandler("CompletionFailureHandler", robot_namespace)
    ])
    
    # Build the main execution sequence with all failure-protected components
    action_execution.add_children([
        pushing_sequence_with_fallback, 
        parallel_move_replan_with_fallback, 
        picking_up_sequence_with_fallback
    ])
    
    # Add action execution and its failure handler
    action_execution_with_fallback.add_children([
        action_execution,
        FailureHandler("ActionExecutionFailureHandler", robot_namespace)
    ])
    
    # Build the pair sequence with loop condition check and failure handling
    pair_sequence.add_children([
        loop_condition,  # Check if we should continue the loop
        action_execution_with_fallback, 
        completion_sequence_with_fallback
    ])
    
    # Add pair sequence and its failure handler
    pair_sequence_with_fallback.add_children([
        pair_sequence,
        FailureHandler("PairSequenceFailureHandler", robot_namespace)
    ])
    
    # Main loop decorator with failure-aware repeat
    main_loop = py_trees.decorators.Repeat(
        name="MainLoop",
        child=pair_sequence_with_fallback,
        num_success=-1  # Run indefinitely until failure
    )
    
    root.add_children([init_bb, main_loop])
    return root


def create_root_with_immediate_stop(robot_namespace="turtlebot0"):
    """Create behavior tree root node with immediate stop on any failure"""
    root = py_trees.composites.Sequence(name="MainSequence", memory=True)
    
    # Initialize blackboard variables
    init_bb = py_trees.composites.Sequence(name="InitSequence", memory=True)
    init_bb.add_children([
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize parcel index",
            variable_name=f"{robot_namespace}/current_parcel_index",
            variable_value=0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize pushing estimated time",
            variable_name=f"{robot_namespace}/pushing_estimated_time",
            variable_value=45.0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize system failure flag",
            variable_name=f"{robot_namespace}/system_failed",
            variable_value=False,
            overwrite=True
        )
    ])
    
    # Single pair execution with strict failure handling
    pair_sequence = py_trees.composites.Sequence(name="PairSequence", memory=True)
    
    # Action execution sequence - any failure stops everything
    action_execution = py_trees.composites.Sequence(
        name="ActionExecution",
        memory=True  # Stop on first failure
    )
    
    # Individual sequences (no failure recovery - let them fail)
    pushing_sequence = py_trees.composites.Sequence(name="PushingSequence", memory=True)
    pushing_sequence.add_children([
        WaitForPush("WaitingPush", 30.0, robot_namespace, distance_threshold=0.14),
        ApproachObject("ApproachingPush", robot_namespace),
        PushObject("Pushing", robot_namespace, distance_threshold=0.14)
    ])
    
    parallel_move_replan = py_trees.composites.Parallel(
        name="ParallelMoveReplan",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()  # Both must succeed
    )
    parallel_move_replan.add_children([
        MoveBackward("BackwardToSafeDistance", distance=0.2),
        ReplanPath("Replanning", 20.0, robot_namespace, "simple_maze")
    ])
    
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitForPick("WaitingPick", 2.0, robot_namespace),
        PickObject("PickingUp", robot_namespace),
        StopSystem("Stop", 1.5)
    ])
    
    completion_sequence = py_trees.composites.Sequence(name="CompletionSequence", memory=True)
    completion_sequence.add_children([
        CheckPairComplete("CheckPairComplete", robot_namespace),
        IncrementIndex("IncrementIndex", robot_namespace),
    ])
    
    # Build execution sequence
    action_execution.add_children([
        pushing_sequence, 
        parallel_move_replan, 
        picking_up_sequence
    ])
    
    pair_sequence.add_children([action_execution, completion_sequence])
    
    # Loop with condition check - stops immediately on failure
    loop_with_condition = py_trees.composites.Sequence(name="LoopWithCondition", memory=False)
    loop_with_condition.add_children([
        LoopCondition("CheckLoopCondition", robot_namespace),
        pair_sequence
    ])
    
    # Repeat decorator that continues until failure condition is met
    main_loop = py_trees.decorators.Repeat(
        name="MainLoop",
        child=loop_with_condition,
        num_success=-1  # Run until failure
    )
    
    root.add_children([init_bb, main_loop])
    return root


def create_pushing_sequence(robot_namespace="turtlebot0"):
    """Create just the pushing sequence for modular testing"""
    pushing_sequence = py_trees.composites.Sequence(name="PushingSequence", memory=True)
    pushing_sequence.add_children([
        WaitForPush("WaitingPush", 30.0, robot_namespace, distance_threshold=0.15),
        ApproachObject("ApproachingPush", robot_namespace),
        PushObject("Pushing", robot_namespace, distance_threshold=0.14)
    ])
    return pushing_sequence


def create_picking_sequence(robot_namespace="turtlebot0"):
    """Create just the picking sequence for modular testing"""
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitForPick("WaitingPick", 2.0, robot_namespace),
        PickObject("PickingUp", robot_namespace),
        StopSystem("Stop", 1.5)
    ])
    return picking_up_sequence


def create_parallel_move_replan_sequence(robot_namespace="turtlebot0"):
    """Create the parallel MoveBackward and ReplanPath sequence for modular testing"""
    parallel_move_replan = py_trees.composites.Parallel(
        name="ParallelMoveReplan",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()  # Both must succeed
    )
    parallel_move_replan.add_children([
        MoveBackward("BackwardToSafeDistance", distance=0.2),
        ReplanPath("Replanning", 20.0, robot_namespace, "simple_maze")
    ])
    return parallel_move_replan


def create_simple_test_tree(robot_namespace="turtlebot0"):
    """Create a simple test tree for debugging"""
    root = py_trees.composites.Sequence(name="TestSequence", memory=True)
    root.add_children([
        WaitAction("TestWait", 2.0, robot_namespace)
    ])
    return root


def create_robust_root(robot_namespace="turtlebot0"):
    """Create behavior tree with comprehensive failure handling and immediate stop"""
    root = py_trees.composites.Sequence(name="MainSequence", memory=True)
    
    # Initialize blackboard variables
    init_bb = py_trees.composites.Sequence(name="InitSequence", memory=True)
    init_bb.add_children([
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize parcel index",
            variable_name=f"{robot_namespace}/current_parcel_index",
            variable_value=0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize pushing estimated time",
            variable_name=f"{robot_namespace}/pushing_estimated_time",
            variable_value=45.0,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize system failure flag",
            variable_name=f"{robot_namespace}/system_failed",
            variable_value=False,
            overwrite=True
        ),
        py_trees.behaviours.SetBlackboardVariable(
            name="Initialize failure reason",
            variable_name=f"{robot_namespace}/failure_reason",
            variable_value="No failure",
            overwrite=True
        )
    ])
    
    # Create individual sequences with failure detection
    def create_sequence_with_failure_detection(name, children, component_name):
        """Helper function to create a sequence with failure detection"""
        sequence_with_fallback = py_trees.composites.Selector(
            name=f"{name}WithFallback",
            memory=True
        )
        
        main_sequence = py_trees.composites.Sequence(name=name, memory=True)
        main_sequence.add_children(children)
        
        sequence_with_fallback.add_children([
            main_sequence,
            FailureLogger(f"{name}FailureLogger", robot_namespace, component_name)
        ])
        
        return sequence_with_fallback
    
    # Create monitored sequences
    pushing_sequence = create_sequence_with_failure_detection(
        "PushingSequence",
        [
            WaitForPush("WaitingPush", 30.0, robot_namespace, distance_threshold=0.14),
            ApproachObject("ApproachingPush", robot_namespace),
            PushObject("Pushing", robot_namespace, distance_threshold=0.14)
        ],
        "Pushing Phase"
    )
    
    # Parallel execution with failure detection
    parallel_move_replan = py_trees.composites.Parallel(
        name="ParallelMoveReplan",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    parallel_move_replan.add_children([
        MoveBackward("BackwardToSafeDistance", distance=0.2),
        ReplanPath("Replanning", 20.0, robot_namespace, "simple_maze")
    ])
    
    parallel_sequence = create_sequence_with_failure_detection(
        "ParallelSequence",
        [parallel_move_replan],
        "Parallel Move/Replan Phase"
    )
    
    picking_sequence = create_sequence_with_failure_detection(
        "PickingUpSequence",
        [
            WaitForPick("WaitingPick", 2.0, robot_namespace),
            PickObject("PickingUp", robot_namespace),
            StopSystem("Stop", 1.5)
        ],
        "Picking Phase"
    )
    
    completion_sequence = create_sequence_with_failure_detection(
        "CompletionSequence",
        [
            CheckPairComplete("CheckPairComplete", robot_namespace),
            IncrementIndex("IncrementIndex", robot_namespace)
        ],
        "Completion Phase"
    )
    
    # Main pair execution with system monitor
    pair_execution = py_trees.composites.Sequence(name="PairExecution", memory=True)
    pair_execution.add_children([
        SystemMonitor("SystemMonitor", robot_namespace),  # Check system state first
        pushing_sequence,
        parallel_sequence,
        picking_sequence,
        completion_sequence
    ])
    
    # Loop decorator that stops on first failure
    main_loop = py_trees.decorators.Repeat(
        name="MainLoop",
        child=pair_execution,
        num_success=-1  # Run until failure
    )
    
    root.add_children([init_bb, main_loop])
    return root
