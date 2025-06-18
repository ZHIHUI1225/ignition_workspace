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


class FailureHandler(py_trees.behaviour.Behaviour):
    """Simple behavior to handle failures and set blackboard variables"""
    
    def __init__(self, name, robot_namespace="turtlebot0"):
        super().__init__(name)
        self.robot_namespace = robot_namespace
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key=f"{robot_namespace}/system_failed",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key=f"{robot_namespace}/failure_context",
            access=py_trees.common.Access.READ
        )
    
    def update(self):
        print(f"[{self.name}][{self.robot_namespace}] FAILURE DETECTED - STOPPING SYSTEM")
        
        # Set system failure flag
        self.blackboard.set(f"{self.robot_namespace}/system_failed", True)
        
        # Try to read failure context set by individual nodes
        try:
            failure_context = self.blackboard.get(f"{self.robot_namespace}/failure_context")
            if failure_context:
                print(f"[{self.name}][{self.robot_namespace}] Failure context: {failure_context}")
        except KeyError:
            print(f"[{self.name}][{self.robot_namespace}] No failure context available")
        
        # Return FAILURE to stop the loop from restarting
        return py_trees.common.Status.FAILURE


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
                print(f"[{self.name}][{self.robot_namespace}] System failure detected - preventing loop restart")
                return py_trees.common.Status.FAILURE
            else:
                return py_trees.common.Status.SUCCESS
        except KeyError:
            # If key doesn't exist, system hasn't failed
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
        PushObject("Pushing", robot_namespace, distance_threshold=0.09)
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
        PickObject("PickingUp", robot_namespace, timeout=100.0),
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
