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


def create_root(robot_namespace="turtlebot0"):
    """Create behavior tree root node - optimized structure version"""
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
        )
    ])
    
    # Main loop structure with fixed Repeat decorator
    pair_sequence = py_trees.composites.Sequence(name="PairSequence", memory=True)
    
    # Action execution sequence - pushing must complete before picking starts
    action_execution = py_trees.composites.Sequence(
        name="ActionExecution",
        memory=True
    )
    
    # Pushing sequence - without MoveBackward (moved to parallel section)
    pushing_sequence = py_trees.composites.Sequence(name="PushingSequence", memory=True)
    pushing_sequence.add_children([
        WaitForPush("WaitingPush", 30.0, robot_namespace, distance_threshold=0.14),
        ApproachObject("ApproachingPush", robot_namespace),
        PushObject("Pushing", robot_namespace, distance_threshold=0.14)
    ])
    
    # Parallel execution: MoveBackward and ReplanPath run simultaneously
    parallel_move_replan = py_trees.composites.Parallel(
        name="ParallelMoveReplan",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()  # Both must succeed
    )
    parallel_move_replan.add_children([
        MoveBackward("BackwardToSafeDistance", distance=0.2),
        ReplanPath("Replanning", 20.0, robot_namespace, "simple_maze")
    ])
    
    # PickingUp sequence - only pick and stop (wait and replan moved to parallel section)
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitForPick("WaitingPick", 2.0, robot_namespace),
        PickObject("PickingUp", robot_namespace),
        StopSystem("Stop", 1.5)
    ])
    
    # Completion check sequence with blackboard variable reset
    completion_sequence = py_trees.composites.Sequence(name="CompletionSequence", memory=True)
    completion_sequence.add_children([
        CheckPairComplete("CheckPairComplete", robot_namespace),
        IncrementIndex("IncrementIndex", robot_namespace),
    ])
    
    # Combine structure with proper node connections - sequential execution with parallel middle section
    action_execution.add_children([pushing_sequence, parallel_move_replan, picking_up_sequence])
    pair_sequence.add_children([action_execution, completion_sequence])
    
    # Main loop decorator with correct usage
    main_loop = py_trees.decorators.Repeat(
        name="MainLoop",
        child=pair_sequence,
        num_success=2  # Run only 2 iterations for testing
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
