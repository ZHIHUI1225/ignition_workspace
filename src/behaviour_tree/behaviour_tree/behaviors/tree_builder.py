#!/usr/bin/env python3
"""
Behavior tree structure and creation functions.
Contains the main tree building logic and sequences.
"""

import py_trees
from .basic_behaviors import (
    ResetFlags, WaitAction, WaitForPush, WaitForPick, ReplanPath, StopSystem, 
    CheckPairComplete, IncrementIndex, PrintMessage
)
from .movement_behaviors import ApproachObject, MoveBackward
from .manipulation_behaviors import PushObject, PickObject


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
        ResetFlags("ResetFlags")
    ])
    
    # Main loop structure with fixed Repeat decorator
    pair_sequence = py_trees.composites.Sequence(name="PairSequence", memory=True)
    
    # Action execution sequence - pushing must complete before picking starts
    action_execution = py_trees.composites.Sequence(
        name="ActionExecution",
        memory=True
    )
    
    # Pushing sequence - direct class calls
    pushing_sequence = py_trees.composites.Sequence(name="PushingSequence", memory=True)
    pushing_sequence.add_children([
        WaitForPush("WaitingPush", 30.0, robot_namespace),
        ApproachObject("ApproachingPush", robot_namespace),
        PushObject("Pushing", robot_namespace),
        MoveBackward("BackwardToSafeDistance")
    ])
    
    # PickingUp sequence - direct class calls
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitForPick("WaitingPick", 2.0, robot_namespace),
        ReplanPath("Replanning", 2.0),
        PickObject("PickingUp"),
        StopSystem("Stop", 1.5)
    ])
    
    # Completion check sequence with blackboard variable reset
    completion_sequence = py_trees.composites.Sequence(name="CompletionSequence", memory=True)
    completion_sequence.add_children([
        CheckPairComplete("CheckPairComplete"),
        ResetFlags("ResetFlags"),
        IncrementIndex("IncrementIndex", robot_namespace),
        PrintMessage(
            name="PrintCompletedPair",
            message=lambda blackboard: f"Completed pair, current_parcel_index: {getattr(blackboard, f'{robot_namespace}/current_parcel_index', 0)}"
        )
    ])
    
    # Combine structure with proper node connections
    action_execution.add_children([pushing_sequence, picking_up_sequence])
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
        WaitForPush("WaitingPush", 30.0, robot_namespace),
        ApproachObject("ApproachingPush", robot_namespace),
        PushObject("Pushing", robot_namespace),
        MoveBackward("BackwardToSafeDistance")
    ])
    return pushing_sequence


def create_picking_sequence(robot_namespace="turtlebot0"):
    """Create just the picking sequence for modular testing"""
    picking_up_sequence = py_trees.composites.Sequence(name="PickingUpSequence", memory=True)
    picking_up_sequence.add_children([
        WaitForPick("WaitingPick", 2.0, robot_namespace),
        ReplanPath("Replanning", 2.0),
        PickObject("PickingUp"),
        StopSystem("Stop", 1.5)
    ])
    return picking_up_sequence


def create_simple_test_tree(robot_namespace="turtlebot0"):
    """Create a simple test tree for debugging"""
    root = py_trees.composites.Sequence(name="TestSequence", memory=True)
    root.add_children([
        WaitAction("TestWait", 2.0, robot_namespace),
        PrintMessage("TestMessage", "Simple test completed!")
    ])
    return root
