#!/usr/bin/env python3
"""
Test script for ReplanPath behavior node
"""

import sys
sys.path.append('/root/workspace/src/behaviour_tree')

import py_trees
from behaviour_tree.behaviors.ReplanPath_behaviour import ReplanPath

def test_replan_behavior():
    """Test the ReplanPath behavior"""
    
    # Create the behavior
    replan_behavior = ReplanPath("TestReplan", duration=2.0, robot_namespace="turtlebot0", case="simple_maze")
    
    # Initialize the behavior
    replan_behavior.initialise()
    
    # Test the behavior update
    print("Testing ReplanPath behavior...")
    print(f"Behavior name: {replan_behavior.name}")
    print(f"Robot ID: {replan_behavior.robot_id}")
    print(f"Case: {replan_behavior.case}")
    
    # Simulate a blackboard value
    replan_behavior.blackboard.set(f"{replan_behavior.robot_namespace}/pushing_estimated_time", 30.0)
    
    # Run one update cycle
    status = replan_behavior.update()
    print(f"Update status: {status}")
    
    print("ReplanPath behavior test completed!")

if __name__ == "__main__":
    test_replan_behavior()
