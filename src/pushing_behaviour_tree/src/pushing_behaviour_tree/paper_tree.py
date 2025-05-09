#!/usr/bin/env python

import py_trees
import py_trees_ros
import py_trees_msgs.msg as py_trees_msgs
from geometry_msgs.msg import PoseStamped
import rospy
import pushing_behaviour_tree as pbt
import functools
import sys
import py_trees.console as console
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid, Path
import py_trees.decorators
import py_trees.display

def create_root():
    # behaviours
    
    selector1 = py_trees.composites.Selector(name="?\nManipulation\nTask")
    obj_at_target = pbt.CheckObjectInTarget(name="Is object\n at target?")
    sequence1 = py_trees.composites.Sequence(name="->\nManipulate")
    #sequence1 = py_trees.composites.Sequence(name="Manipulate")
    selector2 = py_trees.composites.Selector(name="?\nIs there a\nvalid \ntrajectory?")
    check_push_traj = pbt.CheckPushingPaths(name="Is current\ntrajectory\nvalid?")
    compute_traj = py_trees_ros.actions.ActionClient(name="Compute\nTrajectory")
    
    selector3 = py_trees.composites.Selector(name="?\nCan Robot\nPush?")
    position_robot = pbt.PositionRobot(name="Is robot\n in correct\n contact?")
    sequence2 = py_trees.composites.Sequence(name="->*\nReposition\nRobot")
    selector4 = py_trees.composites.Selector(name="?\nCan robot\nmove safely?")
    robot_near_object = pbt.CheckRobotNearObject(name="Is robot\nfar from\nthe object?")
    detach_from_object = pbt.DetachfromObj(name="Move away \nfrom\nthe object")
    move_to_approach = pbt.MoveToApproach(name="Align with\n desired \nobject side")
    approach_to_obj = pbt.Approach(name="Move to \ncontact\n position")
    execute_pushing_traj = pbt.PushingTrajectory(name="Push\nalong\ntrajectory")

    root = py_trees.decorators.Decorator(name="ROOT",child=selector1)
    # struttura albero
    
    selector1.add_children([obj_at_target,sequence1])
    sequence1.add_children([selector2,selector3, execute_pushing_traj])
    selector2.add_children([check_push_traj,compute_traj])
    selector3.add_children([position_robot,sequence2])
    sequence2.add_children([selector4,move_to_approach,approach_to_obj])
    selector4.add_children([robot_near_object, detach_from_object])
    return root


if __name__ == '__main__':
    root = create_root()    
    py_trees.display.render_dot_tree(root,name="paper_tree")