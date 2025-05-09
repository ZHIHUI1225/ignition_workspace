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
import imageio.v2 as imageio
from nav_msgs.msg import OccupancyGrid, Path


def pre_tick_handler(behaviour_tree):
    """
    This prints a banner and will run immediately before every tick of the tree.

    Args:
        behaviour_tree (:class:`~py_trees.trees.BehaviourTree`): the tree custodian

    """
    print("\n--------- Run %s ---------\n" % behaviour_tree.count)
    
def create_root():
    # behaviours
    root = py_trees.composites.Parallel(name="ROOT")
    # topic sequence
    topic_par = py_trees.composites.Parallel(name="Data Acquisition")
    # riporto sulla BB la posizione dell'oggetto
    obj_pose2BB = py_trees_ros.subscribers.ToBlackboard(
        name="obj2BB", topic_name="/vrpn_client_node/Box/pose", topic_type=PoseStamped, blackboard_variables={'obj_pose': 'pose'})
    # riporto sulla BB la posizione del robot
    robot_pose2BB = py_trees_ros.subscribers.ToBlackboard(
        name="robot2BB", topic_name="/vrpn_client_node/turtle5/pose", topic_type=PoseStamped, blackboard_variables={'robot_pose': 'pose'})
    # riporto sulla BB la posizione target dell'oggetto
    target_pose2BB = py_trees_ros.subscribers.ToBlackboard(
        name="target2BB", topic_name="target/pose", topic_type=PoseStamped, blackboard_variables={'target_pose': 'pose'})
    target_pose2BB = py_trees_ros.subscribers.ToBlackboard(
        name="target2BB", topic_name="target/pose", topic_type=PoseStamped, blackboard_variables={'target_pose': 'pose'})
    mpc_path2BB = py_trees_ros.subscribers.ToBlackboard(
        name="mpc_path2BB", topic_name="/pusher/path", topic_type=Path, blackboard_variables={'mpc_path': 'poses'})
    mpc_ref2BB = py_trees_ros.subscribers.ToBlackboard(
        name="mpc_path2BB", topic_name="/pusher/reference", topic_type=Path, blackboard_variables={'mpc_ref': 'poses'})
    always_running = py_trees.behaviours.Running(name="Idle")
    always_running2= py_trees.behaviours.Running(name="AlwaysRunning2")
    selector1 = py_trees.composites.Selector(name="Manipulation Task")
    obj_at_target = pbt.CheckObjectInTarget(name="Is object\n at target?")
    sequence1 = pbt.SequenceNoMem(name="SequenceNoMem",memory=False)
    #sequence1 = py_trees.composites.Sequence(name="Manipulate")
    selector2 = py_trees.composites.Selector(name="Is there a valid trajectory?")
    check_push_traj = pbt.CheckPushingPaths(name="Check\nTrajectory")
    compute_traj = pbt.ComputeTrajectory(name="Compute\nTrajectory")
    
    selector3 = py_trees.composites.Selector(name="Can Robot Push")
    position_robot = pbt.PositionRobot(name="Is robot\n in correct\n contact?")
    sequence2 = py_trees.composites.Sequence(name="Reposition Robot")
    selector4 = py_trees.composites.Selector(name="Is robot\n safe to move?")
    robot_near_object = pbt.CheckRobotNearObject(name="Is robot far\n from the object?")
    detach_from_object = pbt.DetachfromObj(name="Move away \nfrom the object")
    move_to_approach = pbt.MoveToApproach(name="Align with\n desired \nobject side")
    approach_to_obj = pbt.Approach(name="Move to \ncontact\n position")
    execute_pushing_traj = pbt.PushingTrajectory(name="Push along\n trajectory")

    # struttura albero
    root.add_children([topic_par, selector1])
    topic_par.add_children([obj_pose2BB, robot_pose2BB, target_pose2BB,mpc_path2BB,mpc_ref2BB])
    selector1.add_children([obj_at_target,sequence1,always_running])
    sequence1.add_children([selector2,selector3, execute_pushing_traj])
    selector2.add_children([check_push_traj,compute_traj])
    selector3.add_children([position_robot,sequence2])
    sequence2.add_children([selector4,move_to_approach,approach_to_obj])
    selector4.add_children([robot_near_object, detach_from_object])
    blk = py_trees.blackboard.Blackboard()
    obst = load_map('/home/filippo/pushing_ws/src/pushing_supervisor/res/map6.png')
    blk.set("obstacles",obst)
    blk.set("b_",0.08)
    blk.set("traj_in_progress",False)
    blk.set("planning",False)
    

    return root


def shutdown(behaviour_tree):
    behaviour_tree.interrupt()

def load_map(path):
        if path == "":
            return
        im = imageio.imread(path)
        map_msg = OccupancyGrid()
        map_msg.info.origin.position.x = -0.75
        map_msg.info.origin.position.y = -1.0
        map_msg.info.origin.orientation.w = 1.0
        map_msg.info.resolution = 0.05
        map_msg.info.width = im.shape[1]
        map_msg.info.height = im.shape[0]
        map_msg.header.frame_id = "world"
        im = 100 - im / 255 * 100
        count = 0
        obstacles = []
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                map_msg.data.append(int(im[im.shape[0]-1-i][j]))
                if im[im.shape[0]-1-i][j] > 70:
                    obspose = Pose()
                    obspose.position.y = map_msg.info.origin.position.y + (i+0.5)*map_msg.info.resolution
                    obspose.position.x = map_msg.info.origin.position.x + (j+0.5)*map_msg.info.resolution
                    obspose.orientation.w = 1.0
                    obstacles.append(obspose)
        return obstacles

def main():
    print("main")
    rospy.init_node("tree")
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    root = create_root()
    behaviour_tree = py_trees_ros.trees.BehaviourTree(root)
    
    rospy.on_shutdown(functools.partial(shutdown, behaviour_tree))
    if not behaviour_tree.setup(timeout=15):
        console.logerror("failed to setup the tree,aborting.")
        sys.exit(1)
    behaviour_tree.tick_tock(200,pre_tick_handler=pre_tick_handler)


if __name__ == "__main__":
    main()
