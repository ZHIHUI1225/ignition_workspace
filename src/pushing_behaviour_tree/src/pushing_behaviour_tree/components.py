from math import fabs, pi
import py_trees
import pickle
import math
from copy import deepcopy
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from math import sqrt
import py_trees_msgs.msg as py_trees_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from pushing_msgs.srv import RobotTrajectory, RobotTrajectoryRequest, GetPushingPathsRequest, GetPushingPathsResponse, GetPushingPaths, SetTargetRequest, SetTarget, SetObstacles, SetObstaclesRequest
from nav_msgs.msg import OccupancyGrid, Path
import imageio.v2 as imageio
import threading
import numpy as np
from pushing_behaviour_tree.traj_validator import check_traj, transform_trajectory


class CheckObjectInTarget(py_trees.behaviour.Behaviour):
    def __init__(self, name="CheckObjinTarg"):
        super(CheckObjectInTarget, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, unused):
        self.feedback_message = "setup"
        self.logger.debug(
            "%s.setup()"
            % (self.__class__.__name__)
        )
        self.blackboard.obj_pose = Pose()
        self.blackboard.target_pose = Pose()
        return True

    def update(self):
        self.feedback_message = "update"
        self.logger.debug(
            "%s.update()"
            % (self.__class__.__name__)
        )
        obj = self.blackboard.obj_pose
        target = self.blackboard.target_pose
        norm = sqrt((obj.position.x - target.position.x)**2 +
                    (obj.position.y - target.position.y)**2)
        if norm < 0.1:
            self.feedback_message = "object at target"
            return py_trees.common.Status.SUCCESS
        self.feedback_message = "object not at target"
        return py_trees.common.Status.FAILURE

    def terminate(self, unused):
        self.logger.debug(
            "%s.terminate()"
            % (self.__class__.__name__)
        )


class PositionRobot(py_trees.behaviour.Behaviour):
    def __init__(self, name="CheckPositionRobot"):
        super(PositionRobot, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, unused):
        self.feedback_message = "setup"
        self.logger.debug(
            "%s.setup()"
            % (self.__class__.__name__)
        )
        self.blackboard.sides = [0, 1, 2, 3]
        self.b_ = self.blackboard.get("b_")

        return True

    def update(self):

        self.feedback_message = "update"
        self.logger.debug(
            "%s.update()"
            % (self.__class__.__name__)
        )
        side = self.blackboard.current_side
        robot = self.blackboard.robot_pose
        q = [robot.orientation.x, robot.orientation.y,
             robot.orientation.z, robot.orientation.w]
        thetar = euler_from_quaternion(q)[2]
        pr = np.array([[robot.position.x], [robot.position.y], [thetar]])
        obj = self.blackboard.obj_pose
        thetao = q = [obj.orientation.x, obj.orientation.y,
                      obj.orientation.z, obj.orientation.w]
        thetao = euler_from_quaternion(q)[2]
        xr_d = obj.position.x + self.b_ * \
            math.cos(thetao + (side * (2/4)*math.pi))
        yr_d = obj.position.y + self.b_ * \
            math.sin(thetao + (side * (2/4)*math.pi))
        thetar_d = thetao + (side*(2/4)*math.pi) + math.pi
        thetar_d = np.arctan2(np.sin(thetar_d), np.cos(thetar_d))
        print("robot desired position")
        print([xr_d, yr_d, thetar_d])
        d = np.array([[xr_d], [yr_d], [thetar_d]]) - pr
        if (math.sqrt(d[0][0]*d[0][0] + d[1][0]*d[1][0]) < 0.025) & (np.abs(d[2][0]) < 0.1):
            self.feedback_message = "object is in correct position"
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "object is not in correct position"
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()"
            % (self.__class__.__name__)
        )
        return


class CheckRobotNearObject(py_trees.behaviour.Behaviour):
    def __init__(self, name="CheckRobotNear"):
        super(CheckRobotNearObject, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, unused):
        self.feedback_message = "setup"
        print("setup checkrobotnearobj")
        self.blackboard.obj_pose = Pose()
        self.blackboard.robot_pose = Pose()
        self.logger.debug(
            "%s.setup()"
            % (self.__class__.__name__)
        )
        return True

    def update(self):
        self.feedback_message = "update"
        self.logger.debug(
            "%s.update()"
            % (self.__class__.__name__)
        )
        obj = self.blackboard.obj_pose
        robot = self.blackboard.robot_pose
        dist = math.sqrt((obj.position.x - robot.position.x) ** 2
                         + (obj.position.y - robot.position.y) ** 2)
        if dist < 0.1:
            self.feedback_message = "robot near object"
            return py_trees.common.Status.FAILURE
        self.feedback_message = "robot far to object"
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()"
            % (self.__class__.__name__)
        )
        pass


class DetachfromObj(py_trees.behaviour.Behaviour):
    def __init__(self, name="DetachfromObj"):
        super(DetachfromObj, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.client = rospy.ServiceProxy(
            "/pushing_controller/SetTarget",   SetTarget)
        self.pub = rospy.Publisher(
            "pushing_tree/robot_target", PoseStamped, queue_size=1)

    def setup(self, unused):
        self.feedback_message = "setup"
        self.logger.debug(
            "%s.setup()"
            % (self.__class__.__name__)
        )
        self.Ds = 0.250
        return True

    def initialise(self):
        self.feedback_message = "initialise"
        self.logger.debug(
            "%s.initialize()"
            % (self.__class__.__name__)
        )
        robot = self.blackboard.robot_pose
        q = [robot.orientation.x, robot.orientation.y,
             robot.orientation.z, robot.orientation.w]
        thetar = euler_from_quaternion(q)[2]
        pr = np.array([robot.position.x, robot.position.y])
        obj = self.blackboard.obj_pose
        po = np.array([obj.position.x, obj.position.y])
        x = pr-po
        V = (self.Ds-np.linalg.norm(x))*1.2
        self.blackboard.psafe = pr + V * \
            np.array([math.cos(thetar+math.pi), math.sin(thetar+math.pi)])
        req = SetTargetRequest()
        req.enable_constraints = 2
        req.target.x = self.blackboard.psafe[0]
        req.target.y = self.blackboard.psafe[1]
        req.target.theta = thetar
        p_msg = PoseStamped()
        p_msg.pose.position.x = float(req.target.x)
        p_msg.pose.position.y = float(req.target.y)
        q = quaternion_from_euler(0, 0, thetar)
        p_msg.pose.orientation.x = q[0]
        p_msg.pose.orientation.y = q[1]
        p_msg.pose.orientation.z = q[2]
        p_msg.pose.orientation.w = q[3]
        self.pub.publish(p_msg)
        self.client(req)
        self.tend = rospy.Time.now()

    def update(self):
        self.feedback_message = "update"
        self.logger.debug(
            "%s.update()"
            % (self.__class__.__name__)
        )
        robot = self.blackboard.robot_pose
        pr = np.array([robot.position.x, robot.position.y])
        d = pr - self.blackboard.psafe
        if rospy.Time.now() < self.tend + rospy.Duration(30):
            if math.sqrt(d[0]**2 + d[1]**2) < 0.01:
                self.feedback_message = "robot is detached in first useful position"
                return py_trees.common.Status.RUNNING
            else:
                return py_trees.common.Status.RUNNING
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()"
            % (self.__class__.__name__)
        )
        pass


class MoveToApproach(py_trees.behaviour.Behaviour):
    def __init__(self, name="MovetoAppr"):
        super(MoveToApproach, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.client = rospy.ServiceProxy(
            "/pushing_controller/SetTarget",   SetTarget)
        self.set_obstacles = rospy.ServiceProxy(
            "pushing_controller/SetObstacles",   SetObstacles)
        self.pub = rospy.Publisher(
            "pushing_tree/robot_target", PoseStamped, queue_size=1)
        self.pr_d = []

    def setup(self, unused):
        self.feedback_message = "setup"
        print("setup movetoappr")
        self.blackboard.sides = [0, 1, 2, 3]
        self.obst = Pose()
        self.obstacles = deepcopy(self.blackboard.get("obstacles"))
        self.logger.debug(
            "%s.setup()"
            % (self.__class__.__name__)
        )
        return True
    def compute_target(self, obj_pose: Pose):
        side = self.blackboard.get("current_side")
        obj = obj_pose
        q = [obj.orientation.x, obj.orientation.y,
             obj.orientation.z, obj.orientation.w]
        thetao = euler_from_quaternion(q)[2]
        xr_d = obj.position.x + 0.25*math.cos(thetao + (side*(2/4)*math.pi))
        yr_d = obj.position.y + 0.25 * \
            math.sin(thetao + (side * (2/4) * math.pi))
        thetar_d = math.pi + side*(2/4)*math.pi + thetao
        thetar_d = np.arctan2(np.sin(thetar_d), np.cos(thetar_d))
        pr_d = np.array([xr_d, yr_d, thetar_d])
        return pr_d

    def initialise(self):
        self.logger.debug(
            "%s.initialize()"
            % (self.__class__.__name__)
        )
        self.feedback_message = "initialize"
        side = self.blackboard.get("current_side")
        obj = self.blackboard.obj_pose
        self.obst.position.x = obj.position.x
        self.obst.position.y = obj.position.y
        self.obstacles.append(self.obst)
        req = SetObstaclesRequest()
        req.obstacles = self.obstacles
        self.set_obstacles(req)
        q = [obj.orientation.x, obj.orientation.y,
             obj.orientation.z, obj.orientation.w]
        thetao = euler_from_quaternion(q)[2]
        xr_d = obj.position.x + 0.25*math.cos(thetao + (side*(2/4)*math.pi))
        yr_d = obj.position.y + 0.25 * \
            math.sin(thetao + (side * (2/4) * math.pi))
        thetar_d = math.pi + side*(2/4)*math.pi + thetao
        thetar_d = np.arctan2(np.sin(thetar_d), np.cos(thetar_d))
        self.pr_d = np.array([xr_d, yr_d, thetar_d])
        req = SetTargetRequest()
        req.enable_constraints = 2
        req.target.x = xr_d
        req.target.y = yr_d
        req.target.theta = thetar_d
        p_msg = PoseStamped()
        p_msg.pose.position.x = req.target.x
        p_msg.pose.position.y = req.target.y
        q = quaternion_from_euler(0, 0, thetar_d)
        p_msg.pose.orientation.x = q[0]
        p_msg.pose.orientation.y = q[1]
        p_msg.pose.orientation.z = q[2]
        p_msg.pose.orientation.w = q[3]
        self.pub.publish(p_msg)
        self.client(req)
        self.tend = rospy.Time.now()

    def update(self):
        self.feedback_message = "update"
        targ_now = self.compute_target(self.blackboard.obj_pose)
        robot = self.blackboard.robot_pose
        q = [robot.orientation.x, robot.orientation.y,
             robot.orientation.z, robot.orientation.w]
        thetar = euler_from_quaternion(q)[2]
        pr = np.array([robot.position.x, robot.position.y, thetar])
        if np.linalg.norm(targ_now-self.pr_d) > 0.001:
            req = SetTargetRequest()
            req.enable_constraints = 2
            req.target.x = targ_now[0]
            req.target.y = targ_now[1]
            req.target.theta = targ_now[2]
            self.pr_d = targ_now
            self.client(req)
            p_msg = PoseStamped()
            p_msg.pose.position.x = req.target.x
            p_msg.pose.position.y = req.target.y
            q = quaternion_from_euler(0, 0, targ_now[2])
            p_msg.pose.orientation.x = q[0]
            p_msg.pose.orientation.y = q[1]
            p_msg.pose.orientation.z = q[2]
            p_msg.pose.orientation.w = q[3]
            self.pub.publish(p_msg)
        print("Robot position %s" % pr)
        print("Robot desired position %s" % self.pr_d)
        dd = pr - self.pr_d
        self.logger.debug(
            "%s.update()[dd = %s]"
            % (self.__class__.__name__,dd)
        )
        # if rospy.Time.now() < self.tend + rospy.Duration(30):
        if math.sqrt(dd[0]**2 + dd[1]**2) < 0.02 and np.abs(dd[2]) < 0.1:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING
        # return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()"
            % (self.__class__.__name__)
        )
        pass


class Approach(py_trees.behaviour.Behaviour):
    def __init__(self, name="Approach"):
        super(Approach, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.client = rospy.ServiceProxy(
            "/pushing_controller/SetTarget",   SetTarget)
        self.set_obstacles = rospy.ServiceProxy(
            "pushing_controller/SetObstacles",   SetObstacles)
        self.pub = rospy.Publisher(
            "pushing_tree/robot_target", PoseStamped, queue_size=1)
        self.prd2 = []

    def setup(self, unused):
        self.feedback_message = "setup"
        self.logger.debug(
            "%s.setup()"
            % (self.__class__.__name__)
        )
        self.b_ = self.blackboard.get("b_")
        self.blackboard.sides = [0, 1, 2, 3]
        return True

    def initialise(self):
        self.feedback_message = "initalise"
        self.logger.debug(
            "%s.initialize()"
            % (self.__class__.__name__)
        )
        req = SetObstaclesRequest()
        req.obstacles = deepcopy(self.blackboard.get("obstacles"))
        self.set_obstacles(req)
        side = self.blackboard.current_side
        obj = self.blackboard.obj_pose
        q = [obj.orientation.x, obj.orientation.y,
             obj.orientation.z, obj.orientation.w]
        thetao = euler_from_quaternion(q)[2]
        xr_2 = obj.position.x + self.b_ * \
            math.cos(thetao + (side * (2/4)*math.pi))
        yr_2 = obj.position.y + self.b_ * \
            math.sin(thetao + (side * (2/4) * math.pi))
        thetar_2 = thetao + (side*(2/4)*math.pi) + math.pi
        thetar_2 = np.arctan2(np.sin(thetar_2), np.cos(thetar_2))
        self.prd2 = np.array([xr_2, yr_2, thetar_2])
        req = SetTargetRequest()
        req.enable_constraints = 0
        req.target.x = xr_2
        req.target.y = yr_2
        req.target.theta = thetar_2
        p_msg = PoseStamped()
        p_msg.pose.position.x = float(req.target.x)
        p_msg.pose.position.y = float(req.target.y)
        q = quaternion_from_euler(0, 0, thetar_2)
        p_msg.pose.orientation.x = q[0]
        p_msg.pose.orientation.y = q[1]
        p_msg.pose.orientation.z = q[2]
        p_msg.pose.orientation.w = q[3]
        self.pub.publish(p_msg)
        self.client(req)
        self.tend = rospy.Time.now()

    def update(self):
        self.feedback_message = "update"
        robot = self.blackboard.robot_pose
        q = [robot.orientation.x, robot.orientation.y,
             robot.orientation.z, robot.orientation.w]
        thetar = euler_from_quaternion(q)[2]
        pr = np.array([robot.position.x, robot.position.y, thetar])
        obj = self.blackboard.obj_pose
        q = [obj.orientation.x, obj.orientation.y,
             obj.orientation.z, obj.orientation.w]
        print(pr)
        print(self.prd2)
        d = pr - self.prd2
        self.logger.debug(
            "%s.update()"
            % (self.__class__.__name__)
        )
        new_status = py_trees.common.Status.INVALID
        if rospy.Time.now() < self.tend + rospy.Duration(30):
            if math.sqrt(d[0]**2 + d[1]**2) < 0.01 and np.abs(d[2]) < 0.1:
                new_status = py_trees.common.Status.SUCCESS
            else:
                new_status = py_trees.common.Status.RUNNING
        else:
            new_status = py_trees.common.Status.FAILURE
        self.logger.debug(
            "%s.update()[new_status = %s]"
            % (self.__class__.__name__, new_status)
        )
        return new_status

    def terminate(self, new_status):
        pass


class PushingTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self, name="PushingTraj"):
        super(PushingTrajectory, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.traj_in_progress = False
        self.plan = []

    def setup(self, unused):
        self.feedback_message = "setup"
        self.b_ = self.blackboard.get("b_")
        self.pub = rospy.Publisher(
            "pushing_tree/object_target", Path, queue_size=1)
        rospy.wait_for_service('pushing_controller/SetTrajectory', 10)
        self.set_trajectory = rospy.ServiceProxy(
            'pushing_controller/SetTrajectory', RobotTrajectory)
        self.resume_client = rospy.ServiceProxy(
            'pushing_controller/Resume', Trigger)
        self.stop_client = rospy.ServiceProxy(
            'pushing_controller/Stop', Trigger)
        self.logger.debug(
            "%s.setup()"
            % (self.__class__.__name__)
        )
        return True

    def path_to_trajectory_msg(self, path):
        traj = Float64MultiArray()
        q = path.poses[0].pose.orientation
        old_theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        for pose in path.poses:
            q = pose.pose.orientation
            theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            traj.data.append(pose.pose.position.x + self.b_*math.cos(theta))
            traj.data.append(pose.pose.position.y + self.b_*math.sin(theta))
            traj.data.append(theta)
            traj.data.extend([0.05, ((theta-old_theta)/0.1)])
            old_theta = theta
        return traj

    def initialise(self):
        self.logger.debug(
            "%s.initialize()"
            % (self.__class__.__name__)
        )
        self.feedback_message = "initialise"
        
        if self.blackboard.get("traj_in_progress"):
            self.feedback_message = "resuming trajectory"
            self.resume_client()
            self.logger.debug(
                "%s.resuming"
                % (self.__class__.__name__)
            )
        else:
            self.logger.debug(
                "%s.initiating a new push"
                % (self.__class__.__name__)
            )
            self.plan = self.blackboard.get("plan")
            self.path_target_ = self.plan.paths.pop(0)
            self.blackboard.set("current_path", self.path_target_)
            self.pub.publish(self.path_target_)
            self.blackboard.set("current_side", self.plan.sides.pop(0))
            req = RobotTrajectoryRequest()
            req.length = len(self.path_target_.poses)
            req.constraints = 1
            req.trajectory = self.path_to_trajectory_msg(self.path_target_)
            resp = self.set_trajectory(req)
            print("sending trajectory")
            self.tend = rospy.Time.now() + rospy.Duration(req.length*0.1+2.5)
            target = self.path_target_.poses[-1].pose
            q = [target.orientation.x, target.orientation.y,
                 target.orientation.z, target.orientation.w]
            thetat = euler_from_quaternion(q)[2]
            self.blackboard.pt = np.array(
                [target.position.x, target.position.y, thetat])
            self.blackboard.set("traj_in_progress",True)

    def update(self):
        self.feedback_message = "pushing update"
        obj = self.blackboard.obj_pose
        q = [obj.orientation.x, obj.orientation.y,
             obj.orientation.z, obj.orientation.w]
        thetao = euler_from_quaternion(q)[2]
        po = np.array([obj.position.x, obj.position.y, thetao])
        d = po - self.blackboard.pt
        self.logger.debug(
            "%s.update()"
            % (self.__class__.__name__)
        )
        if rospy.Time.now() < self.tend + rospy.Duration(30.0):
            return py_trees.common.Status.RUNNING
        else:
            self.blackboard.set("traj_in_progress", False)
            self.logger.debug(
                "%s.update()[%s]"
                % (self.__class__.__name__, "trajectory completed")
            )
            if math.sqrt(d[0]**2 + d[1]**2) < 0.01 and np.abs(d[2] < 0.1):
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        
        if new_status == py_trees.common.Status.FAILURE or new_status == py_trees.common.Status.INVALID:
            # self.stop_client()
            print("Trajectory terminate")
        pass


class ComputeTrajectory(py_trees.behaviour.Behaviour):
    def __init__(self, name="ComputeTraj"):
        super(ComputeTrajectory, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, unused):
        self.feedback_message = "setup"
        print("setup computetraj")
        self.get_plan = rospy.ServiceProxy(
            '/pushing_planner/GetPushingPlan', GetPushingPaths)
        self.stop_robot = rospy.ServiceProxy(
            '/pushing_controller/Stop', Trigger)
        self.plan = []
        self.b_ = self.blackboard.get("b_")
        self.map_msg = self.load_map(
            '/home/filippo/pushing_ws/src/pushing_supervisor/res/map6.png')
        self.R_p = 0.47
        self.R_m = -0.47
        self.side_nr = 4
        self.lock = threading.Lock()
        return True

    def initialise(self):
        self.feedback_message = "initialize"
        print("initialise computetraj")
        self.stop_robot()
        req = GetPushingPathsRequest()
        # Compose request
        self.target = self.blackboard.target_pose
        self.obj = self.blackboard.obj_pose

        req.start.x = self.obj.position.x  # posizione oggetto
        req.start.y = self.obj.position.y
        q = [self.obj.orientation.x, self.obj.orientation.y,
             self.obj.orientation.z, self.obj.orientation.w]
        req.start.theta = euler_from_quaternion(q)[2]  # posizione oggetto
        req.goal.x = self.target.position.x
        req.goal.y = self.target.position.y
        q = [self.target.orientation.x, self.target.orientation.y,
             self.target.orientation.z, self.target.orientation.w]
        req.goal.theta = euler_from_quaternion(q)[2]  # posizione target
        req.map = self.map_msg
        req.R_p = self.R_p  # raggio sterzata sinistra
        req.R_m = self.R_m  # raggio sterzata destra
        req.b_ = self.b_  # distanza quando oggetto e robot sono vicini
        req.sides_nr = self.side_nr  # numero di lati (5 se è un pentagono)
        self.t1 = threading.Thread(target=self.ask_planner, args=([req]))
        self.t1.start()  # avvio il thread
        self.blackboard.set("planning",True)

    def update(self):
        self.feedback_message = "update"
        print("update computetraj")
        self.logger.debug(
            "%s.update()"
            % (self.__class__.__name__)
        )
        if self.t1.is_alive():
            return py_trees.common.Status.RUNNING
        if self.plan:
            self.feedback_message = "traj is computed"
            return py_trees.common.Status.RUNNING
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )

    def ask_planner(self, req):
        print("planning Request")
        with self.lock:
            try:
                self.plan = self.get_plan(req)
                self.plan.sides = list(self.plan.sides)
                self.blackboard.set("plan", self.plan)
                self.blackboard.set("current_side", self.plan.sides[0])
                self.blackboard.set("current_path", self.plan.paths[0])
                # Open a file and use dump()
                # with open('file_paths.pkl', 'wb') as file:
                # A new file will be created
                # pickle.dump(self.plan, file)
            except:
                self.plan = None
            finally:
                self.blackboard.set("planning",False)

    def load_map(self, path):
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
                    obspose.position.y = map_msg.info.origin.position.y + \
                        (i+0.5)*map_msg.info.resolution
                    obspose.position.x = map_msg.info.origin.position.x + \
                        (j+0.5)*map_msg.info.resolution
                    obspose.orientation.w = 1.0
                    obstacles.append(obspose)
        return map_msg


class CheckPushingPaths(py_trees.behaviour.Behaviour):
    def __init__(self, name="CheckPushTraj"):
        super(CheckPushingPaths, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.curr_path = None
        self.R_p = 0.47

    def setup(self, unused):
        self.feedback_message = "setup"
        self.stop_robot = rospy.ServiceProxy(
            '/pushing_controller/Stop', Trigger)
        self.b_ = self.blackboard.get("b_")
        print("setup checkpushtraj")
        return True

    def initialise(self):
        self.feedback_message = "initialize"
        print("initialise checkpushtraj")

    def update(self):
        self.feedback_message = "update"
        self.logger.debug(
            "%s.update()[%s]"
            % (self.__class__.__name__, self.feedback_message)
        )
        self.resp = self.blackboard.get("plan")
        self.curr_path = self.blackboard.get("current_path")
        self.curr_side = self.blackboard.get("current_side")
        if self.blackboard.get("planning"):
            return py_trees.common.Status.FAILURE
        if not self.resp:
            self.feedback_message = "response empty"
            self.logger.debug(
                "%s.update()[%s]"
                % (self.__class__.__name__, self.feedback_message)
            )
            return py_trees.common.Status.FAILURE
        return self.check_tunnel()

    def check_tunnel(self):
        if not self.curr_path.poses:
            self.feedback_message = "no current path"
            self.logger.debug(
                "%s.update()[%s]"
                % (self.__class__.__name__, self.feedback_message)
            )
            return py_trees.common.Status.FAILURE

        try:
            #mpc_pred = self.blackboard.get("mpc_path")
            mpc_ref = self.blackboard.get("mpc_ref")
        except:
            self.feedback_message = "default on success since no data is available"
            return py_trees.common.Status.SUCCESS
        t_in_progress = self.blackboard.get("traj_in_progress")
        if not t_in_progress:
            self.feedback_message =  "default on success since not pushing"
            return py_trees.common.Status.SUCCESS
        
        obj_x = self.blackboard.obj_pose.position.x
        obj_y = self.blackboard.obj_pose.position.y
        q = [self.blackboard.obj_pose.orientation.x, self.blackboard.obj_pose.orientation.y,
             self.blackboard.obj_pose.orientation.z, self.blackboard.obj_pose.orientation.w]
        thetao = euler_from_quaternion(
            q)[2] + (self.curr_side*(2/4)*math.pi) + math.pi
        p_r = (obj_x-self.b_*math.cos(thetao),obj_y-self.b_*math.sin(thetao),thetao)
        robot_traj = transform_trajectory(mpc_ref)

        if not check_traj(p_r,robot_traj,self.R_p, self.blackboard.get("obstacles")):
            self.feedback_message = "no valid trajectory found"
            return py_trees.common.Status.FAILURE
        # if dist(mpc_pred[0].pose,self.blackboard.obj_pose)[0] < 0.01:
        #     d_start = dist(mpc_pred[0].pose,mpc_ref[0].pose)[0]
        #     d_end = dist(mpc_pred[-1].pose,mpc_ref[-1].pose)[0]
        #     if (d_start > d_end):
        #         return py_trees.common.Status.SUCCESS 
        #     else:
        #         self.feedback_message = "mpc not converging"
        #         return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS
        # obj_x = self.blackboard.obj_pose.position.x
        # obj_y = self.blackboard.obj_pose.position.y
        # q = [self.blackboard.obj_pose.orientation.x, self.blackboard.obj_pose.orientation.y,
        #      self.blackboard.obj_pose.orientation.z, self.blackboard.obj_pose.orientation.w]
        # thetao = euler_from_quaternion(
        #     q)[2] + (self.curr_side*(2/4)*math.pi) + math.pi

        # def theta_from_quat(a):
        #     q = [a.pose.orientation.x, a.pose.orientation.y,
        #          a.pose.orientation.z, a.pose.orientation.w]
        #     return euler_from_quaternion(q)[2]

        # def dist(a): return (sqrt(((obj_x - a.pose.position.x)**2) + ((obj_y -
        #                                                                a.pose.position.y)**2)), transform_to_pipi(thetao-theta_from_quat(a))[0])
        # dlist = map(dist, self.curr_path.poses)
        # d = min(dlist, key=lambda x: x[0])

        # if d[0] > 0.15 or abs(d[1]) > 0.34:
        #     self.logger.debug(
        #         "%s.update()[distance = %s]"
        #         % (self.__class__.__name__, d)
        #     )
        #     return py_trees.common.Status.FAILURE
        # else:
        #     self.logger.debug(
        #         "%s.update()[distance OK]"
        #         % (self.__class__.__name__)
        #     )
        #     return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.FAILURE and self.status == py_trees.common.Status.SUCCESS:
            self.stop_robot()
            self.blackboard.set("traj_in_progress",False)    
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )


def transform_to_pipi(input_angle):
    revolutions = int((input_angle + np.sign(input_angle) * pi) / (2 * pi))

    p1 = truncated_remainder(input_angle + np.sign(input_angle) * pi, 2 * pi)
    p2 = (np.sign(np.sign(input_angle)
                  + 2 * (np.sign(fabs((truncated_remainder(input_angle + pi, 2 * pi))
                                      / (2 * pi))) - 1))) * pi

    output_angle = p1 - p2

    return output_angle, revolutions


def truncated_remainder(dividend, divisor):
    divided_number = dividend / divisor
    divided_number = \
        -int(-divided_number) if divided_number < 0 else int(divided_number)

    remainder = dividend - divisor * divided_number

    return remainder

def theta_from_quat(a: Pose):
            q = [a.orientation.x, a.orientation.y,
                 a.orientation.z, a.orientation.w]
            return euler_from_quaternion(q)[2]

def dist(a: Pose,b: Pose): 
    return (sqrt(((a.position.x - b.position.x)**2) + ((a.position.y  - b.position.y)**2)), transform_to_pipi(theta_from_quat(a)-theta_from_quat(b))[0])