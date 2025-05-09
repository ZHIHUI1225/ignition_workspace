#! /usr/bin/python3
import numpy as np
import rospy
import pickle
import time
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from pushing_msgs.srv import RobotTrajectory, RobotTrajectoryRequest, SetTarget, SetTargetRequest, GetPushingPaths, GetPushingPathsRequest, SetObstacles, SetObstaclesRequest


class FakeSystem:
    def __init__(self):
        self.obj_pub = rospy.Publisher(
            "/vrpn_client_node/Box/pose", PoseStamped, queue_size=1)
        self.target_pub = rospy.Publisher(
            "/target/pose", PoseStamped, queue_size=1)
        self.robot_pub = rospy.Publisher(
            "/vrpn_client_node/turtle5/pose", PoseStamped, queue_size=1)
        self.set_target_srv = rospy.Service(
            "/pushing_controller/SetTarget",   SetTarget, self.set_target)
        self.set_trajectory_srv = rospy.Service(
            "/pushing_controller/SetTrajectory",   RobotTrajectory, self.trajectory_cb)
        self.set_obstacles_srv = rospy.Service(
            "/pushing_controller/SetObstacles", SetObstacles, self.set_obstacles)
        self.set_trajectory_srv = rospy.Service(
            "/pushing_planner/GetPushingPlan",   GetPushingPaths, self.planner_cb)
        self.t = 0
        self.tmsg = PoseStamped()
        self.tmsg.pose.position.x = 1.9
        self.tmsg.pose.position.y = 1.0
        self.tmsg.pose.orientation.w = 1.0
        self.tmsg.header.frame_id = "world"
        self.omsg = PoseStamped()
        self.omsg.pose.position.x = -0.350
        self.omsg.pose.position.y = -0.50
        self.omsg.pose.orientation.w = 1.0
        self.omsg.header.frame_id = "world"
        self.rmsg = PoseStamped()
        self.rmsg.pose.position.x = -0.650
        self.rmsg.pose.position.y = -0.60
        self.rmsg.pose.orientation.w = 1.0
        self.rmsg.header.frame_id = "world"
        self.robot_poselist = np.array([[-0.65,-0.6,0.0]])
        self.robot_trajlist = np.array([])
        self.theta_diff = 0.0
        self.mode = "POSE"

    def set_obstacles(self, req: SetObstaclesRequest):
        print("set obstacle called")
        return True

    def set_target(self, req: SetTargetRequest):
        print("set target called")
        self.mode = "POSE"
        xi = self.rmsg.pose.position.x
        yi = self.rmsg.pose.position.y
        qi = self.rmsg.pose.orientation
        ti = euler_from_quaternion([qi.x,qi.y,qi.z,qi.w])[2]
        xf = req.target.x
        yf = req.target.y
        tf = req.target.theta
        dists = np.sqrt((xf-xi)**2 + (yf-yi)**2)
        num = np.ceil(dists*10/0.05)
        t = np.linspace(0,1,num)
        xnew = np.interp(t,[0,1],[xi,xf])
        ynew = np.interp(t,[0,1],[yi,yf])
        tnew = np.interp(t,[0,1],[ti,tf])
        self.robot_poselist = np.vstack([xnew,ynew,tnew]).T

        return True

    def trajectory_cb(self, req: RobotTrajectoryRequest):
        print("set trajectoy called")
        self.mode = "TRAJECTORY"
        self.robot_trajlist = np.ndarray(shape=(req.length,3),dtype=float)
        for i in range(req.length):
            self.robot_trajlist[i,0] = req.trajectory.data[i*5]
            self.robot_trajlist[i,1] = req.trajectory.data[i*5+1]
            self.robot_trajlist[i,2] = req.trajectory.data[i*5+2]

        oq = self.omsg.pose.orientation
        otheta = euler_from_quaternion([oq.x,oq.y,oq.z,oq.w])[2]
        self.theta_diff = otheta-req.trajectory.data[2]
        return True

    def planner_cb(self, req: GetPushingPathsRequest):
        print("plan called")
        time.sleep(3)
        with open('/home/federico/catkin_ws/src/pushing_behaviour_tree/res/objs.pkl','rb') as f:
            res = pickle.load(f)
            return res
        
    def step(self):
        self.rmsg.header.stamp = rospy.Time.now()
        self.omsg.header.stamp = rospy.Time.now()
        if self.mode == "POSE":
            # update robot position
            if self.robot_poselist.size > 0:
                self.rmsg.pose.position.x = self.robot_poselist[0,0]
                self.rmsg.pose.position.y = self.robot_poselist[0,1]
                eul = [0,0,self.robot_poselist[0,2]]
                quat = quaternion_from_euler(*eul)
                self.rmsg.pose.orientation.x = quat[0]
                self.rmsg.pose.orientation.y = quat[1]
                self.rmsg.pose.orientation.z = quat[2]
                self.rmsg.pose.orientation.w = quat[3]
                self.robot_poselist = self.robot_poselist[1:,:]
        elif self.mode == "TRAJECTORY":
            # update robot position
            b_ = 0.085
            if self.robot_trajlist.size > 0:
                self.rmsg.pose.position.x = self.robot_trajlist[0,0]
                self.rmsg.pose.position.y = self.robot_trajlist[0,1]
                eul = [0,0,self.robot_trajlist[0,2]]
                quat = quaternion_from_euler(*eul)
                self.rmsg.pose.orientation.x = quat[0]
                self.rmsg.pose.orientation.y = quat[1]
                self.rmsg.pose.orientation.z = quat[2]
                self.rmsg.pose.orientation.w = quat[3]
                self.omsg.pose.position.x = self.robot_trajlist[0,0] + b_*np.cos(self.robot_trajlist[0,2])
                self.omsg.pose.position.y = self.robot_trajlist[0,1] + b_*np.sin(self.robot_trajlist[0,2])
                otheta = self.robot_trajlist[0,2] + self.theta_diff
                eul = [0,0,otheta]
                quat = quaternion_from_euler(*eul)
                self.omsg.pose.orientation.x = quat[0]
                self.omsg.pose.orientation.y = quat[1]
                self.omsg.pose.orientation.z = quat[2]
                self.omsg.pose.orientation.w = quat[3]
                self.robot_trajlist = self.robot_trajlist[1:,:]
        

    def spin(self):
        r = rospy.Rate(10)
        while(not rospy.is_shutdown()):
            self.step()
            self.obj_pub.publish(self.omsg)
            self.target_pub.publish(self.tmsg)
            self.robot_pub.publish(self.rmsg)
            self.t = self.t+0.01
            r.sleep()


if __name__ == "__main__":
    rospy.init_node("FakeSystem")
    fs = FakeSystem()
    fs.spin()
