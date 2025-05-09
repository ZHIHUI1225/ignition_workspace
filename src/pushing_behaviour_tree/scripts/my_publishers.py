#!/usr/bin/env python

import math
import rospy
from geometry_msgs.msg import PoseStamped
if __name__=="__main__":
    rospy.init_node("Publishers")
    p1 = rospy.Publisher("/obj/pose",PoseStamped,queue_size=1)
    p2 = rospy.Publisher("/target/pose",PoseStamped,queue_size=1)
    p3 = rospy.Publisher("/robot/pose",PoseStamped,queue_size=1)

    r = rospy.Rate(100)
    t = 0
    tmsg= PoseStamped()
    tmsg.pose.position.x = 1.0
    omsg = PoseStamped()
    while(not rospy.is_shutdown()):
        omsg.pose.position.x = math.cos(t)
        omsg.pose.position.y = math.sin(t)
        p1.publish(omsg)
        p2.publish(tmsg)
        p3.publish(tmsg)
        t = t+0.01
        r.sleep()
  