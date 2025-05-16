#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import tf2_ros
from tf2_ros import TransformException

class TurtleBotPosePublisher(Node):
    def __init__(self):
        super().__init__('turtlebot_pose_publisher')
        self.publisher = self.create_publisher(Pose, 'turtlebot_pose', 10)
        # create TF buffer & listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.publish_pose)

    def publish_pose(self):
        pose_msg = Pose()
        try:
            # lookup transform from world → turtlebot1
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('world', 'turtlebot1', now)
            # fill Pose from transform
            pose_msg.position = trans.transform.translation
            pose_msg.orientation = trans.transform.rotation
            self.publisher.publish(pose_msg)
            self.get_logger().info(f'Publishing: {pose_msg}')
        except TransformException as e:
            self.get_logger().warn(f'Cannot get transform: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()