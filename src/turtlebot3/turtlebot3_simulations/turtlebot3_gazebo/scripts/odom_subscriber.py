#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from transforms3d.euler import quat2euler
class OdomSubscriber(Node):
    def __init__(self,namespace):
        super().__init__('odom_subscriber')
        self.namespace = namespace
        self.subscription = self.create_subscription(
            Odometry,
            f'{self.namespace}/pushing_robot/odom',
            self.odom_callback,
            10)
        self.subscription  # prevent unused variable warning

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Convert quaternion to Euler angles
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        euler = quat2euler(quaternion)
        yaw = euler[2]  # Yaw is the third element in the tuple

        self.get_logger().info(f'Position: x={position.x}, y={position.y}, z={position.z}')
        self.get_logger().info(f'Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}')
        self.get_logger().info(f'Roll: {euler[0]}') #  orientation
        self.get_logger().info(f'Pitch: {euler[1]}')
        self.get_logger().info(f'Yaw: {yaw}')

def main(args=None):
    rclpy.init(args=args)
    namespace = 'tb0'  # Change this to the desired namespace
    odom_subscriber = OdomSubscriber(namespace)
    rclpy.spin(odom_subscriber)
    odom_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()