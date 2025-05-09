#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from tf2_geometry_msgs import do_transform_pose_stamped
import math

class TurtlebotPose(Node):
    def __init__(self,namespace):
        super().__init__('turtlebot_pose_node')
        # allow remapping robot namespace
        ns = namespace  # Strip leading/trailing slashes

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Store latest odometry message
        self.latest_odom: Odometry = None
        # subscribe under the chosen namespace
        self.odom_subscriber = self.create_subscription(
            Odometry,
            f'/{ns}/odometry',
            self.odom_callback,
            10)
        # publish combined pose and twist
        self.odom_publisher = self.create_publisher(
            Odometry,
            f'/{ns}/odom_map',
            10)
        # Timer to calculate and publish pose periodically
        self.timer = self.create_timer(1.0, self.calculate_and_publish_pose)

    def odom_callback(self, msg: Odometry):
        """Stores the latest odometry message."""
        self.latest_odom = msg

    def calculate_and_publish_pose(self):
        """Calculates and logs the base_footprint pose and twist in the map frame."""
        if not self.latest_odom:
            self.get_logger().info("Waiting for odometry message...")
            return

        # use the frame_id from the odom header (e.g. "turtlebot1/odom")
        from_frame = self.latest_odom.header.frame_id
        to_frame = 'map'

        try:
            # 1. Get the transform from map → odom
            now = rclpy.time.Time()
            # self.get_logger().info(f"Looking up transform from {to_frame} to {from_frame}...")
            transform = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            # self.get_logger().info(f"Transform found: {transform}")

            # 2. Create a PoseStamped for base_footprint in odom frame
            pose_in_odom = PoseStamped()
            pose_in_odom.header = self.latest_odom.header
            pose_in_odom.header.frame_id = from_frame
            pose_in_odom.pose = self.latest_odom.pose.pose

            # 3. Transform the pose to the map frame
            pose_in_map = do_transform_pose_stamped(pose_in_odom, transform)

            # Create an Odometry message for pose and twist in the map frame
            odom_in_map = Odometry()
            odom_in_map.header = self.latest_odom.header
            odom_in_map.header.frame_id = to_frame

            # Set the transformed pose
            odom_in_map.pose.pose = pose_in_map.pose

            # Transform the twist (linear and angular velocities)
            linear = self.latest_odom.twist.twist.linear
            angular = self.latest_odom.twist.twist.angular
            odom_in_map.twist.twist.linear.x = (
                transform.transform.rotation.w * linear.x -
                transform.transform.rotation.z * linear.y
            )
            odom_in_map.twist.twist.linear.y = (
                transform.transform.rotation.z * linear.x +
                transform.transform.rotation.w * linear.y
            )
            odom_in_map.twist.twist.linear.z = linear.z

            # Angular velocity remains the same in this case
            odom_in_map.twist.twist.angular = angular

            # Publish the combined Odometry message
            self.odom_publisher.publish(odom_in_map)

            x = pose_in_map.pose.position.x
            y = pose_in_map.pose.position.y
            yaw = self.quaternion_to_yaw(pose_in_map.pose.orientation)
            # self.get_logger().info(
            #     f"Base footprint in map: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
            # )

        except TransformException as ex:
            self.get_logger().warn(f"Could not transform {from_frame} to {to_frame}: {ex}")
        except Exception as e:
            self.get_logger().error(f"Error ({type(e).__name__}): {e}")

    def quaternion_to_yaw(self, q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny, cosy)

def main(args=None):
    rclpy.init(args=args)
    # Initialize the node first
    node = rclpy.create_node('namespace_resolver')
    namespace = node.declare_parameter('namespace', '').get_parameter_value().string_value
    node.destroy_node()  # Destroy the temporary node after retrieving the parameter

    # Pass the namespace to the TurtlebotPose node
    node = TurtlebotPose(namespace)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()