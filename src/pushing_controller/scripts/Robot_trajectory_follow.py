#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from pushing_interfaces.action import MoveRobotTo
from geometry_msgs.msg import Pose, Quaternion
import tf_transformations

class MoveRobotToClient(Node):

    def __init__(self):
        super().__init__('move_robot_to_client')
        # Get the fully qualified namespace for this node
        namespace = self.get_namespace()
        if namespace == '/':
            namespace = ''
        
        # Use the namespace to create the correct action server name
        self._action_client = ActionClient(self, MoveRobotTo, f'{namespace}/pushing/move_robot_to')
        self.get_logger().info(f'Connected to action server: {namespace}/pushing/move_robot_to')
        
        # Set default goal values
        self.declare_parameter('goal_x', 0.6)
        self.declare_parameter('goal_y', 1.2)
        self.declare_parameter('goal_theta', 0.8)

    def send_goal(self, x=None, y=None, theta=None):
        # Use parameters if no values are provided
        if x is None:
            x = self.get_parameter('goal_x').value
        if y is None:
            y = self.get_parameter('goal_y').value
        if theta is None:
            theta = self.get_parameter('goal_theta').value
            
        self.get_logger().info(f'Sending goal: x={x}, y={y}, theta={theta}')
        
        goal_msg = MoveRobotTo.Goal()
        goal_msg.target.position.x = x
        goal_msg.target.position.y = y
        goal_msg.target.position.z = 0.0
        quaternion = tf_transformations.quaternion_from_euler(0, 0, theta)
        goal_msg.target.orientation = Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3]
        )

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: Success: {result.success}, Distance: {result.distance}')
        # Don't shut down - keep the node running
        # rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.distance}')

def main(args=None):
    rclpy.init(args=args)
    client = MoveRobotToClient()
    client.send_goal()  # Use parameters by default
    rclpy.spin(client)

if __name__ == '__main__':
    main()