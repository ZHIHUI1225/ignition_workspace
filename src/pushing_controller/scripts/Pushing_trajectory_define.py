#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from pushing_interfaces.action import ApplyPush
from pushing_interfaces.msg import PushingAction, PushingSample
import json
import os

class PushingTrajectoryClient(Node):
    def __init__(self):
        super().__init__('pushing_trajectory_client')
        
        # Get the node's namespace
        namespace = self.get_namespace()
        if namespace == '/':
            namespace = ''
            robot_id = 0
        else:
            # Extract robot ID from namespace (assuming format 'tb{i}')
            robot_id = int(namespace.strip('/').replace('tb', ''))
            
        self.get_logger().info(f"Initializing with namespace: {namespace}, robot_id: {robot_id}")
        
        # Create action client using the namespace
        self._action_client = ActionClient(self, ApplyPush, f'{namespace}/pushing/apply_push')
        
        # Set the trajectory file path based on the robot ID - use container path
        self.trajectory_file = f'/root/workspace/data/tb{robot_id}_DiscreteTrajectory.json'
        self.get_logger().info(f"Using trajectory file: {self.trajectory_file}")

    def send_goal(self):
        goal_msg = ApplyPush.Goal()
        goal_msg.action = PushingAction()

        # Load the trajectory file specific to this robot
        try:
            with open(self.trajectory_file, 'r') as f:
                data = json.load(f)
                trajectory_data = data["Trajectory"]
                self.get_logger().info(f"Loaded {len(trajectory_data)} trajectory points")
        except Exception as e:
            self.get_logger().error(f"Failed to load trajectory file: {e}")
            return

        # Define the trajectory
        for point in trajectory_data:
            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = 0.0
            pose.orientation.z = point[2]  # Assuming the orientation is given as yaw
            pose.orientation.w = 1.0  # Assuming no rotation around x and y axes

            twist = Twist()
            twist.linear.x = point[3]
            twist.angular.z = point[4]

            pushing_sample = PushingSample(pose=pose, twist=twist)
            goal_msg.action.trajectory.append(pushing_sample)

        # Wait for the server and send the goal
        self.get_logger().info("Waiting for action server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Action server found, sending goal...")
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
        self.get_logger().info(f'Result: {result}')

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback}')

def main(args=None):
    try:
        rclpy.init(args=args)
        client = PushingTrajectoryClient()
        client.send_goal()
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in Pushing_trajectory_define: {e}")
    finally:
        # Ensure proper cleanup
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()