#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import re

class CmdVelPublisher(Node):
    def __init__(self, namespace, data):
        super().__init__('cmd_vel_publisher')
        self.namespace = namespace
        self.data = data
        self.num=int( ''.join(filter(str.isdigit, namespace)))
        self.publisher_ = self.create_publisher(Twist, f'/turtlebot{self.num}/cmd_vel', 10)
        self.index = 0
        timer_period = 0.1# seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.index < len(self.data):
            msg = Twist()
            msg.linear.x = self.data[self.index][3]# cm to m
            msg.angular.z = self.data[self.index][4]
            # msg.linear.x = 0.0 # cm to m
            # msg.angular.z = 0.2
            self.publisher_.publish(msg)
            print(f'{self.num}..................')
            self.get_logger().info(f'Publishing to {self.num}/cmd_vel: "{msg}"')
            self.index += 1
        else:
            self.get_logger().info('Finished publishing all data.')
            self.timer.cancel()
    
def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('cmd_vel_publisher_node')
    node.declare_parameter('namespace', 'tb0')  # Declare the namespace parameter with a default value

    namespace = node.get_parameter('namespace').get_parameter_value().string_value  # Get the namespace parameter
    # Load the data from the JSON file
    json_file_path = f'/root/workspace/data/{namespace}_DiscreteTrajectory.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)['Trajectory']

    cmd_vel_publisher = CmdVelPublisher(namespace, data)
    rclpy.spin(cmd_vel_publisher)
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()