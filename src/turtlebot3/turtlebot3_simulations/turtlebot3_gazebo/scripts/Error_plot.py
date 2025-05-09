#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
import re
from rclpy.qos import QoSProfile, ReliabilityPolicy
import matplotlib.pyplot as plt
import csv

class ErrorPlot(Node):
    def __init__(self, namespace):
        # Unique node name with namespace
        super().__init__(f'error_plot_{namespace}')
        
        # Get turtlebot number from namespace
        match = re.search(r'\d+$', namespace)
        self.turtlebot_num = match.group() if match else '0'
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # Create topic names using namespace
        twist_topic = f'/{namespace}/turtlebot{self.turtlebot_num}/twist'
        cmd_vel_topic = f'/{namespace}/cmd_vel'

        # Subscribers with proper topic names
        self.twist_sub = self.create_subscription(
            TwistStamped, twist_topic, self.twist_callback, qos
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, cmd_vel_topic, self.cmd_vel_callback, qos
        )

        self.twist_data = []
        self.cmd_vel_data = []
        self.errors = []

    def twist_callback(self, msg):
        self.twist_data.append(msg)
        self.log_data()

    def cmd_vel_callback(self, msg):
        self.cmd_vel_data.append(msg)
        # self.log_data()

    def log_data(self):
        if len(self.twist_data) > 0 and len(self.cmd_vel_data) > 0:
            twist = self.twist_data[-1].twist
            cmd_vel = self.cmd_vel_data[-1]
            
            linear_error = abs(twist.linear.x - cmd_vel.linear.x)
            angular_error = abs(twist.angular.z - cmd_vel.angular.z)
            
            self.errors.append((linear_error, angular_error))
            # self.get_logger().info(
            #     f'Errors{self.turtlebot_num} - Linear: {linear_error:.4f}, Angular: {angular_error:.4f}',
            #     throttle_duration_sec=1.0  # Limit to 1 message/sec
            # )
    def plot_errors(self):
        linear_twist = [msg.twist.linear.x for msg in self.twist_data]
        angular_twist = [msg.twist.angular.z for msg in self.twist_data]
        linear_cmd_vel = [msg.linear.x for msg in self.cmd_vel_data]
        angular_cmd_vel = [msg.angular.z for msg in self.cmd_vel_data]
        linear_errors = [error[0] for error in self.errors]
        angular_errors = [error[1] for error in self.errors]

        plt.figure()
        
        plt.subplot(2, 1, 1)
        plt.plot(linear_twist, label=f'Linear Twist {self.turtlebot_num}')
        plt.plot(linear_cmd_vel, label=f'Linear Cmd_vel {self.turtlebot_num}')
        plt.plot(linear_errors, label=f'Linear Error {self.turtlebot_num}')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(angular_twist, label=f'Angular Twist {self.turtlebot_num}')
        plt.plot(angular_cmd_vel, label=f'Angular Cmd Vel {self.turtlebot_num}')  
        plt.plot(angular_errors, label=f'Angular Error {self.turtlebot_num}')
        plt.legend()

        plt.savefig(f'/root/workspace/turtlebot3_ws/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/error_plot{self.turtlebot_num}.png')
        plt.show()

    def save_errors_to_file(self):
        with open(f'/root/workspace/turtlebot3_ws/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/error_data{self.turtlebot_num}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Linear Error', 'Angular Error'])
            csvwriter.writerows(self.errors)

def main(args=None):
    rclpy.init(args=args)
    
    # Get namespace parameter
    temp_node = Node('temp_param_node')
    namespace = temp_node.declare_parameter('namespace', 'tb0').value
    temp_node.destroy_node()

        # Create and spin node
    error_node = ErrorPlot(namespace)
    try:
        rclpy.spin(error_node)
    except KeyboardInterrupt:
        pass
    finally:
        error_node.save_errors_to_file()
        error_node.plot_errors()
        error_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()