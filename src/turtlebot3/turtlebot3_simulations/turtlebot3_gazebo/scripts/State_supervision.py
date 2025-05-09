#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import math
import threading
import re

class ProximityChecker(Node):
    def __init__(self,namespace):
        super().__init__('Pushing_checker')
        # Declare and get namespace parameter
        # self.declare_parameter('namespace', 'tb0')
        # namespace = self.get_parameter('namespace').value
        
        # Extract number from namespace using regex
        self.namespace_number = self.extract_namespace_number(namespace)
        self.get_logger().info(f'Operating in namespace: {namespace}, Extracted number: {self.namespace_number}')
        
        # Parameters
        self.declare_parameter('distance_threshold', 0.08)  # 10cm
        self.threshold = self.get_parameter('distance_threshold').value
        
        # Subscribers
        self.parcel_pose = None
        self.relay_pose = None
        self.lock = threading.Lock()
        
        self.parcel_sub = self.create_subscription(
            PoseStamped,
            f'/parcel{self.namespace_number}/pose',
            self.parcel_callback,
            10)
        nub_relay = self.namespace_number+1
        self.relay_sub = self.create_subscription(
            PoseStamped,
            f'/Relaypoint{nub_relay}/pose',
            self.relay_callback,
            10)
        
        # Publisher
        self.flag_pub = self.create_publisher(Bool, 'Pushing_flag', 10)
        
        # Timer for periodic checking
        self.timer = self.create_timer(0.1, self.check_proximity)  # 10Hz

    def extract_namespace_number(self, namespace):
        """Extract numerical index from namespace string using regex"""
        match = re.match(r"^tb(\d+)$", namespace)
        if match:
            return int(match.group(1))
        self.get_logger().error(f"Invalid namespace format: {namespace}. Using default 0")
        return 0
    
    def parcel_callback(self, msg):
        with self.lock:
            self.parcel_pose = msg.pose
            self.get_logger().debug('Updated parcel pose')

    def relay_callback(self, msg):
        with self.lock:
            self.relay_pose = msg.pose
            self.get_logger().debug('Updated relay point pose')

    def calculate_distance(self, pose1, pose2):
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx**2 + dy**2)

    def check_proximity(self):
        with self.lock:
            if self.parcel_pose is None or self.relay_pose is None:
                self.get_logger().warn('Waiting for both poses...')
                return
            distance = self.calculate_distance(self.parcel_pose, self.relay_pose)
            flag = Bool()
            flag.data = distance < self.threshold
            
            self.flag_pub.publish(flag)
            if flag.data:
                # self.get_logger().info(f'Flag: {flag.data}')
                self.get_logger().info(f'Distance: {distance:.2f}m, Flag: {flag.data}')

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('Publish_flag_node')
    node.declare_parameter('namespace', 'tb0')  # Declare the namespace parameter with a default value

    namespace = node.get_parameter('namespace').get_parameter_value().string_value 
    node = ProximityChecker(namespace)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()