#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Bool
from geometry_msgs.msg import PoseStamped
import threading

class ParcelManager(Node):
    def __init__(self):
        super().__init__('parcel_manager')
        
        # Current parcel index (starts with 0 for parcel0)
        self.current_parcel_index = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Publisher to notify all robots about the current parcel index
        self.parcel_index_pub = self.create_publisher(
            Int32,
            '/current_parcel_index',
            10
        )
        
        # Subscribe to the spawn_next_parcel topic (from ParcelSpawner.py)
        self.spawn_sub = self.create_subscription(
            String,
            '/spawn_next_parcel',
            self.spawn_next_callback,
            10
        )
        
        # Subscribe to the pushing_flag from each turtlebot
        for i in range(5):  # Assuming 5 turtlebots (0-4)
            namespace = f'tb{i}'
            self.create_subscription(
                Bool,
                f'/{namespace}/Pushing_flag',
                lambda msg, ns=namespace: self.pushing_flag_callback(msg, ns),
                10
            )
        
        # Keep track of which robots have completed pushing
        self.pushing_complete = {f'tb{i}': False for i in range(5)}
        
        # Publish initial parcel index
        self.publish_parcel_index()
        
        # Set up a timer to periodically publish the current parcel index
        self.timer = self.create_timer(1.0, self.publish_parcel_index)
        
        self.get_logger().info('Parcel Manager initialized')
    
    def spawn_next_callback(self, msg):
        """Handle request to spawn a new parcel"""
        if msg.data == 'spawn_next':
            with self.lock:
                self.current_parcel_index += 1
                self.get_logger().info(f'Incrementing parcel index to {self.current_parcel_index}')
                self.publish_parcel_index()
    
    def pushing_flag_callback(self, msg, namespace):
        """Handle pushing flag from turtlebots"""
        with self.lock:
            # Check if this is notification of pushing complete
            if msg.data:
                self.pushing_complete[namespace] = True
                self.get_logger().info(f'{namespace} has completed pushing parcel{self.current_parcel_index}')
            else:
                self.pushing_complete[namespace] = False
    
    def publish_parcel_index(self):
        """Publish the current parcel index for all robots to use"""
        msg = Int32()
        msg.data = self.current_parcel_index
        self.parcel_index_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    parcel_manager = ParcelManager()
    rclpy.spin(parcel_manager)
    parcel_manager.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()