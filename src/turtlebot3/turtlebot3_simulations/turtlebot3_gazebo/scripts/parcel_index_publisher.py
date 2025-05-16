#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_srvs.srv import Trigger

class ParcelIndexPublisher(Node):
    def __init__(self):
        super().__init__('parcel_index_publisher')
        
        # Declare parameters
        self.declare_parameter('namespace', 'tb0')
        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
        
        # Initialize parcel index starting at 0
        self.current_parcel_index = 0
        
        # Create publisher for current parcel index
        self.parcel_index_pub = self.create_publisher(
            Int32, 
            f'/{self.namespace}/current_parcel_index', 
            10
        )
        
        # Create service server for picking_finished
        self.picking_finished_server = self.create_service(
            Trigger,
            f'/{self.namespace}/picking_finished',
            self.picking_finished_callback
        )
        
        # Timer for periodic publishing
        self.timer = self.create_timer(1.0, self.publish_parcel_index)
        
        self.get_logger().info(f'Parcel index publisher initialized for {self.namespace}')
        self.get_logger().info(f'Current parcel index: {self.current_parcel_index}')
        
        # Publish initial index immediately
        self.publish_parcel_index()
    
    def picking_finished_callback(self, request, response):
        """Service callback for when picking is finished"""
        # Increment parcel index
        self.current_parcel_index += 1
        self.get_logger().info(f'Pickup complete, incremented parcel index to {self.current_parcel_index}')
        
        # Publish updated index immediately
        self.publish_parcel_index()
        
        # Set response
        response.success = True
        response.message = f'Parcel index incremented to {self.current_parcel_index}'
        return response
    
    def publish_parcel_index(self):
        """Publish the current parcel index"""
        msg = Int32()
        msg.data = self.current_parcel_index
        self.parcel_index_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParcelIndexPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
