#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
import subprocess
import os


class CameraSetupNode(Node):
    def __init__(self):
        super().__init__('camera_setup_node')
        
        # Wait for Gazebo to fully start
        time.sleep(3.0)
        
        self.get_logger().info("Setting up top-down camera view...")
        
        # Try to set the camera pose using Ignition service
        try:
            # Reset camera to top-down view over maze
            # Maze center is approximately at (4, 2.5)
            cmd = [
                'ign', 'service', '-s', '/gui/camera/view_control',
                '--reqtype', 'ignition.msgs.CameraCmd',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '1000',
                '--req', 'pose: {position: {x: 4.0, y: 2.5, z: 20.0}, orientation: {x: 0.0, y: 0.7071068, z: 0.0, w: 0.7071068}}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.get_logger().info("Successfully set camera to top-down view")
            else:
                self.get_logger().warn(f"Failed to set camera pose: {result.stderr}")
                
        except Exception as e:
            self.get_logger().error(f"Exception setting camera pose: {e}")
        
        # Alternative method: try using the move_to service
        try:
            time.sleep(1.0)
            cmd2 = [
                'ign', 'service', '-s', '/gui/move_to/pose',
                '--reqtype', 'ignition.msgs.StringMsg',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '1000',
                '--req', 'data: "4 2.5 20 0 1.5708 0"'
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=5)
            if result2.returncode == 0:
                self.get_logger().info("Alternative camera setup successful")
            else:
                self.get_logger().warn("Alternative camera setup failed")
                
        except Exception as e:
            self.get_logger().warn(f"Alternative camera setup exception: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    node = CameraSetupNode()
    
    # Run for a short time then exit
    rclpy.spin_once(node, timeout_sec=1.0)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
