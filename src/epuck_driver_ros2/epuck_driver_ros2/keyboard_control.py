#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select
import threading
import subprocess
import os


class KeyboardControl(Node):
    """
    Keyboard control node to stop all robots with 's' key
    """

    def __init__(self):
        super().__init__('keyboard_control')
        
        # Publishers for all three robots
        self.robot0_pub = self.create_publisher(Twist, '/robot0/cmd_vel', 10)
        self.robot1_pub = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        self.robot2_pub = self.create_publisher(Twist, '/robot2/cmd_vel', 10)
        
        # Store original terminal settings
        self.settings = termios.tcgetattr(sys.stdin)
        
        # Flag to control continuous stop publishing
        self.stop_active = False
        
        # Timer for continuous stop commands
        self.stop_timer = self.create_timer(0.02, self.stop_timer_callback)  # 50Hz
        
        self.get_logger().info("Keyboard control active. Press 's' to stop all robots, 'r' to resume, 'q' to quit.")
        
        # Start keyboard monitoring in a separate thread
        self.keyboard_thread = threading.Thread(target=self.monitor_keyboard, daemon=True)
        self.keyboard_thread.start()

    def get_key(self):
        """Get a single key press from terminal"""
        tty.setraw(sys.stdin.fileno())
        key = ''
        if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
            key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def stop_movement_publishers(self):
        """Kill the ros2 topic pub processes that are publishing movement commands"""
        try:
            # Kill all ros2 topic pub processes for cmd_vel
            subprocess.run(['pkill', '-f', 'ros2 topic pub.*cmd_vel'], check=False)
            self.get_logger().info("🔪 Stopped movement command publishers")
        except Exception as e:
            self.get_logger().error(f"Error stopping publishers: {str(e)}")

    def stop_timer_callback(self):
        """Timer callback to continuously publish stop commands"""
        if self.stop_active:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.linear.y = 0.0
            stop_cmd.linear.z = 0.0
            stop_cmd.angular.x = 0.0
            stop_cmd.angular.y = 0.0
            stop_cmd.angular.z = 0.0
            
            # Continuously publish stop commands to override any other commands
            self.robot0_pub.publish(stop_cmd)
            self.robot1_pub.publish(stop_cmd)
            self.robot2_pub.publish(stop_cmd)

    def activate_emergency_stop(self):
        """Activate emergency stop - kill publishers and start continuous stop commands"""
        self.stop_movement_publishers()
        self.stop_active = True
        self.get_logger().info("🛑 EMERGENCY STOP ACTIVATED - All robots STOPPED!")

    def resume_robots(self):
        """Resume normal robot operation"""
        self.stop_active = False
        self.get_logger().info("✅ ROBOTS RESUMED - Stop commands disabled")

    def monitor_keyboard(self):
        """Monitor keyboard input in a separate thread"""
        try:
            while rclpy.ok():
                key = self.get_key()
                
                if key == 's' or key == 'S':
                    self.activate_emergency_stop()
                elif key == 'r' or key == 'R':
                    self.resume_robots()
                elif key == 'q' or key == 'Q':
                    self.get_logger().info("Quitting keyboard control...")
                    break
                elif key == '\x03':  # Ctrl+C
                    break
                    
        except Exception as e:
            self.get_logger().error(f"Keyboard monitoring error: {str(e)}")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def __del__(self):
        """Restore terminal settings on destruction"""
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        keyboard_control = KeyboardControl()
        rclpy.spin(keyboard_control)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        # Restore terminal settings
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, keyboard_control.settings)
        except:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
