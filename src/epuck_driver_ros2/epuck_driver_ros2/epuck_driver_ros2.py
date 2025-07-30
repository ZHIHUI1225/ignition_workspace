#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge.core import CvBridge

from ePuck import ePuck
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import math
from tf2_ros import TransformBroadcaster
import time

## Camera parameters
IMAGE_FORMAT = 'RGB_365'
CAMERA_ZOOM = 8

## Epuck dimensions
# Wheel Radio (cm)
WHEEL_DIAMETER = 4
# Separation between wheels (cm)
WHEEL_SEPARATION = 5.3

# Distance between wheels in meters (axis length); it's the same value as "WHEEL_SEPARATION" but expressed in meters.
WHEEL_DISTANCE = 0.053
# Wheel circumference (meters).
WHEEL_CIRCUMFERENCE = ((WHEEL_DIAMETER*math.pi)/100.0)
# Distance for each motor step (meters); a complete turn is 1000 steps.
MOT_STEP_DIST = (WHEEL_CIRCUMFERENCE/1000.0)    # 0.000125 meters per step (m/steps)

# available sensors
sensors = ['accelerometer', 'proximity', 'motor_position', 'light',
           'floor', 'camera', 'selector', 'motor_speed', 'microphone']


class EPuckDriverROS2(Node):
    """
    ROS2 e-puck driver
    """

    def __init__(self):
        super().__init__('epuck_driver_ros2')
        
        # Declare parameters
        self.declare_parameter('epuck_address', '')
        self.declare_parameter('epuck_name', 'epuck')
        self.declare_parameter('xpos', 0.0)
        self.declare_parameter('ypos', 0.0)
        self.declare_parameter('theta', 0.0)
        
        # Declare sensor parameters
        for sensor in sensors:
            self.declare_parameter(sensor, False)
        
        # Get parameters
        epuck_address = self.get_parameter('epuck_address').get_parameter_value().string_value
        self._name = self.get_parameter('epuck_name').get_parameter_value().string_value
        init_xpos = self.get_parameter('xpos').get_parameter_value().double_value
        init_ypos = self.get_parameter('ypos').get_parameter_value().double_value
        init_theta = self.get_parameter('theta').get_parameter_value().double_value
        
        if not epuck_address:
            self.get_logger().error('epuck_address parameter is required!')
            return
            
        self._bridge = ePuck(epuck_address, True)  # Enable debug for troubleshooting

        self.enabled_sensors = {}
        for sensor in sensors:
            self.enabled_sensors[sensor] = self.get_parameter(sensor).get_parameter_value().bool_value

        self.prox_publisher = []
        self.prox_msg = []

        self.theta = init_theta
        self.x_pos = init_xpos
        self.y_pos = init_ypos
        self.leftStepsPrev = 0
        self.rightStepsPrev = 0
        self.leftStepsDiff = 0
        self.rightStepsDiff = 0
        self.deltaSteps = 0
        self.deltaTheta = 0
        self.startTime = time.time()
        self.endTime = time.time()
        self.br = TransformBroadcaster(self)

    def greeting(self):
        """
        Hello by robot.
        """
        self._bridge.set_body_led(1)
        self._bridge.set_front_led(1)
        time.sleep(0.5)
        self._bridge.set_body_led(0)
        self._bridge.set_front_led(0)

    def disconnect(self):
        """
        Close bluetooth connection
        """
        self._bridge.close()

    def setup_sensors(self):
        """
        Disable all sensors - only motor control enabled
        """
        # Disable all sensors to reduce data transfer
        self._bridge.enable()  # Enable only basic communication

    def run(self):
        # Connect to the ePuck
        try:
            self._bridge.connect()
            self.get_logger().info(f'Connected to e-puck at {self.get_parameter("epuck_address").get_parameter_value().string_value}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to e-puck: {str(e)}')
            return

        # Setup the necessary sensors.
        self.setup_sensors()

        self.greeting()

        self._bridge.step()

        # Subscribe to Command Velocity Topic
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.handler_velocity,
            10)

        # No sensor publishers - only motor control

        # Create timer for motor commands only
        self.timer = self.create_timer(0.1, self.update_sensors)  # 10 Hz
        self.startTime = time.time()

    def update_sensors(self):
        """
        Update function - now only sends motor commands without reading sensors
        """
        try:
            # Only send motor commands, no sensor reading
            self._bridge.step()
            
            # No sensor data processing - removed all sensor publishing code
            
        except Exception as e:
            self.get_logger().error(f'Error in update cycle: {str(e)}')

    def publish_proximity_transforms(self):
        """Publish transforms for proximity sensors"""
        current_time = self.get_clock().now().to_msg()
        
        # e-puck proximity positions and orientations
        prox_poses = [
            (0.035, -0.010, 0.034, 6.11),  # P0
            (0.025, -0.025, 0.034, 5.59),  # P1
            (0.000, -0.030, 0.034, 4.71),  # P2
            (-0.035, -0.020, 0.034, 3.49), # P3
            (-0.035, 0.020, 0.034, 2.8),   # P4
            (0.000, 0.030, 0.034, 1.57),   # P5
            (0.025, 0.025, 0.034, 0.70),   # P6
            (0.035, 0.010, 0.034, 0.17)    # P7
        ]
        
        for i, (x, y, z, yaw) in enumerate(prox_poses):
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = f'{self._name}/base_link'
            t.child_frame_id = f'{self._name}/base_prox{i}'
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            
            # Convert yaw to quaternion
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = sy
            t.transform.rotation.w = cy
            
            self.br.sendTransform(t)

    def publish_odometry(self, motor_pos):
        """Publish odometry based on motor positions"""
        leftSteps = motor_pos[0]
        rightSteps = motor_pos[1]
        
        self.leftStepsDiff = leftSteps - self.leftStepsPrev
        self.rightStepsDiff = rightSteps - self.rightStepsPrev
        
        # Calculate distance moved by each wheel
        leftDist = self.leftStepsDiff * MOT_STEP_DIST
        rightDist = self.rightStepsDiff * MOT_STEP_DIST
        
        # Calculate robot movement
        deltaS = (leftDist + rightDist) / 2.0
        self.deltaTheta = (rightDist - leftDist) / WHEEL_DISTANCE
        
        # Update position
        deltaX = deltaS * math.cos(self.theta + self.deltaTheta/2.0)
        deltaY = deltaS * math.sin(self.theta + self.deltaTheta/2.0)
        
        self.x_pos += deltaX
        self.y_pos += deltaY
        self.theta += self.deltaTheta
        
        # Normalize theta
        while self.theta > math.pi:
            self.theta -= 2.0 * math.pi
        while self.theta < -math.pi:
            self.theta += 2.0 * math.pi
        
        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = f'{self._name}/base_link'
        
        # Position
        odom_msg.pose.pose.position.x = self.x_pos
        odom_msg.pose.pose.position.y = self.y_pos
        odom_msg.pose.pose.position.z = 0.0
        
        # Orientation
        cy = math.cos(self.theta * 0.5)
        sy = math.sin(self.theta * 0.5)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = sy
        odom_msg.pose.pose.orientation.w = cy
        
        # Velocity (simplified)
        self.endTime = time.time()
        dt = self.endTime - self.startTime
        if dt > 0:
            odom_msg.twist.twist.linear.x = deltaS / dt
            odom_msg.twist.twist.angular.z = self.deltaTheta / dt
        
        self.odom_publisher.publish(odom_msg)
        
        # Also publish transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = f'{self._name}/base_link'
        t.transform.translation.x = self.x_pos
        t.transform.translation.y = self.y_pos
        t.transform.translation.z = 0.0
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.br.sendTransform(t)
        
        # Update previous values
        self.leftStepsPrev = leftSteps
        self.rightStepsPrev = rightSteps
        self.startTime = self.endTime

    def handler_velocity(self, data):
        """
        Controls the velocity of each wheel based on linear and angular velocities.
        """
        linear = data.linear.x
        angular = data.angular.z

        # Kinematic model for differential robot.
        wl = (linear - (WHEEL_SEPARATION / 2.) * angular) / WHEEL_DIAMETER
        wr = (linear + (WHEEL_SEPARATION / 2.) * angular) / WHEEL_DIAMETER

        # At input 1000, angular velocity is 1 cycle / s or  2*pi/s.
        left_vel = wl * 1000.
        right_vel = wr * 1000.
        
        try:
            self._bridge.set_motors_speed(left_vel, right_vel)
        except Exception as e:
            self.get_logger().error(f'Error setting motor speed: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        epuck_driver = EPuckDriverROS2()
        epuck_driver.run()
        rclpy.spin(epuck_driver)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        if 'epuck_driver' in locals():
            epuck_driver.disconnect()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
