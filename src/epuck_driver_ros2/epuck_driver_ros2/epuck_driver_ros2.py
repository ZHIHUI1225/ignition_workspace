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
# Wheel Radio (cm) - CORRECTED from official e-puck2 documentation
WHEEL_DIAMETER = 4.1  # 41 mm according to official specs
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
        # Use the node's namespace as the robot name (removes leading '/' if present)
        self._name = self.get_namespace().lstrip('/')
        if not self._name:  # If no namespace, use epuck_name parameter as fallback
            self._name = self.get_parameter('epuck_name').get_parameter_value().string_value
        init_xpos = self.get_parameter('xpos').get_parameter_value().double_value
        init_ypos = self.get_parameter('ypos').get_parameter_value().double_value
        init_theta = self.get_parameter('theta').get_parameter_value().double_value
        
        if not epuck_address:
            self.get_logger().error('epuck_address parameter is required!')
            return
            
        self._bridge = ePuck(epuck_address, False)  # Disable debug to reduce console output

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
        
        # Disable odometry publisher to avoid conflict with camera-based localization
        # self.odom_publisher = self.create_publisher(Odometry, 'odom', 10)
        
        # Motor speed tracking for velocity feedback
        self.commanded_left_speed = 0.0
        self.commanded_right_speed = 0.0
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

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
        Enable minimal sensors for motor control and velocity feedback
        """
        # Enable basic communication
        self._bridge.enable()
        
        # Enable motor position sensing for odometry/velocity feedback
        # Use the correct method for enabling motor position sensors
        try:
            if hasattr(self._bridge, 'enable_motor_position'):
                self._bridge.enable_motor_position()
                self.get_logger().info("Motor position sensors enabled via enable_motor_position()")
            elif hasattr(self._bridge, 'set_motors_position_sensor'):
                self._bridge.set_motors_position_sensor(True)
                self.get_logger().info("Motor position sensors enabled via set_motors_position_sensor()")
            elif hasattr(self._bridge, 'enable_position'):
                self._bridge.enable_position()
                self.get_logger().info("Motor position sensors enabled via enable_position()")
            else:
                # Only log this once
                self.get_logger().info("Motor position sensing: Will attempt to read positions without explicit enabling")
        except Exception as e:
            self.get_logger().warn(f"Could not enable motor position sensors: {str(e)}")
            self.get_logger().info("Will attempt to read motor positions anyway")

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
            'cmd_vel',  # Use relative topic name - namespace will be added by launch file
            self.handler_velocity,
            10)

        # No sensor publishers - only motor control

        # Create timer for motor commands only
        self.timer = self.create_timer(0.1, self.update_sensors)  # 10 Hz
        self.startTime = time.time()

    def update_sensors(self):
        """
        Update function - sends motor commands and reads motor positions for velocity feedback
        """
        try:
            # Send motor commands and read sensor data
            self._bridge.step()
            
            # Read motor positions for odometry and velocity feedback
            try:
                motor_pos = None
                # Try different methods to get motor position
                if hasattr(self._bridge, 'get_motor_position_sensor'):
                    motor_pos = self._bridge.get_motor_position_sensor()
                elif hasattr(self._bridge, 'get_motors_position'):
                    motor_pos = self._bridge.get_motors_position()
                elif hasattr(self._bridge, 'get_position'):
                    motor_pos = self._bridge.get_position()
                elif hasattr(self._bridge, 'motor_position'):
                    motor_pos = self._bridge.motor_position
                else:
                    # List available methods for debugging
                    if not hasattr(self, '_methods_logged'):
                        methods = [method for method in dir(self._bridge) if not method.startswith('_')]
                        self.get_logger().info(f"Available ePuck methods: {sorted(methods)}")
                        self._methods_logged = True
                
                if motor_pos is not None and len(motor_pos) >= 2:
                    # Update internal odometry tracking but don't publish (camera provides localization)
                    self.update_internal_odometry(motor_pos)
                else:
                    # If no motor position available, update internal tracking without encoder feedback
                    if not hasattr(self, '_no_encoder_logged'):
                        self.get_logger().info("No motor position data available - using commanded velocities for internal tracking")
                        self._no_encoder_logged = True
                    # Still update internal odometry based on commanded velocities
                    self.update_internal_odometry_without_encoders()
                    
            except Exception as e:
                # Only log motor position errors occasionally to avoid spam
                if not hasattr(self, '_motor_error_count'):
                    self._motor_error_count = 0
                self._motor_error_count += 1
                if self._motor_error_count % 50 == 1:  # Log every 50th error (once per 5 seconds)
                    self.get_logger().debug(f'Could not read motor positions: {str(e)}')
            
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

    def update_internal_odometry(self, motor_pos):
        """Update internal odometry tracking without publishing (camera provides localization)"""
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
        
        # Calculate actual velocities
        self.endTime = time.time()
        dt = self.endTime - self.startTime
        
        if dt > 0.001:  # Avoid division by very small numbers
            self.current_linear_vel = deltaS / dt
            self.current_angular_vel = self.deltaTheta / dt
        else:
            self.current_linear_vel = 0.0
            self.current_angular_vel = 0.0
        
        # Don't publish odometry - camera node provides localization
        # Just update internal tracking for velocity feedback
        
        # Periodic velocity feedback logging (every 100 updates ≈ 10 seconds at 10Hz)
        if hasattr(self, '_odom_log_count'):
            self._odom_log_count += 1
        else:
            self._odom_log_count = 0
            
        if self._odom_log_count % 100 == 0:
            # Calculate commanded velocities for comparison
            cmd_linear = (self.commanded_left_speed + self.commanded_right_speed) * MOT_STEP_DIST / 2.0 / dt if dt > 0 else 0
            cmd_angular = (self.commanded_right_speed - self.commanded_left_speed) * MOT_STEP_DIST / WHEEL_DISTANCE / dt if dt > 0 else 0
            
            self.get_logger().info(f'=== Velocity Feedback (E-puck Internal) ===')
            self.get_logger().info(f'Commanded: linear={cmd_linear:.3f} m/s, angular={cmd_angular:.3f} rad/s')
            self.get_logger().info(f'Actual:    linear={self.current_linear_vel:.3f} m/s, angular={self.current_angular_vel:.3f} rad/s')
            self.get_logger().info(f'Internal Position: x={self.x_pos:.3f} m, y={self.y_pos:.3f} m, theta={self.theta:.3f} rad')
            self.get_logger().info(f'Note: Camera node provides published localization, this is internal tracking only')
        
        # Update previous values
        self.leftStepsPrev = leftSteps
        self.rightStepsPrev = rightSteps
        self.startTime = self.endTime

    def update_internal_odometry_without_encoders(self):
        """Update internal odometry tracking without publishing when encoders are not available"""
        # Calculate time difference
        self.endTime = time.time()
        dt = self.endTime - self.startTime
        
        if dt > 0.001:  # Avoid division by very small numbers
            # Use commanded velocities to estimate movement
            # Convert commanded motor speeds back to linear/angular velocities
            left_speed_ms = self.commanded_left_speed * MOT_STEP_DIST
            right_speed_ms = self.commanded_right_speed * MOT_STEP_DIST
            
            # Calculate estimated robot movement based on commanded speeds
            estimated_linear_vel = (left_speed_ms + right_speed_ms) / 2.0
            estimated_angular_vel = (right_speed_ms - left_speed_ms) / WHEEL_DISTANCE
            
            # Update position based on estimated movement
            deltaS = estimated_linear_vel * dt
            deltaTheta = estimated_angular_vel * dt
            
            deltaX = deltaS * math.cos(self.theta + deltaTheta/2.0)
            deltaY = deltaS * math.sin(self.theta + deltaTheta/2.0)
            
            self.x_pos += deltaX
            self.y_pos += deltaY
            self.theta += deltaTheta
            
            # Normalize theta
            while self.theta > math.pi:
                self.theta -= 2.0 * math.pi
            while self.theta < -math.pi:
                self.theta += 2.0 * math.pi
            
            # Store estimated velocities
            self.current_linear_vel = estimated_linear_vel
            self.current_angular_vel = estimated_angular_vel
        else:
            self.current_linear_vel = 0.0
            self.current_angular_vel = 0.0
        
        # Don't publish odometry - camera node provides localization
        # This is just internal tracking for velocity feedback
        
        # Periodic velocity feedback logging (every 100 updates ≈ 10 seconds at 10Hz)
        if hasattr(self, '_odom_fallback_log_count'):
            self._odom_fallback_log_count += 1
        else:
            self._odom_fallback_log_count = 0
            
        if self._odom_fallback_log_count % 100 == 0:
            # self.get_logger().info(f'=== Velocity Feedback (E-puck Internal, Estimated) ===')
            self.get_logger().info(f'Estimated: linear={self.current_linear_vel:.3f} m/s, angular={self.current_angular_vel:.3f} rad/s')
            # self.get_logger().info(f'Internal Position: x={self.x_pos:.3f} m, y={self.y_pos:.3f} m, theta={self.theta:.3f} rad')
        self.startTime = self.endTime

    def handler_velocity(self, data):
        """
        Controls the velocity of each wheel based on linear and angular velocities.
        Optimized for efficiency: reduced logging frequency, minimized attribute checks, and reduced time calls.
        """
        now = time.time()
        # Use a static attribute for last time, avoid hasattr check every call
        try:
            last_time = self._last_cmd_vel_time
        except AttributeError:
            last_time = now
        real_dt = now - last_time
        self._last_cmd_vel_time = now
        # Only log dt every 10th message to reduce log spam
        try:
            self._cmd_vel_log_count += 1
        except AttributeError:
            self._cmd_vel_log_count = 1
        if self._cmd_vel_log_count % 10 == 0:
            self.get_logger().info(f"[cmd_vel] Real dt: {real_dt:.4f} s")

        linear = data.linear.x
        angular = data.angular.z

        # Convert physical constants to meters for kinematic calculations
        wheel_diameter_m = WHEEL_DIAMETER / 100.0  # Convert cm to m (0.041 m)
        wheel_separation_m = WHEEL_SEPARATION / 100.0  # Convert cm to m (0.053 m)

        # Kinematic model for differential robot.
        wl = (linear - (wheel_separation_m / 2.) * angular) / (wheel_diameter_m / 2.)
        wr = (linear + (wheel_separation_m / 2.) * angular) / (wheel_diameter_m / 2.)

        # Convert wheel angular velocity (rad/s) to motor steps/s
        left_vel = wl * 1000. / (2 * math.pi)
        right_vel = wr * 1000. / (2 * math.pi)

        # Store original calculated speeds for debugging
        left_vel_orig = left_vel
        right_vel_orig = right_vel

        # Apply e-puck2 speed limits (max 1200 steps/s according to specs)
        MAX_STEPS_PER_SEC = 1200
        left_vel = max(-MAX_STEPS_PER_SEC, min(MAX_STEPS_PER_SEC, left_vel))
        right_vel = max(-MAX_STEPS_PER_SEC, min(MAX_STEPS_PER_SEC, right_vel))

        # Check if speeds were clamped
        left_clamped = (abs(left_vel_orig) > MAX_STEPS_PER_SEC)
        right_clamped = (abs(right_vel_orig) > MAX_STEPS_PER_SEC)

        # Store commanded speeds for velocity feedback
        self.commanded_left_speed = left_vel
        self.commanded_right_speed = right_vel

        # Set motor speeds, catch errors only if they occur
        try:
            self._bridge.set_motors_speed(left_vel, right_vel)
        except Exception as e:
            self.get_logger().error(f'Error setting motor speed: {str(e)}')

        # Log velocity commands every 100 cycles (10s at 10Hz)
        try:
            self._velocity_log_count += 1
        except AttributeError:
            self._velocity_log_count = 1
        if self._velocity_log_count % 100 == 0:
            self.get_logger().info(f'Commanded: linear={linear:.3f} m/s, angular={angular:.3f} rad/s')
            self.get_logger().info(f'Calculated speeds: left={left_vel_orig:.0f}, right={right_vel_orig:.0f} steps/s')
            self.get_logger().info(f'Clamped speeds: left={left_vel:.0f}, right={right_vel:.0f} steps/s (max: {MAX_STEPS_PER_SEC})')
            if left_clamped or right_clamped:
                self.get_logger().warn(f'Motor speeds were clamped! Original too high for hardware.')
                max_linear = MAX_STEPS_PER_SEC * (wheel_diameter_m / 2.) * (2 * math.pi) / 1000.
                max_angular = 2 * MAX_STEPS_PER_SEC * (wheel_diameter_m / 2.) * (2 * math.pi) / (1000. * wheel_separation_m)
                self.get_logger().warn(f'Max achievable: linear={max_linear:.3f} m/s, angular={max_angular:.3f} rad/s')


    def update_sensors(self):
        """
        Efficient update: sends motor commands and reads motor positions for velocity feedback.
        Reduced exception handling overhead and attribute checks.
        """
        # Send motor commands and read sensor data
        try:
            self._bridge.step()
        except Exception as e:
            self.get_logger().error(f'Error in _bridge.step(): {str(e)}')
            return

        # Read motor positions for odometry and velocity feedback
        motor_pos = None
        # Try different methods to get motor position (no nested try/except)
        if hasattr(self._bridge, 'get_motor_position_sensor'):
            motor_pos = self._bridge.get_motor_position_sensor()
        elif hasattr(self._bridge, 'get_motors_position'):
            motor_pos = self._bridge.get_motors_position()
        elif hasattr(self._bridge, 'get_position'):
            motor_pos = self._bridge.get_position()
        elif hasattr(self._bridge, 'motor_position'):
            motor_pos = self._bridge.motor_position
        else:
            if not hasattr(self, '_methods_logged'):
                methods = [method for method in dir(self._bridge) if not method.startswith('_')]
                self.get_logger().info(f"Available ePuck methods: {sorted(methods)}")
                self._methods_logged = True

        if motor_pos is not None and len(motor_pos) >= 2:
            self.update_internal_odometry(motor_pos)
        else:
            if not hasattr(self, '_no_encoder_logged'):
                self.get_logger().info("No motor position data available - using commanded velocities for internal tracking")
                self._no_encoder_logged = True
            self.update_internal_odometry_without_encoders()


def main(args=None):
    rclpy.init(args=args)
    try:
        epuck_driver = EPuckDriverROS2()
        epuck_driver.run()
        rclpy.spin(epuck_driver)
    except KeyboardInterrupt:
        pass
    finally:
        if 'epuck_driver' in locals():
            epuck_driver.disconnect()
        rclpy.shutdown()

if __name__ == "__main__":
    main()