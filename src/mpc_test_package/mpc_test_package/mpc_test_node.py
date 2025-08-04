#!/usr/bin/env python3
"""
MPC Test Node
Tests MobileRobotMPC class by subscribing to /robot0/odom and publishing to /robot0/cmd_vel
Loads trajectory data from tb0_Trajectory.json
"""

import rclpy
from rclpy.node import Node
import numpy as np
import json
import os
import yaml
import traceback
import math
import matplotlib.pyplot as plt
import datetime
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster
import tf_transformations as tf

from .mobile_robot_mpc import MobileRobotMPC


class MPCTestNode(Node):
    def __init__(self):
        super().__init__('mpc_test_node')
        
        # Load configuration
        self.config = self.load_config()
        self.control_dt = self.config.get('planning', {}).get('discrete_dt', 0.1)
        
        self.get_logger().info(f'Starting MPC Test Node with control frequency {1.0/self.control_dt:.1f}Hz (dt={self.control_dt}s)...')
        
        # Initialize MPC controller with error handling
        try:
            self.get_logger().info('Initializing MPC controller...')
            self.mpc = MobileRobotMPC()
            self.get_logger().info('MPC controller initialized successfully!')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MPC controller: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            # Create a dummy MPC object to continue
            self.mpc = None
        
        # State variables
        self.current_state = None
        self.ref_trajectory = None
        self.trajectory_index = 0
        
        # Data recording variables
        self.recording_data = True
        self.robot_positions = []  # List to store [timestamp, x, y, theta, v, omega]
        self.target_positions = []  # List to store [timestamp, x_ref, y_ref, theta_ref, v_ref, omega_ref]
        self.control_commands = []  # List to store [timestamp, v_cmd, omega_cmd]
        self.start_time = None
        self.trajectory_completed = False
        
        self.get_logger().info('Data recording initialized - will track robot position, targets, and control commands')
        
        # Load trajectory data
        self.load_trajectory()
        
        # ROS2 Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, 
            '/robot0/cmd_vel', 
            10
        )
        
        # Visualization publishers
        self.reference_pub = self.create_publisher(
            Path, 
            '/robot0/reference_path', 
            10
        )
        
        self.predicted_pub = self.create_publisher(
            Path, 
            '/robot0/predicted_path', 
            10
        )
        
        # ROS2 Subscribers
        self.pose_sub = self.create_subscription(
            Odometry,
            '/robot0/odom',  # Subscribe to Odometry messages
            self.odom_callback,
            10
        )
        
        # Transform broadcaster for visualization
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Control timer - run MPC at configured frequency
        self.get_logger().info(f'Creating control timer with frequency {1.0/self.control_dt:.1f}Hz (dt={self.control_dt}s)')
        self.control_timer = self.create_timer(
            self.control_dt,  # Use configured dt to match trajectory generation frequency
            self.control_loop
        )
        
        # Debug timer to verify ROS2 spinning is working
        self.debug_timer = self.create_timer(
            1.0,  # 1 second
            self.debug_callback
        )
        
        # Path message objects for visualization
        self.ref_path = Path()
        self.ref_path.header.frame_id = 'world'
        self.pred_path = Path()
        self.pred_path.header.frame_id = 'world'
        
        self.get_logger().info('MPC Test Node initialized successfully!')
        
        # Add a timer for periodic data saving (every 30 seconds as backup)
        self.data_save_timer = self.create_timer(
            30.0,  # 30 seconds
            self.periodic_data_save
        )
        
    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle (rotation around Z-axis)"""
        # Extract quaternion components
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        
        # Convert to yaw angle for pure Z-axis rotation
        # For quaternions created with qz = sin(yaw/2), qw = cos(yaw/2), qx = 0, qy = 0
        # The inverse formula is: yaw = 2 * atan2(qz, qw)
        yaw = 2 * math.atan2(z, w)
        
        return yaw
        
    def periodic_data_save(self):
        """Periodically save data as backup (in case trajectory doesn't complete)"""
        if (self.recording_data and not self.trajectory_completed and 
            len(self.robot_positions) > 50):  # Only save if we have some data
            
            self.get_logger().info(f'Periodic data backup - recorded {len(self.robot_positions)} robot positions')
            # Could add backup save here if needed
        
    def load_trajectory(self):
        """Load reference trajectory from JSON file"""
        # Load the specific trajectory file from experi folder
        json_file_path = '/root/workspace/data/experi/tb0_Trajectory.json'
        
        trajectory_loaded = False
        
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                    self.ref_trajectory = data['Trajectory']
                    trajectory_loaded = True
                    self.get_logger().info(f'Loaded trajectory from: {json_file_path}')
                    self.get_logger().info(f'Trajectory length: {len(self.ref_trajectory)} points')
                    
                    # Log first few trajectory points for verification
                    if len(self.ref_trajectory) > 0:
                        first_point = self.ref_trajectory[0]
                        self.get_logger().info(f'First trajectory point: x={first_point[0]:.3f}, y={first_point[1]:.3f}, θ={first_point[2]:.3f}')
                        if len(self.ref_trajectory) > 1:
                            last_point = self.ref_trajectory[-1]
                            self.get_logger().info(f'Last trajectory point: x={last_point[0]:.3f}, y={last_point[1]:.3f}, θ={last_point[2]:.3f}')
                            
            except Exception as e:
                self.get_logger().error(f'Failed to load trajectory from {json_file_path}: {e}')
                trajectory_loaded = False
        else:
            self.get_logger().error(f'Trajectory file not found: {json_file_path}')
        
        if not trajectory_loaded:
            self.get_logger().error('Could not load the required trajectory file!')
            # Create a simple test trajectory as fallback
            self.create_test_trajectory()
        
        # Set reference trajectory in MPC
        if self.ref_trajectory and self.mpc is not None:
            ref_array = self.get_reference_trajectory_segment()
            self.mpc.set_reference_trajectory(ref_array)
            
            # Log initial trajectory information for debugging
            first_ref = self.ref_trajectory[0]
            self.get_logger().info(f'Setting reference trajectory. First point: x={first_ref[0]:.3f}, y={first_ref[1]:.3f}, θ={first_ref[2]:.3f}')
            self.get_logger().info('Note: Will find closest trajectory point when robot position is available')
    
    def find_closest_trajectory_point(self, current_position):
        """Find the closest point on the reference trajectory to current robot position"""
        if not self.ref_trajectory:
            return 0
        
        min_distance = float('inf')
        closest_index = 0
        
        current_x, current_y = current_position[0], current_position[1]
        
        for i, traj_point in enumerate(self.ref_trajectory):
            # Calculate Euclidean distance to trajectory point
            distance = np.sqrt((current_x - traj_point[0])**2 + (current_y - traj_point[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        self.get_logger().info(f'Found closest trajectory point: index {closest_index}/{len(self.ref_trajectory)}, distance: {min_distance:.3f}m')
        closest_point = self.ref_trajectory[closest_index]
        self.get_logger().info(f'Closest point: x={closest_point[0]:.3f}, y={closest_point[1]:.3f}, θ={closest_point[2]:.3f}')
        
        return closest_index
            
    def create_test_trajectory(self):
        """Create a simple test trajectory as fallback"""
        self.get_logger().info('Creating fallback test trajectory...')
        
        # Create a simple circular trajectory
        n_points = 50
        radius = 1.0
        center_x, center_y = 0.0, 0.0
        
        self.ref_trajectory = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            theta = angle + np.pi/2  # Tangent to circle
            v = 0.1  # 10 cm/s
            omega = 0.1  # Small angular velocity
            
            self.ref_trajectory.append([x, y, theta, v, omega])
            
        self.get_logger().info(f'Created test trajectory with {len(self.ref_trajectory)} points')
    
    def odom_callback(self, msg):
        """Update robot state from Odometry message"""
        try:
            # Extract position
            position_x = msg.pose.pose.position.x
            position_y = msg.pose.pose.position.y
            
            # Extract orientation using direct quaternion to yaw conversion
            yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
            
            # Extract velocities
            linear_x = msg.twist.twist.linear.x
            angular_z = msg.twist.twist.angular.z
            
            # Update current state [x, y, theta, v, omega]
            self.current_state = np.array([
                position_x, position_y, yaw, linear_x, angular_z
            ])
            
            # Record robot position data
            if self.recording_data and not self.trajectory_completed:
                if self.start_time is None:
                    self.start_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
                
                current_time = self.get_clock().now().nanoseconds / 1e9
                relative_time = current_time - self.start_time
                
                self.robot_positions.append([
                    relative_time, position_x, position_y, yaw, linear_x, angular_z
                ])
            
            # Log first odometry received and find closest trajectory point
            if not hasattr(self, '_first_odom_received'):
                self._first_odom_received = True
                self.get_logger().info(f'First Odometry received: x={position_x:.3f}, y={position_y:.3f}, θ={yaw:.3f}')
                
                # Find closest trajectory point and start from there
                if self.ref_trajectory:
                    self.trajectory_index = self.find_closest_trajectory_point([position_x, position_y])
                    self.get_logger().info(f'Starting trajectory following from index {self.trajectory_index}')
                    
                    # Update MPC with new reference starting from closest point
                    if self.mpc is not None:
                        ref_array = self.get_reference_trajectory_segment()
                        self.mpc.set_reference_trajectory(ref_array)
            
            # Publish transform for visualization
            self.publish_robot_transform(msg)
                
        except Exception as e:
            self.get_logger().error(f'Error in odom callback: {e}')
    
    def publish_robot_transform(self, odom_msg):
        """Publish transform from world to odom for visualization (connects camera_frame to robot frames)"""
        try:
            # The e-puck driver already publishes odom -> robot0/base_link
            # We need to publish world -> odom to complete the transform tree
            # This allows RViz to visualize everything in the world frame
            transform = TransformStamped()
            
            # Header
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'world'  # Parent frame (used in RViz)
            transform.child_frame_id = 'odom'    # Child frame (used by e-puck driver)
            
            # For now, assume world and odom are aligned (identity transform)
            # In a real system, this would come from SLAM or localization
            transform.transform.translation.x = 0.0
            transform.transform.translation.y = 0.0
            transform.transform.translation.z = 0.0
            
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0
            
            # Broadcast transform
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing robot transform: {e}')
    
    def debug_callback(self):
        """Debug callback to verify timer is working"""
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter <= 3:
            self.get_logger().info(f'Debug timer working #{self._debug_counter}')
        elif self._debug_counter % 5 == 0:
            self.get_logger().info(f'Debug timer #{self._debug_counter} - Node is alive')
    
    def get_reference_trajectory_segment(self):
        """Get reference trajectory segment for MPC with current horizon"""
        if not self.ref_trajectory:
            # Use maximum horizon for fallback
            horizon = getattr(self.mpc, 'N_max', 5) if self.mpc else 5
            return np.zeros((5, horizon + 1))
        
        # Get current horizon from MPC controller
        horizon = getattr(self.mpc, 'N', 5) if self.mpc else 5
        
        ref_array = np.zeros((5, horizon + 1))
        
        # Fill reference trajectory
        for i in range(horizon + 1):
            if self.trajectory_index + i < len(self.ref_trajectory):
                # Use normal trajectory point
                traj_idx = self.trajectory_index + i
                ref_point = self.ref_trajectory[traj_idx]
            else:
                # Use last point if we're at the end
                ref_point = self.ref_trajectory[-1]
            
            ref_array[0, i] = ref_point[0]  # x
            ref_array[1, i] = ref_point[1]  # y
            ref_array[2, i] = ref_point[2]  # theta
            ref_array[3, i] = ref_point[3]  # v
            ref_array[4, i] = ref_point[4]  # omega
        
        return ref_array
    
    def control_loop(self):
        """Main MPC control loop"""
        # Debug: Add entry logging
        if not hasattr(self, '_control_loop_counter'):
            self._control_loop_counter = 0
        self._control_loop_counter += 1
        
        # Always log first few calls to confirm timer is working
        if self._control_loop_counter <= 5:
            self.get_logger().info(f'Control loop called #{self._control_loop_counter}')
        elif self._control_loop_counter % 10 == 0:
            self.get_logger().info(f'Control loop called #{self._control_loop_counter}')
        
        # Check if we have valid pose data
        if self.current_state is None:
            # Calculate log interval for 2 second intervals
            log_interval_2s = max(1, int(2.0 / self.control_dt))
            if self._control_loop_counter % log_interval_2s == 1:
                self.get_logger().warn('No pose data received yet, skipping control loop')
            return
        
        if not self.ref_trajectory:
            self.get_logger().warn('No reference trajectory available')
            return
        
        # Debug: Log that we're entering the control computation
        log_interval_2s = max(1, int(2.0 / self.control_dt))
        if self._control_loop_counter % log_interval_2s == 1:
            self.get_logger().info(f'Control loop {self._control_loop_counter}: Computing MPC control...')
        
        try:
            # Check if robot is near the final target of trajectory
            final_target = self.ref_trajectory[-1]
            distance_to_final = np.sqrt(
                (self.current_state[0] - final_target[0])**2 + 
                (self.current_state[1] - final_target[1])**2
            )
            
            # If within 0.05m of final target, stop the robot
            if distance_to_final < 0.05:
                if not hasattr(self, '_final_target_reached'):
                    self._final_target_reached = True
                    self.get_logger().info(f'Robot reached final target! Distance: {distance_to_final:.3f}m < 0.05m. Stopping robot.')
                    
                    # Mark trajectory as completed and save data
                    if not self.trajectory_completed:
                        self.trajectory_completed = True
                        self.get_logger().info('Trajectory completed - robot at final target!')
                        self.save_and_plot_data()
                
                # Send zero control commands to stop the robot
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
                
                # Record the stop command
                if self.recording_data and not self.trajectory_completed and self.start_time is not None:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    relative_time = current_time - self.start_time
                    self.control_commands.append([relative_time, 0.0, 0.0])
                
                return  # Exit control loop early when stopped
            
            # Get reference trajectory segment
            ref_array = self.get_reference_trajectory_segment()
            
            # Debug: Log reference trajectory info
            log_interval_5s = max(1, int(5.0 / self.control_dt))
            if self._control_loop_counter % log_interval_5s == 1:
                self.get_logger().info(f'Reference trajectory shape: {ref_array.shape}, first point: [{ref_array[0,0]:.3f}, {ref_array[1,0]:.3f}]')
                self.get_logger().info(f'Distance to final target: {distance_to_final:.3f}m')
            
            # Update MPC reference
            self.mpc.set_reference_trajectory(ref_array)
            
            # Debug: Log before MPC update
            if self._control_loop_counter % log_interval_2s == 1:
                self.get_logger().info(f'Calling MPC update with state: [{self.current_state[0]:.3f}, {self.current_state[1]:.3f}, {self.current_state[2]:.3f}]')
                # Debug: Also log the current reference point
                current_ref = self.ref_trajectory[self.trajectory_index]
                self.get_logger().info(f'Current reference point: x={current_ref[0]:.3f}, y={current_ref[1]:.3f}, θ={current_ref[2]:.3f}')
                # Calculate distance to reference
                distance_to_ref = np.sqrt((self.current_state[0] - current_ref[0])**2 + (self.current_state[1] - current_ref[1])**2)
                angle_diff = abs(self.current_state[2] - current_ref[2])
                self.get_logger().info(f'Distance to reference: {distance_to_ref:.3f}m, angle difference: {angle_diff:.3f}rad')
            
            # Compute control command
            control_cmd = self.mpc.update(self.current_state)
            
            # Debug: Log MPC result with more detail
            if control_cmd is not None:
                if self._control_loop_counter % log_interval_2s == 1:
                    self.get_logger().info(f'MPC returned control: v={control_cmd[0]:.3f}, ω={control_cmd[1]:.3f}')
            else:
                self.get_logger().warn('MPC returned None control command')
            
            # Check if control command is valid
            if control_cmd is None or len(control_cmd) < 2:
                self.get_logger().warn('MPC returned invalid control command')
                control_cmd = [0.0, 0.0]  # Use zero command as fallback
            
            # Publish control command
            cmd_msg = Twist()
            cmd_msg.linear.x = float(control_cmd[0])
            cmd_msg.angular.z = float(control_cmd[1])
            
            self.cmd_vel_pub.publish(cmd_msg)
            
            # Record control command and target data
            if self.recording_data and not self.trajectory_completed and self.start_time is not None:
                current_time = self.get_clock().now().nanoseconds / 1e9
                relative_time = current_time - self.start_time
                
                # Record control command
                self.control_commands.append([
                    relative_time, float(control_cmd[0]), float(control_cmd[1])
                ])
                
                # Record current target (reference point)
                if self.trajectory_index < len(self.ref_trajectory):
                    current_ref = self.ref_trajectory[self.trajectory_index]
                    self.target_positions.append([
                        relative_time, current_ref[0], current_ref[1], current_ref[2], current_ref[3], current_ref[4]
                    ])
            
            # Log control command periodically
            if not hasattr(self, '_control_counter'):
                self._control_counter = 0
            self._control_counter += 1
            
            # Calculate log interval for 1 second intervals
            log_interval_1s = max(1, int(1.0 / self.control_dt))
            if self._control_counter % log_interval_1s == 1:
                self.get_logger().info(
                    f'Control: v={cmd_msg.linear.x:.3f} m/s, ω={cmd_msg.angular.z:.3f} rad/s | '
                    f'State: x={self.current_state[0]:.3f}, y={self.current_state[1]:.3f}, '
                    f'θ={self.current_state[2]:.3f} | Traj idx: {self.trajectory_index}/{len(self.ref_trajectory)}'
                )
            
            # Update trajectory index based on progress
            self.update_trajectory_index()
            
            # Update MPC reference trajectory as robot progresses
            self.update_mpc_reference()
            
            # Publish visualization
            self.publish_visualization()
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            # Publish zero command on error
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
    
    def update_trajectory_index(self):
        """Update trajectory index based on robot progress"""
        if not self.ref_trajectory or self.current_state is None:
            return
        
        # Calculate distance to current reference point
        if self.trajectory_index < len(self.ref_trajectory):
            ref_point = self.ref_trajectory[self.trajectory_index]
            distance = np.sqrt(
                (self.current_state[0] - ref_point[0])**2 + 
                (self.current_state[1] - ref_point[1])**2
            )
            
            # If robot is very far from current reference point, find closest point again
            if distance > 1.0:  # 1 meter threshold for re-searching
                self.get_logger().warn(f'Robot far from trajectory (distance: {distance:.3f}m), finding new closest point')
                self.trajectory_index = self.find_closest_trajectory_point(self.current_state[:2])
                return
            
            # Advance if we're close to current reference point
            if distance < 0.1:  # 10cm threshold
                self.trajectory_index = min(self.trajectory_index + 1, len(self.ref_trajectory) - 1)
                if self.trajectory_index % 10 == 0:  # Log progress every 10 points
                    self.get_logger().info(f'Advanced to trajectory point {self.trajectory_index}/{len(self.ref_trajectory)}')
                
        # Check if we've completed the trajectory
        if self.trajectory_index >= len(self.ref_trajectory) - 1:
            final_point = self.ref_trajectory[-1]
            final_distance = np.sqrt(
                (self.current_state[0] - final_point[0])**2 + 
                (self.current_state[1] - final_point[1])**2
            )
            
            if final_distance < 0.05:  # 5cm threshold
                if not self.trajectory_completed:
                    self.trajectory_completed = True
                    self.get_logger().info('Trajectory completed successfully!')
                    self.get_logger().info('Stopping data recording and generating plots...')
                    
                    # Save data and generate plots
                    self.save_and_plot_data()
    
    def update_mpc_reference(self):
        """Update MPC controller's reference trajectory as robot progresses"""
        if not self.ref_trajectory or self.current_state is None or self.mpc is None:
            return
        
        try:
            # Get updated reference trajectory segment starting from current trajectory index
            ref_array = self.get_reference_trajectory_segment()
            
            # Update MPC with new reference segment
            self.mpc.set_reference_trajectory(ref_array)
            
            # Log reference update periodically for debugging
            if not hasattr(self, '_ref_update_counter'):
                self._ref_update_counter = 0
            self._ref_update_counter += 1
            
            # Calculate log interval for 5 second intervals
            log_interval_5s = max(1, int(5.0 / self.control_dt))
            if self._ref_update_counter % log_interval_5s == 1:
                current_ref = self.ref_trajectory[self.trajectory_index]
                self.get_logger().info(f'Updated MPC reference - Current target: x={current_ref[0]:.3f}, y={current_ref[1]:.3f}, θ={current_ref[2]:.3f}')
                self.get_logger().info(f'Reference segment: {self.trajectory_index} to {min(self.trajectory_index + self.mpc.N, len(self.ref_trajectory)-1)}')
                
        except Exception as e:
            self.get_logger().error(f'Error updating MPC reference: {e}')
    
    def publish_visualization(self):
        """Publish reference and predicted trajectories for visualization"""
        try:
            # Publish reference trajectory
            self.publish_reference_trajectory()
            
            # Publish predicted trajectory
            self.publish_predicted_trajectory()
            
        except Exception as e:
            self.get_logger().error(f'Error publishing visualization: {e}')
    
    def publish_reference_trajectory(self):
        """Publish reference trajectory for visualization"""
        if not self.ref_trajectory:
            return
            
        self.ref_path.poses.clear()
        self.ref_path.header.stamp = self.get_clock().now().to_msg()
        
        # Publish a segment around current index
        start_idx = max(0, self.trajectory_index)
        end_idx = min(len(self.ref_trajectory), start_idx + self.mpc.N + 5)
        
        for i in range(start_idx, end_idx):
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'world'
            pose_msg.header.stamp = self.ref_path.header.stamp
            
            pose_msg.pose.position.x = self.ref_trajectory[i][0]
            pose_msg.pose.position.y = self.ref_trajectory[i][1]
            pose_msg.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            yaw = self.ref_trajectory[i][2]
            quat = quaternion_from_euler(0.0, 0.0, yaw)
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
            
            self.ref_path.poses.append(pose_msg)
        
        self.reference_pub.publish(self.ref_path)
    
    def publish_predicted_trajectory(self):
        """Publish predicted trajectory from MPC"""
        try:
            pred_traj = self.mpc.get_predicted_trajectory()
            if pred_traj is None:
                return
            
            self.pred_path.poses.clear()
            self.pred_path.header.stamp = self.get_clock().now().to_msg()
            
            if isinstance(pred_traj, np.ndarray) and not np.isnan(pred_traj).any():
                for i in range(min(self.mpc.N+1, pred_traj.shape[1])):
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = "world"
                    pose_msg.header.stamp = self.pred_path.header.stamp
                    
                    pose_msg.pose.position.x = pred_traj[0, i]
                    pose_msg.pose.position.y = pred_traj[1, i]
                    pose_msg.pose.position.z = 0.0
                    
                    # Convert yaw to quaternion
                    yaw = pred_traj[2, i]
                    if not np.isnan(yaw):
                        quat = quaternion_from_euler(0.0, 0.0, yaw)
                        pose_msg.pose.orientation.x = quat[0]
                        pose_msg.pose.orientation.y = quat[1]
                        pose_msg.pose.orientation.z = quat[2]
                        pose_msg.pose.orientation.w = quat[3]
                    else:
                        pose_msg.pose.orientation.w = 1.0
                    
                    self.pred_path.poses.append(pose_msg)
                
                if len(self.pred_path.poses) > 0:
                    self.predicted_pub.publish(self.pred_path)
            
        except Exception as e:
            pass

    def save_and_plot_data(self):
        """Save recorded data to files and generate plots"""
        try:
            # Create timestamp for unique filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = f'/root/workspace/data/mpc_tracking_{timestamp}'
            
            # Create directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            self.get_logger().info(f'Saving data to: {data_dir}')
            
            # Convert lists to numpy arrays for easier handling
            robot_data = np.array(self.robot_positions) if self.robot_positions else np.array([])
            target_data = np.array(self.target_positions) if self.target_positions else np.array([])
            control_data = np.array(self.control_commands) if self.control_commands else np.array([])
            
            # Save raw data to JSON files
            self.save_data_json(data_dir, robot_data, target_data, control_data)
            
            # Generate and save plots
            self.generate_plots(data_dir, robot_data, target_data, control_data)
            
            self.get_logger().info(f'Data analysis complete! Files saved to: {data_dir}')
            
        except Exception as e:
            self.get_logger().error(f'Error saving and plotting data: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
    
    def save_data_json(self, data_dir, robot_data, target_data, control_data):
        """Save data to JSON files"""
        try:
            # Save robot positions
            if robot_data.size > 0:
                robot_dict = {
                    'description': 'Robot position data: [time, x, y, theta, v, omega]',
                    'data': robot_data.tolist(),
                    'headers': ['time_s', 'x_m', 'y_m', 'theta_rad', 'v_ms', 'omega_rads']
                }
                with open(os.path.join(data_dir, 'robot_positions.json'), 'w') as f:
                    json.dump(robot_dict, f, indent=2)
            
            # Save target positions
            if target_data.size > 0:
                target_dict = {
                    'description': 'Target position data: [time, x_ref, y_ref, theta_ref, v_ref, omega_ref]',
                    'data': target_data.tolist(),
                    'headers': ['time_s', 'x_ref_m', 'y_ref_m', 'theta_ref_rad', 'v_ref_ms', 'omega_ref_rads']
                }
                with open(os.path.join(data_dir, 'target_positions.json'), 'w') as f:
                    json.dump(target_dict, f, indent=2)
            
            # Save control commands
            if control_data.size > 0:
                control_dict = {
                    'description': 'Control command data: [time, v_cmd, omega_cmd]',
                    'data': control_data.tolist(),
                    'headers': ['time_s', 'v_cmd_ms', 'omega_cmd_rads']
                }
                with open(os.path.join(data_dir, 'control_commands.json'), 'w') as f:
                    json.dump(control_dict, f, indent=2)
                    
            self.get_logger().info('Raw data saved to JSON files')
            
        except Exception as e:
            self.get_logger().error(f'Error saving JSON data: {e}')
    
    def generate_plots(self, data_dir, robot_data, target_data, control_data):
        """Generate and save analysis plots"""
        try:
            plt.style.use('default')  # Reset any previous style
            
            # Plot 1: Trajectory tracking (X-Y plot)
            plt.figure(figsize=(12, 8))
            
            if robot_data.size > 0 and target_data.size > 0:
                plt.subplot(2, 2, 1)
                plt.plot(robot_data[:, 1], robot_data[:, 2], 'b-', linewidth=2, label='Robot Actual Path')
                plt.plot(target_data[:, 1], target_data[:, 2], 'r--', linewidth=2, label='Reference Path')
                plt.plot(robot_data[0, 1], robot_data[0, 2], 'go', markersize=10, label='Start')
                plt.plot(robot_data[-1, 1], robot_data[-1, 2], 'ro', markersize=10, label='End')
                plt.xlabel('X Position (m)')
                plt.ylabel('Y Position (m)')
                plt.title('Trajectory Tracking Performance')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                
                # Plot 2: Position tracking over time
                plt.subplot(2, 2, 2)
                plt.plot(robot_data[:, 0], robot_data[:, 1], 'b-', label='Robot X')
                plt.plot(robot_data[:, 0], robot_data[:, 2], 'g-', label='Robot Y')
                plt.plot(target_data[:, 0], target_data[:, 1], 'r--', alpha=0.7, label='Target X')
                plt.plot(target_data[:, 0], target_data[:, 2], 'm--', alpha=0.7, label='Target Y')
                plt.xlabel('Time (s)')
                plt.ylabel('Position (m)')
                plt.title('Position vs Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot 3: Orientation tracking
                plt.subplot(2, 2, 3)
                plt.plot(robot_data[:, 0], np.degrees(robot_data[:, 3]), 'b-', label='Robot θ')
                plt.plot(target_data[:, 0], np.degrees(target_data[:, 3]), 'r--', alpha=0.7, label='Target θ')
                plt.xlabel('Time (s)')
                plt.ylabel('Orientation (degrees)')
                plt.title('Orientation Tracking')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot 4: Control commands
                if control_data.size > 0:
                    plt.subplot(2, 2, 4)
                    plt.plot(control_data[:, 0], control_data[:, 1], 'b-', label='Linear Velocity')
                    plt.plot(control_data[:, 0], control_data[:, 2], 'r-', label='Angular Velocity')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Control Commands')
                    plt.title('Control Commands vs Time')
                    plt.legend(['v (m/s)', 'ω (rad/s)'])
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, 'mpc_tracking_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate tracking error analysis
            if robot_data.size > 0 and target_data.size > 0:
                self.plot_tracking_errors(data_dir, robot_data, target_data)
            
            self.get_logger().info('Analysis plots generated and saved')
            
        except Exception as e:
            self.get_logger().error(f'Error generating plots: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
    
    def plot_tracking_errors(self, data_dir, robot_data, target_data):
        """Generate detailed tracking error analysis"""
        try:
            # Simple error analysis without scipy interpolation
            if len(target_data) < 2 or len(robot_data) < 2:
                return
            
            # Find closest target points for each robot timestamp (simple approach)
            x_error = []
            y_error = []
            theta_error = []
            robot_times = []
            
            for i, robot_point in enumerate(robot_data):
                robot_time = robot_point[0]
                robot_x, robot_y, robot_theta = robot_point[1], robot_point[2], robot_point[3]
                
                # Find closest target point in time
                time_diffs = np.abs(target_data[:, 0] - robot_time)
                closest_idx = np.argmin(time_diffs)
                
                if time_diffs[closest_idx] < 1.0:  # Only use if within 1 second
                    target_point = target_data[closest_idx]
                    target_x, target_y, target_theta = target_point[1], target_point[2], target_point[3]
                    
                    x_error.append(robot_x - target_x)
                    y_error.append(robot_y - target_y)
                    
                    # Normalize theta error to [-pi, pi]
                    theta_err = robot_theta - target_theta
                    theta_err = np.arctan2(np.sin(theta_err), np.cos(theta_err))
                    theta_error.append(theta_err)
                    robot_times.append(robot_time)
            
            if len(x_error) == 0:
                self.get_logger().warn('No matching target data found for error analysis')
                return
                
            x_error = np.array(x_error)
            y_error = np.array(y_error)
            theta_error = np.array(theta_error)
            robot_times = np.array(robot_times)
            
            # Calculate distance error
            distance_error = np.sqrt(x_error**2 + y_error**2)
            
            # Create error plot
            plt.figure(figsize=(12, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(robot_times, x_error, 'r-', label='X Error', linewidth=2)
            plt.plot(robot_times, y_error, 'b-', label='Y Error', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Position Error (m)')
            plt.title('Position Tracking Errors')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 2)
            plt.plot(robot_times, np.degrees(theta_error), 'g-', label='Orientation Error', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Orientation Error (degrees)')
            plt.title('Orientation Tracking Error')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 3)
            plt.plot(robot_times, distance_error, 'm-', label='Distance Error', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Distance Error (m)')
            plt.title('Distance Tracking Error')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, 'tracking_errors.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate and save statistics
            stats = {
                'mean_x_error': float(np.mean(np.abs(x_error))),
                'std_x_error': float(np.std(x_error)),
                'max_x_error': float(np.max(np.abs(x_error))),
                'mean_y_error': float(np.mean(np.abs(y_error))),
                'std_y_error': float(np.std(y_error)),
                'max_y_error': float(np.max(np.abs(y_error))),
                'mean_distance_error': float(np.mean(distance_error)),
                'std_distance_error': float(np.std(distance_error)),
                'max_distance_error': float(np.max(distance_error)),
                'mean_theta_error': float(np.mean(np.abs(theta_error))),
                'std_theta_error': float(np.std(theta_error)),
                'max_theta_error': float(np.max(np.abs(theta_error))),
                'total_trajectory_time': float(robot_times[-1] - robot_times[0]) if len(robot_times) > 1 else 0.0,
                'num_data_points': len(robot_times)
            }
            
            with open(os.path.join(data_dir, 'tracking_statistics.json'), 'w') as f:
                json.dump(stats, f, indent=2)
                
            self.get_logger().info(f'Tracking Statistics:')
            self.get_logger().info(f'  Mean distance error: {stats["mean_distance_error"]:.4f} m')
            self.get_logger().info(f'  Max distance error: {stats["max_distance_error"]:.4f} m')
            self.get_logger().info(f'  Mean orientation error: {np.degrees(stats["mean_theta_error"]):.2f} degrees')
            self.get_logger().info(f'  Total trajectory time: {stats["total_trajectory_time"]:.2f} s')
            self.get_logger().info(f'  Data points analyzed: {stats["num_data_points"]}')
            
        except Exception as e:
            self.get_logger().error(f'Error in tracking error analysis: {e}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

    def load_config(self):
        """Load configuration from YAML file"""
        config_path = '/root/workspace/config/config.yaml'
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            self.get_logger().error(f"Error loading config from {config_path}: {e}")
            self.get_logger().warn("Using default configuration")
            return {
                'planning': {
                    'discrete_dt': 0.1  # Default fallback
                }
            }


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    node = None
    try:
        node = MPCTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down MPC test node...')
        if node and node.recording_data and len(node.robot_positions) > 10:
            print('Saving recorded data before shutdown...')
            node.save_and_plot_data()
    except Exception as e:
        print(f'Error in main: {e}')
        if node and node.recording_data and len(node.robot_positions) > 10:
            print('Saving recorded data due to error...')
            try:
                node.save_and_plot_data()
            except:
                pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
