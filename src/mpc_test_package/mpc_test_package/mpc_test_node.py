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
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster
import tf_transformations as tf

from .mobile_robot_mpc import MobileRobotMPC


class MPCTestNode(Node):
    def __init__(self):
        super().__init__('mpc_test_node')
        
        self.get_logger().info('Starting MPC Test Node...')
        
        # Initialize MPC controller with error handling
        try:
            self.get_logger().info('Initializing MPC controller...')
            self.mpc = MobileRobotMPC()
            self.get_logger().info('MPC controller initialized successfully!')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MPC controller: {e}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            # Create a dummy MPC object to continue
            self.mpc = None
        
        # State variables
        self.current_state = None
        self.ref_trajectory = None
        self.trajectory_index = 0
        
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
        
        # Control timer - run MPC at 2Hz
        self.control_timer = self.create_timer(
            0.5,  # 0.5 seconds = 2Hz
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
            self.get_logger().info('Note: Robot should be positioned close to the trajectory start for best results')
            
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
            
            # Extract orientation (quaternion to euler)
            quat = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            euler = euler_from_quaternion(quat)
            yaw = euler[2]
            
            # Extract velocities
            linear_x = msg.twist.twist.linear.x
            angular_z = msg.twist.twist.angular.z
            
            # Update current state [x, y, theta, v, omega]
            self.current_state = np.array([
                position_x, position_y, yaw, linear_x, angular_z
            ])
            
            # Log first odometry received
            if not hasattr(self, '_first_odom_received'):
                self._first_odom_received = True
                self.get_logger().info(f'First Odometry received: x={position_x:.3f}, y={position_y:.3f}, θ={yaw:.3f}')
            
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
        """Get reference trajectory segment for MPC"""
        if not self.ref_trajectory:
            return np.zeros((5, self.mpc.N + 1))
        
        ref_array = np.zeros((5, self.mpc.N + 1))
        
        # Fill reference trajectory
        for i in range(self.mpc.N + 1):
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
            if self._control_loop_counter % 4 == 1:  # Log every 2 seconds
                self.get_logger().warn('No pose data received yet, skipping control loop')
            return
        
        if not self.ref_trajectory:
            self.get_logger().warn('No reference trajectory available')
            return
        
        # Debug: Log that we're entering the control computation
        if self._control_loop_counter % 4 == 1:
            self.get_logger().info(f'Control loop {self._control_loop_counter}: Computing MPC control...')
        
        try:
            # Get reference trajectory segment
            ref_array = self.get_reference_trajectory_segment()
            
            # Debug: Log reference trajectory info
            if self._control_loop_counter % 10 == 1:
                self.get_logger().info(f'Reference trajectory shape: {ref_array.shape}, first point: [{ref_array[0,0]:.3f}, {ref_array[1,0]:.3f}]')
            
            # Update MPC reference
            self.mpc.set_reference_trajectory(ref_array)
            
            # Debug: Log before MPC update
            if self._control_loop_counter % 4 == 1:
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
                if self._control_loop_counter % 4 == 1:
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
            
            # Log control command periodically
            if not hasattr(self, '_control_counter'):
                self._control_counter = 0
            self._control_counter += 1
            
            if self._control_counter % 2 == 1:  # Log every 1 second (2 * 0.5s)
                self.get_logger().info(
                    f'Control: v={cmd_msg.linear.x:.3f} m/s, ω={cmd_msg.angular.z:.3f} rad/s | '
                    f'State: x={self.current_state[0]:.3f}, y={self.current_state[1]:.3f}, '
                    f'θ={self.current_state[2]:.3f} | Traj idx: {self.trajectory_index}/{len(self.ref_trajectory)}'
                )
            
            # Update trajectory index based on progress
            self.update_trajectory_index()
            
            # Publish visualization
            self.publish_visualization()
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
            import traceback
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
            
            # Advance if we're close to current reference point
            if distance < 0.1:  # 10cm threshold
                self.trajectory_index = min(self.trajectory_index + 1, len(self.ref_trajectory) - 1)
                
        # Check if we've completed the trajectory
        if self.trajectory_index >= len(self.ref_trajectory) - 1:
            final_point = self.ref_trajectory[-1]
            final_distance = np.sqrt(
                (self.current_state[0] - final_point[0])**2 + 
                (self.current_state[1] - final_point[1])**2
            )
            
            if final_distance < 0.05:  # 5cm threshold
                self.get_logger().info('Trajectory completed successfully!')
    
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


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = MPCTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down MPC test node...')
    except Exception as e:
        print(f'Error in main: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
