#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Pose, TwistStamped
from nav_msgs.msg import Path, Odometry
import re
import casadi as ca
from std_msgs.msg import Header
import tf_transformations as tf
# We don't need to import quaternions module separately
import threading
from std_msgs.msg import Bool, String, Int32
from std_srvs.srv import Trigger


class PickupController(Node):
    def __init__(self):
        super().__init__('pickup_controller')
        
        # Declare parameters
        self.declare_parameter('namespace', 'tb0')
        self.declare_parameter('case', 'simple_maze')  # Add case parameter
        self.declare_parameter('num_robots', 3)        # Add N parameter (num_robots)
        
        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
        self.case = self.get_parameter('case').get_parameter_value().string_value
        self.num_robots = self.get_parameter('num_robots').get_parameter_value().integer_value
        
        # Initialize flags
        self.pickup_complete_flag = Bool()  # Flag to indicate pickup operation completed
        self.pickup_complete_flag.data = False
        self.pickup_ready_flag = False  # Flag to indicate robot is ready for pickup
        self.target_reached = False
        self.picking_finished_service_called_for_current_parcel = False # New flag
        
        # Current parcel index (defaults to 0 for parcel0)
        self.current_parcel_index = 0
        self.parcel_spawn_requested = False # Flag to track if spawn request has been made for current index
        
        # Threading lock for state protection
        self.state_lock = threading.Lock()
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'pickup/cmd_vel', 10)
        self.predicted_path_pub = self.create_publisher(Path, 'pickup/pred_path', 10)
        self.target_path_pub = self.create_publisher(Path, 'pickup/target_path', 10)
        
        # Service client for notifying picking finished
        self.picking_finished_client = self.create_client(Trigger, f'/{self.namespace}/picking_finished')
        
        # Service client for spawning next parcel
        self.spawn_parcel_client = self.create_client(Trigger, '/spawn_next_parcel_service')
        # Wait for the service to be available
        if not self.spawn_parcel_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/spawn_next_parcel_service not available, will not be able to request new parcels.')

        # Subscribe to current parcel index updates from State_switch
        self.parcel_index_sub = self.create_subscription(
            Int32,
            f'/{self.namespace}/current_parcel_index',
            self.parcel_index_callback,
            10
        )
        
        # Subscribers
        self.robot_pose_sub = self.create_subscription(
            Odometry, 'pickup/robot_pose', self.robot_pose_callback, 10)
            
        # Service server for starting picking
        self.start_picking_srv = self.create_service(
            Trigger, f'/{self.namespace}/start_picking', self.start_picking_callback)
        
        
        # Track workflow states
        self.pickup_active = False
        self.pickup_complete = False
        self.waiting_for_next = False
        
        json_file_path = f'/root/workspace/data/{self.case}/{self.namespace}_Trajectory.json'

        with open(json_file_path, 'r') as json_file:
            self.data = json.load(json_file)['Trajectory']
            self.trajectory_data=self.data[::-1]
        self.goal=self.data[0]
        if self.namespace == 'tb0':
            waypoint_file_path = f'/root/workspace/data/{self.case}/Waypoints.json'
            with open(waypoint_file_path, 'r') as waypoint_file:
                data = json.load(waypoint_file)
                relay_points = data['RelayPoints']
            x_parcel = relay_points[0]['Position'][0] / 100
            y_parcel = relay_points[0]['Position'][1] / 100
            self.relaypoint0_pose = [x_parcel,y_parcel]

        # Initialize state variables
        self.current_state = np.zeros(3)  # x, y, theta
        self.target_pose = np.zeros(3)    # x, y, theta
        self.target_pose[0] = self.goal[0]-0.4*np.cos(self.goal[2])
        self.target_pose[1] = self.goal[1]-0.4*np.sin(self.goal[2])
        self.target_pose[2] = self.goal[2]
        # Interpolate between target_pose and goal position
        # This creates a smoother approach to the final position
        num_interp_points = 5  # Number of interpolation points
        interp_points = []

        # Calculate the linear interpolation points
        for i in range(num_interp_points):
            alpha = i / (num_interp_points - 1)  # Interpolation factor (0 to 1)
            interp_x = self.target_pose[0] * alpha + self.goal[0] * (1 - alpha) 
            interp_y = self.target_pose[1] * alpha+ self.goal[1] * (1 - alpha) 
            interp_theta = self.target_pose[2]  # Keep orientation constant for smoother approach
            
            interp_points.append([interp_x, interp_y, interp_theta])
        print(f"Interpolated points: {interp_points}")
        # Add interpolation points to the beginning of trajectory_data
        # This means they'll be executed last in the reverse trajectory
        self.trajectory_data =  self.trajectory_data +interp_points 

        # Create MPC controller
        self.mpc = MobileRobotMPC()
        self.prediction_horizon = self.mpc.N
        
        # Initialize convergence tracking variables
        self.parallel_count = 0  # Count how many times robot is parallel to path
        self.last_cross_track_error = 0.0  # Track previous cross-track error
        
        # Paths for visualization
        self.target_path = Path()
        self.target_path.header.frame_id = 'world'
        self.predicted_path = Path()
        self.predicted_path.header.frame_id = 'world'
        
        # Control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f'Pickup controller initialized for {self.namespace}')
    
    def robot_pose_callback(self, msg):
        """Update position and orientation from pose message"""
        with self.state_lock:
            # Position
            self.current_state[0] = msg.pose.pose.position.x
            self.current_state[1] = msg.pose.pose.position.y
            
                       # Orientation (yaw)
            quat = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            euler = tf.euler_from_quaternion(quat)
            self.current_state[2] = euler[2] 
    
    def start_picking_callback(self, request, response):
        """Service callback to start the picking process"""
        old_flag = self.pickup_ready_flag
        self.pickup_ready_flag = True
        
        self.get_logger().info(f'Pickup service called, pickup_ready_flag: {old_flag} -> {self.pickup_ready_flag}')
        
        if self.waiting_for_next:
            self.get_logger().info('Received pickup service call while waiting for next task. Starting pickup task.')
            self.pickup_active = True
            self.pickup_complete = False
            response.success = True
            response.message = 'Pickup controller activated successfully'
        else:
            self.get_logger().info('Pickup service called, but controller is already active or not ready')
            response.success = True
            response.message = 'Pickup controller updated flag, but was not in waiting state'
            
        return response
    

    def parcel_index_callback(self, msg):
        """Handle updates to the current parcel index from State_switch"""
        new_index = msg.data
        if new_index != self.current_parcel_index:
            old_index = self.current_parcel_index
            self.current_parcel_index = new_index
            self.get_logger().info(f'Pickup controller updated parcel index: {old_index} -> {self.current_parcel_index}')
            # Reset state flags on new parcel
            self.pickup_ready_flag = False 
            self.pickup_active = False
            self.pickup_complete = False
            self.waiting_for_next =  True 
            self.parcel_spawn_requested = False # Reset spawn requested flag
            self.picking_finished_service_called_for_current_parcel = False # Reset new flag
    
    def call_picking_finished_service(self):
        """Call the picking_finished service to notify that picking is complete"""
        self.get_logger().info('Calling picking_finished service...')
        request = Trigger.Request()
        future = self.picking_finished_client.call_async(request)
        future.add_done_callback(self.picking_finished_callback)
        
    def picking_finished_callback(self, future):
        """Callback for the picking_finished service response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully notified picking finished: {response.message}')
                # self.picking_finished_service_called_for_current_parcel remains True
                # It will be reset by parcel_index_callback when the index changes.
            else:
                self.get_logger().warn(f'Failed to notify picking finished: {response.message}. Will retry.')
                # Reset flag to allow retry for the current parcel
                self.picking_finished_service_called_for_current_parcel = False
        except Exception as e:
            self.get_logger().error(f'Service call for picking_finished failed with exception: {str(e)}. Will retry.')
            # Reset flag to allow retry for the current parcel
            self.picking_finished_service_called_for_current_parcel = False
        
    def control_loop(self):
        """Main control loop running at fixed frequency"""
        # Only execute if we're in active pickup mode or if we're waiting for the next task
        if not self.pickup_ready_flag:
            return

        # When pickup is not active but pickup ready flag is true, activate pickup
        if not self.pickup_active and self.pickup_ready_flag and not self.waiting_for_next:
            self.get_logger().info('Starting pickup task')
            self.pickup_active = True
            
        if not self.pickup_active:
            # If we're not in pickup mode, there's nothing to do
            return

        # Calculate distance to target
        distance_error = np.sqrt((self.current_state[0] - self.target_pose[0])**2 + 
                                (self.current_state[1] - self.target_pose[1])**2)
        
        # Calculate orientation error
        angle_error = abs(self.current_state[2] - self.target_pose[2])
        angle_error = min(angle_error, 2*np.pi - angle_error)  # Normalize to [0, pi]
        
        # Define stopping criteria - both position and orientation must be satisfied
        position_tolerance = 0.05  # 5cm position tolerance
        orientation_tolerance = 0.1  # ~5.7 degrees orientation tolerance
        
        # Check if robot should continue moving or stop
        position_reached = distance_error <= position_tolerance
        orientation_reached = angle_error <= orientation_tolerance
        
        # Calculate approach phase scaling factors for smooth stopping
        # Define approach zones for velocity scaling
        approach_distance = 0.15  # Start slowing down at 15cm
        fine_approach_distance = 0.08  # Further slow down at 8cm
        
        # Position-based velocity scaling
        if distance_error <= position_tolerance:
            position_scale = 0.0  # Stop position movement
        elif distance_error <= fine_approach_distance:
            # Very fine approach - scale down to 10-20% speed
            position_scale = 0.1 + 0.1 * (distance_error - position_tolerance) / (fine_approach_distance - position_tolerance)
        elif distance_error <= approach_distance:
            # Approach phase - scale down to 20-60% speed
            position_scale = 0.2 + 0.4 * (distance_error - fine_approach_distance) / (approach_distance - fine_approach_distance)
        else:
            # Normal movement - full speed
            position_scale = 1.0
            
        # Orientation-based velocity scaling
        if angle_error <= orientation_tolerance:
            orientation_scale = 0.0  # Stop rotational movement
        elif angle_error <= 0.2:  # ~11.5 degrees
            # Fine orientation adjustment - scale down to 15-40% speed
            orientation_scale = 0.15 + 0.25 * (angle_error - orientation_tolerance) / (0.2 - orientation_tolerance)
        elif angle_error <= 0.4:  # ~23 degrees
            # Orientation approach - scale down to 40-70% speed
            orientation_scale = 0.4 + 0.3 * (angle_error - 0.2) / (0.4 - 0.2)
        else:
            # Large orientation error - full speed
            orientation_scale = 1.0
        
        if not position_reached and not self.pickup_complete:
            # Find the closest point in the trajectory to the current position
            min_dist = float('inf')
            closest_idx = 0
            
            for i, point in enumerate(self.trajectory_data):
                dist = np.sqrt((self.current_state[0] - point[0])**2 + 
                            (self.current_state[1] - point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Look ahead for better trajectory tracking
            lookahead_distance = 0.3  # meters
            current_s = closest_idx
            
            # Find a point ahead on the trajectory for better convergence
            for i in range(closest_idx, len(self.trajectory_data)):
                point = self.trajectory_data[i]
                dist_ahead = np.sqrt((self.current_state[0] - point[0])**2 + 
                                   (self.current_state[1] - point[1])**2)
                if dist_ahead >= lookahead_distance:
                    current_s = i
                    break
            
            # Generate reference trajectory from trajectory data
            # The reference trajectory should include the entire prediction horizon
            ref_array = np.zeros((3, self.prediction_horizon + 1))
            
            # Create a smooth reference trajectory with proper spacing
            for i in range(self.prediction_horizon + 1):
                # Calculate trajectory index with proper progression
                traj_idx = min(current_s + i, len(self.trajectory_data) - 1)
                
                # If we're near the end of trajectory, project the final point
                if traj_idx >= len(self.trajectory_data) - 1:
                    point = self.trajectory_data[-1]  # Use final point
                else:
                    point = self.trajectory_data[traj_idx]
                
                ref_array[0, i] = point[0]  # x
                ref_array[1, i] = point[1]  # y
                ref_array[2, i] = point[2]  # theta
            
            # Add cross-track error correction for the first reference point
            # This helps the robot converge to the path instead of running parallel
            cross_track_gain = 0.5  # Reduced gain to avoid infeasible reference adjustments
            if closest_idx < len(self.trajectory_data) - 1:
                # Calculate cross-track error
                closest_point = self.trajectory_data[closest_idx]
                next_point = self.trajectory_data[min(closest_idx + 1, len(self.trajectory_data) - 1)]
                
                # Path direction vector
                path_dx = next_point[0] - closest_point[0]
                path_dy = next_point[1] - closest_point[1]
                path_length = np.sqrt(path_dx**2 + path_dy**2)
                
                if path_length > 1e-6:  # Avoid division by zero
                    # Unit vector along path
                    path_unit_x = path_dx / path_length
                    path_unit_y = path_dy / path_length
                    
                    # Vector from closest point to robot
                    robot_dx = self.current_state[0] - closest_point[0]
                    robot_dy = self.current_state[1] - closest_point[1]
                    
                    # Cross-track error (perpendicular distance to path)
                    cross_track_error = robot_dx * (-path_unit_y) + robot_dy * path_unit_x
                    
                    # Check if robot is moving parallel to path (not converging)
                    if abs(cross_track_error - self.last_cross_track_error) < 0.01:  # More tolerant threshold
                        self.parallel_count += 1
                    else:
                        self.parallel_count = 0
                    
                    self.last_cross_track_error = cross_track_error
                    
                    # If robot has been parallel for too long, increase convergence slightly
                    if self.parallel_count > 8:  # Increased threshold
                        cross_track_gain = 1.0  # Only double the gain
                        print(f"Robot parallel to path for {self.parallel_count} steps, increasing convergence gain")
                    
                    # Limit the cross-track correction to avoid infeasible references
                    max_correction = 0.1  # Maximum 10cm correction
                    correction_x = -cross_track_gain * cross_track_error * (-path_unit_y)
                    correction_y = -cross_track_gain * cross_track_error * path_unit_x
                    
                    # Clamp corrections
                    correction_x = np.clip(correction_x, -max_correction, max_correction)
                    correction_y = np.clip(correction_y, -max_correction, max_correction)
                    
                    # Adjust only the first reference point (less aggressive)
                    ref_array[0, 0] = ref_array[0, 0] + correction_x
                    ref_array[1, 0] = ref_array[1, 0] + correction_y

            # Apply MPC to get control inputs
            with self.state_lock:
                current_state = self.current_state.copy()
            
            try:
                self.mpc.set_reference_trajectory(ref_array)
                u = self.mpc.update(current_state)
                
                # Apply velocity scaling for smooth approach to target
                raw_linear_vel = u[0, 0]
                raw_angular_vel = u[1, 0]
                
                # Scale velocities based on distance to target and orientation error
                scaled_linear_vel = raw_linear_vel * position_scale
                scaled_angular_vel = raw_angular_vel * orientation_scale
                
                # Apply minimum velocity constraints for very close approach
                # When both position and orientation are very close, reduce both velocities
                if distance_error <= fine_approach_distance and angle_error <= 0.2:
                    # Very close to target - limit both velocities more aggressively
                    max_approach_linear = 0.02  # 2cm/s max when very close
                    max_approach_angular = 0.1  # 0.1 rad/s max when very close
                    
                    scaled_linear_vel = np.clip(scaled_linear_vel, -max_approach_linear, max_approach_linear)
                    scaled_angular_vel = np.clip(scaled_angular_vel, -max_approach_angular, max_approach_angular)
                
                # Send control commands
                cmd_msg = Twist()
                cmd_msg.linear.x = scaled_linear_vel
                cmd_msg.angular.z = scaled_angular_vel
                self.cmd_vel_pub.publish(cmd_msg)
                
                # Debug output for velocity scaling
                if distance_error < approach_distance or angle_error < 0.4:
                    print(f'Approach mode: dist={distance_error:.3f}m, angle={angle_error:.3f}rad')
                    print(f'Scales: pos={position_scale:.2f}, orient={orientation_scale:.2f}')
                    print(f'Velocities: raw_v={raw_linear_vel:.3f}, scaled_v={scaled_linear_vel:.3f}')
                    print(f'            raw_w={raw_angular_vel:.3f}, scaled_w={scaled_angular_vel:.3f}')
                
                # Get predicted trajectory for error analysis
                predicted_traj = self.mpc.get_predicted_trajectory()
                
                # Calculate and output tracking errors
                self.calculate_and_output_errors(ref_array, predicted_traj, current_state)
                
                print(f'Published cmd_vel: linear.x={cmd_msg.linear.x}, angular.z={cmd_msg.angular.z}') 
                # Publish predicted trajectory for visualization
                self.predicted_path.poses = []
                
                for i in range(self.prediction_horizon + 1):
                    pose_msg = PoseStamped()
                    pose_msg.header = Header(frame_id="world")
                    pose_msg.pose.position.x = predicted_traj[0, i]
                    pose_msg.pose.position.y = predicted_traj[1, i]
                    pose_msg.pose.position.z = 0.0
                    
                    quat = tf.quaternion_from_euler(0.0, 0.0, predicted_traj[2, i])
                    pose_msg.pose.orientation.x = quat[1]
                    pose_msg.pose.orientation.y = quat[2]
                    pose_msg.pose.orientation.z = quat[3]
                    pose_msg.pose.orientation.w = quat[0]
                    
                    self.predicted_path.poses.append(pose_msg)
                
                self.predicted_path_pub.publish(self.predicted_path)
                
            except Exception as e:
                self.get_logger().error(f'MPC error: {str(e)}')
        else:                # Target reached
            if not self.pickup_complete:
                self.get_logger().info('Target reached, pickup complete.')

                if not self.picking_finished_service_called_for_current_parcel:
                    self.call_picking_finished_service() # Notify central index manager
                    self.picking_finished_service_called_for_current_parcel = True
                    self.pickup_complete = True
                    self.pickup_active = False  # Stop MPC/movement
                    self.waiting_for_next = True # Indicate it's ready for the next parcel cycle
                    self.get_logger().info('picking_finished service called and flagged for current parcel.')
                else:
                    self.get_logger().warn('picking_finished service already called for this parcel. Skipping.')

                # Request next parcel spawn if not already requested for this index
                if self.namespace == 'tb0' and not self.parcel_spawn_requested:
                     self.get_logger().info('TB0 finished pickup, requesting next parcel spawn.')
                     self.request_spawn_next_parcel()
                     self.parcel_spawn_requested = True
            
            # Stop the robot
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
            self.get_logger().info('Stopping robot')
            
            # If this is TB0, check if we're away from Relaypoint0 and should spawn a new parcel
            if self.namespace == 'tb0' and self.pickup_complete and not self.parcel_spawn_requested:
                # Calculate distance to Relaypoint0
                dx = self.current_state[0] - self.relaypoint0_pose[0]
                dy = self.current_state[1] - self.relaypoint0_pose[1]
                distance_to_relay = np.sqrt(dx**2 + dy**2)
                print(f"Distance to Relaypoint0: {distance_to_relay:.2f}m")
                
                # If TB0 is far enough away from Relaypoint0, request a new parcel
                if distance_to_relay > 0.2:  # Adjust this threshold as needed
                    self.get_logger().info(f'TB0 is {distance_to_relay:.2f}m away from Relaypoint0, requesting new parcel spawn')
                    self.request_spawn_next_parcel()
                    self.parcel_spawn_requested = True # Set flag after requesting

    def request_spawn_next_parcel(self):
        if not self.spawn_parcel_client.service_is_ready():
            self.get_logger().warn('/spawn_next_parcel_service is not available. Cannot request new parcel.')
            return

        self.get_logger().info('Requesting spawn of the next parcel via service.')
        req = Trigger.Request()
        future = self.spawn_parcel_client.call_async(req)
        future.add_done_callback(self.spawn_parcel_response_callback)

    def spawn_parcel_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully requested parcel spawn: {response.message}')
            else:
                self.get_logger().error(f'Failed to request parcel spawn: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call for parcel spawn failed: {str(e)}')

    def calculate_and_output_errors(self, ref_array, predicted_traj, current_state):
        """Calculate and output tracking errors for reference vs predicted trajectory"""
        
        # Current position error
        current_pos_error = np.sqrt((current_state[0] - ref_array[0, 0])**2 + 
                                   (current_state[1] - ref_array[1, 0])**2)
        current_angle_error = abs(current_state[2] - ref_array[2, 0])
        # Normalize angle error to [-pi, pi]
        current_angle_error = min(current_angle_error, 2*np.pi - current_angle_error)
        
        # Predicted trajectory errors at different horizons
        pos_errors = []
        angle_errors = []
        
        for i in range(min(predicted_traj.shape[1], ref_array.shape[1])):
            pos_err = np.sqrt((predicted_traj[0, i] - ref_array[0, i])**2 + 
                             (predicted_traj[1, i] - ref_array[1, i])**2)
            angle_err = abs(predicted_traj[2, i] - ref_array[2, i])
            # Normalize angle error to [-pi, pi]
            angle_err = min(angle_err, 2*np.pi - angle_err)
            
            pos_errors.append(pos_err)
            angle_errors.append(angle_err)
        
        # Terminal error (at the end of prediction horizon)
        if len(pos_errors) > 0:
            terminal_pos_error = pos_errors[-1]
            terminal_angle_error = angle_errors[-1]
        else:
            terminal_pos_error = 0.0
            terminal_angle_error = 0.0
        
        # Average errors over prediction horizon
        avg_pos_error = np.mean(pos_errors) if pos_errors else 0.0
        avg_angle_error = np.mean(angle_errors) if angle_errors else 0.0
        
        # Output comprehensive error information
        print("="*60)
        print(f"TRACKING ERRORS - Parcel {self.current_parcel_index}")
        print("-"*60)
        print(f"Current State: pos=({current_state[0]:.3f}, {current_state[1]:.3f}), θ={current_state[2]:.3f}")
        print(f"Reference[0]: pos=({ref_array[0, 0]:.3f}, {ref_array[1, 0]:.3f}), θ={ref_array[2, 0]:.3f}")
        print(f"Predicted[0]: pos=({predicted_traj[0, 0]:.3f}, {predicted_traj[1, 0]:.3f}), θ={predicted_traj[2, 0]:.3f}")
        print("-"*60)
        print(f"CURRENT ERRORS:")
        print(f"  Position Error: {current_pos_error:.4f} m")
        print(f"  Angle Error:    {current_angle_error:.4f} rad ({np.degrees(current_angle_error):.1f}°)")
        print("-"*60)
        print(f"PREDICTION HORIZON ERRORS:")
        print(f"  Avg Position Error:      {avg_pos_error:.4f} m")
        print(f"  Avg Angle Error:         {avg_angle_error:.4f} rad ({np.degrees(avg_angle_error):.1f}°)")
        print(f"  Terminal Position Error: {terminal_pos_error:.4f} m")
        print(f"  Terminal Angle Error:    {terminal_angle_error:.4f} rad ({np.degrees(terminal_angle_error):.1f}°)")
        
        # Distance to final target
        target_distance = np.sqrt((current_state[0] - self.target_pose[0])**2 + 
                                 (current_state[1] - self.target_pose[1])**2)
        print(f"  Distance to Target:      {target_distance:.4f} m")
        print("="*60)
        
        # Log critical errors
        if current_pos_error > 0.1:
            self.get_logger().warn(f'High position error: {current_pos_error:.3f}m')
        if current_angle_error > 0.3:  # ~17 degrees
            self.get_logger().warn(f'High angle error: {np.degrees(current_angle_error):.1f}°')
        if terminal_pos_error > 0.05:
            self.get_logger().warn(f'High terminal position error: {terminal_pos_error:.3f}m')


class MobileRobotMPC:
    def __init__(self):
        # Physical parameters from TurtleBot3 model.sdf
        self.wheel_radius = 0.033      # m - from SDF
        self.wheel_separation = 0.160  # m - from SDF
        self.robot_mass = 1.0          # kg - approximate total mass
        
        # MPC parameters
        self.N = 10         # Prediction horizon for pickup operations
        self.N_c = 1        # Control horizon - return single control step
        self.dt = 0.1       # Time step
        
        # Weights optimized for pickup operations (position and orientation control)
        self.Q = np.diag([50.0, 50.0, 15.0])  # State weights (x, y, theta) - balanced for pickup
        # Control input weights - allow more aggressive control for pickup
        self.R = np.diag([0.05, 0.02])        # Lower weights for more responsive control
        # Terminal cost weights - important for precise positioning in pickup
        self.F = np.diag([100.0, 100.0, 30.0])  # Higher terminal weights for accuracy
        
        # Velocity constraints for pickup operations (allow backward movement)
        self.max_vel = 0.15      # m/s - conservative for precise pickup
        self.min_vel = -0.10     # m/s - allow backward movement for pickup
        # Angular velocity limit based on wheel constraints
        max_wheel_speed = self.max_vel / self.wheel_radius  # rad/s
        max_angular_vel = max_wheel_speed * self.wheel_radius / (self.wheel_separation / 2)
        self.max_omega = min(np.pi/2, max_angular_vel * 0.8)  # Conservative limit
        self.min_omega = -self.max_omega
        
        # System dimensions
        self.nx = 3   # Number of states (x, y, theta)
        self.nu = 2   # Number of controls (v, omega)
        
        # Reference trajectory
        self.ref_traj = None
        
        # Solver cache for warm starting
        self.last_solution = None
        
        # Initialize MPC
        self.setup_mpc()

    def setup_mpc(self):
        # Optimization problem
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N+1)  # State trajectory (N+1 states)
        self.U = self.opti.variable(self.nu, self.N)    # Control trajectory
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)          # Initial state
        self.ref = self.opti.parameter(self.nx, self.N+1)  # Reference trajectory

        # Cost function with increasing weights toward the end (Follow_controller style)
        cost = 0
        for k in range(self.N):
            # Increasing weight factor as we progress along horizon (helps convergence)
            weight_factor = 1.0 + 3.0 * k / self.N  # Increases from 1.0 to 4.0
            
            # Position error (x,y) - heavily weighted for precise pickup
            pos_error = self.X[:2, k] - self.ref[:2, k]
            cost += weight_factor * ca.mtimes([pos_error.T, self.Q[:2,:2], pos_error])
            
            # Orientation (theta) - separately weighted for good heading tracking
            theta_error = self.X[2, k] - self.ref[2, k]
            # Normalize angle difference to [-pi, pi] to avoid issues with angle wrapping
            theta_error = ca.fmod(theta_error + ca.pi, 2*ca.pi) - ca.pi
            cost += weight_factor * self.Q[2,2] * theta_error**2
            
            # Control cost (decreased for better tracking)
            control_error = self.U[:, k]
            cost += ca.mtimes([control_error.T, self.R, control_error])
            
            # Penalize large changes in angular velocity for smoother motion
            if k > 0:
                omega_change = self.U[1, k] - self.U[1, k-1]
                cost += 0.1 * omega_change**2
            
            # Dynamics constraints
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Strong terminal cost for precise pickup positioning
        pos_terminal_error = self.X[:2, -1] - self.ref[:2, -1]
        cost += 5.0 * ca.mtimes([pos_terminal_error.T, self.F[:2,:2], pos_terminal_error])
        
        # Terminal orientation error with normalization
        theta_terminal_error = self.X[2, -1] - self.ref[2, -1]
        theta_terminal_error = ca.fmod(theta_terminal_error + ca.pi, 2*ca.pi) - ca.pi
        cost += 5.0 * self.F[2,2] * theta_terminal_error**2

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Velocity and angular velocity constraints (allow backward for pickup)
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))

        # Solver settings optimized for pickup operations
        self.opti.minimize(cost)
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 150,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.limited_memory_max_history': 5,
            'ipopt.linear_solver': 'mumps'
        }
        self.opti.solver('ipopt', opts)

    def robot_model(self, x, u):
        """ 
        Simple system dynamics for TurtleBot3 differential drive robot (3-state model)
        x = [x, y, theta] - robot pose
        u = [v, omega] - linear and angular velocity commands
        """
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * ca.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt,                 # theta
        )

    def validate_and_adjust_reference(self, ref_traj, current_state):
        """
        Validate reference trajectory against TurtleBot3 physical constraints
        and adjust if necessary to ensure feasibility
        """
        if ref_traj is None or ref_traj.shape[1] < 2:
            return ref_traj
            
        adjusted_ref = ref_traj.copy()
        
        # Check and adjust for maximum velocity constraints
        for k in range(ref_traj.shape[1] - 1):
            # Calculate required velocity between consecutive points
            dx = ref_traj[0, k+1] - ref_traj[0, k]
            dy = ref_traj[1, k+1] - ref_traj[1, k]
            dtheta = ref_traj[2, k+1] - ref_traj[2, k]
            
            # Required linear velocity
            required_v = np.sqrt(dx**2 + dy**2) / self.dt
            # Required angular velocity
            required_omega = abs(dtheta) / self.dt
            
            # If required velocity exceeds limits, adjust the trajectory
            if required_v > self.max_vel:
                scale_factor = self.max_vel / required_v
                adjusted_ref[0, k+1] = ref_traj[0, k] + dx * scale_factor
                adjusted_ref[1, k+1] = ref_traj[1, k] + dy * scale_factor
                print(f"Adjusted reference point {k+1} for velocity constraint")
                
            if required_omega > self.max_omega:
                scale_factor = self.max_omega / required_omega
                angle_diff = dtheta * scale_factor
                adjusted_ref[2, k+1] = ref_traj[2, k] + angle_diff
                print(f"Adjusted reference point {k+1} for angular velocity constraint")
        
        return adjusted_ref

    def set_reference_trajectory(self, ref_traj):
        # Validate and adjust reference trajectory before setting
        if ref_traj is not None:
            current_state = np.zeros(3)  # Will be set properly in update()
            self.ref_traj = self.validate_and_adjust_reference(ref_traj, current_state)
        else:
            self.ref_traj = ref_traj

    def update(self, current_state):
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.ref, self.ref_traj)
        
        # Set initial guess for warm starting if available
        if self.last_solution is not None:
            # Shift previous solution forward as initial guess
            u_init = np.zeros((self.nu, self.N))
            x_init = np.zeros((self.nx, self.N+1))
            
            # Copy last N-1 control inputs and append last control input
            u_init[:, :self.N-1] = self.last_solution['u'][:, 1:]
            u_init[:, self.N-1] = self.last_solution['u'][:, self.N-1] 
            
            # Copy last N states and propagate final state
            x_init[:, :self.N] = self.last_solution['x'][:, 1:]
            x_init[:, self.N] = self.robot_model_np(x_init[:, self.N-1], u_init[:, self.N-1])
            
            # Set the initial guess
            self.opti.set_initial(self.X, x_init)
            self.opti.set_initial(self.U, u_init)
        
        try:
            sol = self.opti.solve()
            
            # Store solution for warm starting next time
            self.last_solution = {
                'u': sol.value(self.U),
                'x': sol.value(self.X)
            }
            
            # Return single control step for pickup operations
            return sol.value(self.U)[:, :self.N_c]
        except Exception as e:
            print(f"Pickup MPC Solver failed: {str(e)}")
            # Return simple fallback control for pickup
            if self.ref_traj is not None and self.ref_traj.shape[1] > 0:
                ref_point = self.ref_traj[:, 0]
                dx = ref_point[0] - current_state[0]
                dy = ref_point[1] - current_state[1]
                dtheta = ref_point[2] - current_state[2]
                dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                
                distance = np.sqrt(dx**2 + dy**2)
                desired_heading = np.arctan2(dy, dx)
                heading_error = desired_heading - current_state[2]
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                
                # Simple P controller for pickup with backward capability
                if distance > 0.1:  # Far from target
                    v = 0.8 * distance * np.cos(heading_error)  # Allow backward if needed
                    v = np.clip(v, self.min_vel, self.max_vel)
                else:  # Close to target, focus on orientation
                    v = 0.0
                
                omega = 2.0 * heading_error + 1.0 * dtheta  # Combined heading and orientation correction
                omega = np.clip(omega, self.min_omega, self.max_omega)
                
                return np.array([[v], [omega]])
            else:
                return np.zeros((self.nu, self.N_c))
    
    def robot_model_np(self, x, u):
        """System dynamics in numpy for warm starting (3-state model)"""
        return np.array([
            x[0] + u[0] * np.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * np.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt,                 # theta
        ])

    def get_predicted_trajectory(self):
        try:
            return self.opti.debug.value(self.X)
        except:
            return np.zeros((self.nx, self.N+1))


def main(args=None):
    rclpy.init(args=args)
    pickup_controller = PickupController()
    rclpy.spin(pickup_controller)
    pickup_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
