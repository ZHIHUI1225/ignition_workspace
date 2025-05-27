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
            if self.namespace == 'tb0':
                self.trajectory_data = self.data[::-1]
            else:
                self.trajectory_data=self.data[::-1][::2]
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
            
            v = 0.1  # Low linear velocity for final approach
            omega = 0.0  # No rotation
            
            interp_points.append([interp_x, interp_y, interp_theta, v, omega])

        # Add interpolation points to the beginning of trajectory_data
        # This means they'll be executed last in the reverse trajectory
        self.trajectory_data =  self.trajectory_data +interp_points 

        # Create MPC controller
        self.mpc = MobileRobotMPC()
        self.prediction_horizon = self.mpc.N
        
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
        
        if distance_error > 0.08 and not self.pickup_complete:
            # Find the closest point in the trajectory to the current position
            min_dist = float('inf')
            closest_idx = 0
            
            for i, point in enumerate(self.trajectory_data):
                dist = np.sqrt((self.current_state[0] - point[0])**2 + 
                            (self.current_state[1] - point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Generate reference trajectory from trajectory data
            ref_array = np.zeros((3, self.prediction_horizon + 1))
            
            for i in range(self.prediction_horizon + 1):
                # Use trajectory points, but don't go beyond array bounds
                traj_idx = min(closest_idx + i, len(self.trajectory_data) - 1)
                point = self.trajectory_data[traj_idx]
                
                ref_array[0, i] = point[0]  # x
                ref_array[1, i] = point[1]  # y
                ref_array[2, i] = point[2]  # theta
                
            
            # Apply MPC to get control inputs
            with self.state_lock:
                current_state = self.current_state.copy()
            
            try:
                self.mpc.set_reference_trajectory(ref_array)
                u = self.mpc.update(current_state)
                
                # Send control commands
                cmd_msg = Twist()
                cmd_msg.linear.x = u[0]
                cmd_msg.angular.z = u[1]
                self.cmd_vel_pub.publish(cmd_msg)
                
                # Publish predicted trajectory for visualization
                predicted_traj = self.mpc.get_predicted_trajectory()
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


class MobileRobotMPC:
    def __init__(self):
        # MPC parameters
        self.N = 10           # Prediction horizon
        self.dt = 0.1         # Time step
        self.Q = np.diag([10, 10, 1])  # State weights (x, y, theta, v, omega)
        self.R = np.diag([0.1, 0.1])        # Control input weights
        self.F = np.diag([20, 20, 10]) # Terminal cost weights
        
        # Velocity constraints
        self.max_vel = 0.15     # m/s
        self.min_vel = 0.0    # m/s 
        self.max_omega = np.pi/4 # rad/s
        self.min_omega = -np.pi/4 # rad/s
        
        # System dimensions
        self.nx = 3   # Number of states (x, y, theta)
        self.nu = 2   # Number of controls (v, omega)
        
        # Reference trajectory
        self.ref_traj = None
        
        # Initialize MPC
        self.setup_mpc()

    def setup_mpc(self):
        # Optimization problem
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N+1)  # State trajectory
        self.U = self.opti.variable(self.nu, self.N)    # Control trajectory
        
        # Parameters
        self.x0 = self.opti.parameter(self.nx)          # Initial state
        self.ref = self.opti.parameter(self.nx, self.N+1)  # Reference trajectory

        # Cost function
        cost = 0
        for k in range(self.N):
            state_error = self.X[:, k] - self.ref[:, k]
            cost += ca.mtimes([state_error.T, self.Q, state_error])
            
            control_error = self.U[:, k]
            cost += ca.mtimes([control_error.T, self.R, control_error])
            
            # Dynamics constraints
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost
        terminal_error = 5*(self.X[:, -1] - self.ref[:, -1])
        cost += ca.mtimes([terminal_error.T, self.F, terminal_error])

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Velocity and angular velocity constraints
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))

        # Solver settings
        self.opti.minimize(cost)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100, 'ipopt.tol': 1e-4}
        self.opti.solver('ipopt', opts)

    def robot_model(self, x, u):
        """ System dynamics: x_next = f(x, u) """
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * ca.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt             # theta                                
        )

    def set_reference_trajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def update(self, current_state):
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.ref, self.ref_traj)
        
        try:
            sol = self.opti.solve()
            return sol.value(self.U)[:, 0]  # Return first control input
        except:
            print("Solver failed")
            return np.zeros(self.nu)

    def get_predicted_trajectory(self):
        return self.opti.debug.value(self.X)


def main(args=None):
    rclpy.init(args=args)
    pickup_controller = PickupController()
    rclpy.spin(pickup_controller)
    pickup_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
