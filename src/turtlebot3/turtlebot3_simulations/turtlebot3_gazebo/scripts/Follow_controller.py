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
import transforms3d.euler as euler
import tf_transformations as tf
import threading
from std_msgs.msg import Bool, Int32
import os
from std_srvs.srv import Trigger, SetBool # SetBool might be unused now

class MobileRobotMPC:
    def __init__(self):
        # MPC parameters
        self.N = 12         # Reduced prediction horizon for faster solving
        self.N_c = 3         # Control horizon - now we'll use this for returning multiple control steps
        self.dt = 0.1        # Time step
        # Much higher weights on position (x,y) to force convergence
        # Increased weights on orientation (theta) and angular velocity (omega) for better tracking
        self.Q = np.diag([30.0, 30.0, 8.0, 0.8, 3.0])  # State weights (x, y, theta, v, omega)
        # Lower control input weights for more aggressive control action
        # Even lower weight on angular control to allow better tracking of reference orientation
        self.R = np.diag([0.03, 0.01])       # Control input weights - even lower weight on angular control
        # Higher terminal cost weights for position and orientation
        self.F = np.diag([60.0, 60.0, 20.0, 1.0, 8.0])  # Terminal cost weights with increased emphasis on orientation
        
        # Velocity constraints
        self.max_vel = 0.15      # m/s
        self.min_vel = 0    # m/s (allow reverse)
        self.max_omega = np.pi/2 # rad/s
        self.min_omega = -np.pi/2 # rad/s
        
        # System dimensions
        self.nx = 5   # Number of states (x, y, theta, v, omega)
        self.nu = 2    # Number of controls (v, omega)
        
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

        # Cost function with increasing weights toward the end
        cost = 0
        for k in range(self.N):
            # Increasing weight factor as we progress along horizon (helps convergence)
            weight_factor = 1.0 + 4.0 * k / self.N  # Increases from 1.0 to 5.0
            
            # Position error (x,y) - more heavily weighted
            pos_error = self.X[:2, k] - self.ref[:2, k]
            cost += weight_factor * ca.mtimes([pos_error.T, self.Q[:2,:2], pos_error])
            
            # Orientation (theta) - separately weighted to ensure good heading tracking
            theta_error = self.X[2, k] - self.ref[2, k]
            # Normalize angle difference to [-pi, pi] to avoid issues with angle wrapping
            theta_error = ca.fmod(theta_error + ca.pi, 2*ca.pi) - ca.pi
            cost += weight_factor * self.Q[2,2] * theta_error**2
            
            # Velocity errors (v, omega) - weighted for accurate tracking
            vel_error = self.X[3:, k] - self.ref[3:, k]
            cost += ca.mtimes([vel_error.T, self.Q[3:,3:], vel_error])
            
            # Special focus on angular velocity tracking
            angular_vel_error = self.X[4, k] - self.ref[4, k]
            cost += 1.5 * weight_factor * self.Q[4,4] * angular_vel_error**2
            
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

        # Much stronger terminal cost for better convergence
        
        # Position terminal error gets highest priority
        pos_term_error = self.X[:2, -1] - self.ref[:2, -1]
        cost += 10.0 * ca.mtimes([pos_term_error.T, self.F[:2,:2], pos_term_error])
        
        # Terminal orientation error with normalization
        theta_term_error = self.X[2, -1] - self.ref[2, -1]
        # Normalize angle difference to [-pi, pi] to avoid issues with angle wrapping
        theta_term_error = ca.fmod(theta_term_error + ca.pi, 2*ca.pi) - ca.pi
        cost += 12.0 * self.F[2,2] * theta_term_error**2
        
        # Terminal velocity errors
        v_term_error = self.X[3, -1] - self.ref[3, -1]
        omega_term_error = self.X[4, -1] - self.ref[4, -1]
        cost += self.F[3,3] * v_term_error**2
        cost += 2.0 * self.F[4,4] * omega_term_error**2  # Extra weight on angular velocity tracking

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)  # Initial condition
        
        # Velocity and angular velocity constraints
        self.opti.subject_to(self.opti.bounded(self.min_vel, self.U[0, :], self.max_vel))
        self.opti.subject_to(self.opti.bounded(self.min_omega, self.U[1, :], self.max_omega))
        
        # Terminal constraints: velocity and angular velocity should approach zero at terminal state
        # This ensures the robot stops at the goal position and doesn't just pass through
        self.opti.subject_to(self.X[3, -1] <= 0.01)  # Terminal linear velocity close to zero
        self.opti.subject_to(self.X[3, -1] >= -0.01) # Terminal linear velocity close to zero
        self.opti.subject_to(self.X[4, -1] <= 0.01)  # Terminal angular velocity close to zero
        self.opti.subject_to(self.X[4, -1] >= -0.01) # Terminal angular velocity close to zero

        # Solver settings optimized for fast convergence
        self.opti.minimize(cost)
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.max_iter': 100,      # Reduced iterations for faster solving
            'ipopt.tol': 1e-4,          # Slightly relaxed tolerance for faster solving
            'ipopt.acceptable_tol': 1e-3, # Slightly relaxed tolerance for faster solving
            'ipopt.warm_start_init_point': 'yes', # Use warm starting for faster convergence
            'ipopt.mu_strategy': 'adaptive', # Use adaptive barrier parameter update strategy
            'ipopt.hessian_approximation': 'limited-memory', # Limited memory BFGS for faster iterations
            'ipopt.limited_memory_max_history': 5, # Limited history size for faster iterations
            'ipopt.linear_solver': 'mumps' # Fast linear solver
        }
        self.opti.solver('ipopt', opts)

    def robot_model(self, x, u):
        """ System dynamics: x_next = f(x, u) """
        return ca.vertcat(
            x[0] + u[0] * ca.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * ca.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt,                 # theta
            u[0],                                  # velocity (directly set)
            u[1]                                   # angular velocity (directly set)
        )

    def set_reference_trajectory(self, ref_traj):
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
            
            # Return all N_c control steps
            return sol.value(self.U)[:, :self.N_c]
        except Exception as e:
            print(f"Solver failed: {str(e)}")
            return np.zeros((self.nu, self.N_c))
    
    def robot_model_np(self, x, u):
        """System dynamics in numpy for warm starting"""
        return np.array([
            x[0] + u[0] * np.cos(x[2]) * self.dt,  # x position
            x[1] + u[0] * np.sin(x[2]) * self.dt,  # y position
            x[2] + u[1] * self.dt,                 # theta
            u[0],                                  # velocity (directly set)
            u[1]                                   # angular velocity (directly set)
        ])

    def get_predicted_trajectory(self):
        return self.opti.debug.value(self.X)
    
class CmdVelPublisher(Node):
    def __init__(self): # Removed namespace argument
        super().__init__('follow_controller_node') # Default node name
        # Declare and get the namespace parameter
        self.declare_parameter('namespace', 'tb_default') # Provide a default value
        self.declare_parameter('case', 'simple_maze')  # Add case parameter
        self.declare_parameter('num_robots', 3)        # Add num_robots parameter
        
        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
        self.case = self.get_parameter('case').get_parameter_value().string_value
        self.num_robots = self.get_parameter('num_robots').get_parameter_value().integer_value
        
        # Current parcel index (defaults to 0 for parcel0)
        self.current_parcel_index = 0
        
        # Subscribe to this robot's specific parcel index topic
        self.parcel_index_sub = self.create_subscription(
            Int32,
            f'/{self.namespace}/current_parcel_index',
            self.parcel_index_callback,
            10
        )
        
        # Get relay point number (namespace number + 1)
        self.namespace_number = self.extract_namespace_number(self.namespace)
        self.relay_point_number = self.namespace_number + 1
        
        # Subscribe to relay point pose (the goal)
        self.relay_point_pose = None
        self.relay_point_sub = self.create_subscription(
            PoseStamped,
            f'/Relaypoint{self.relay_point_number}/pose',
            self.relay_point_callback,
            10
        )
        
        self.pushing_ready_flag = False 
        self.state_lock = threading.Lock()
        self.publisher_ = self.create_publisher(Twist, 'pushing/cmd_vel', 10)
        self.prediction_publisher = self.create_publisher(Path, 'pushing/pred_path', 10)
        self.ref_publisher = self.create_publisher(Path, 'pushing/ref_path', 10)
        self.pushing_complete_flag_pub = self.create_publisher(Bool, 'Pushing_Flag', 10)
        # Initialize Pushing_Flag to False
        self.pushing_complete_flag_pub.publish(Bool(data=False))

        # Service client to report pushing status to State_switch
        self.pushing_finish_client = self.create_client(Trigger, f'/{self.namespace}/pushing_finish') # Renamed client and service
        if not self.pushing_finish_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'Service /{self.namespace}/pushing_finish (Trigger) not available.')
        else:
            self.get_logger().info(f'Service /{self.namespace}/pushing_finish (Trigger) available.')

        self.robot_pose_sub = self.create_subscription(
            Odometry, 'pushing/robot_pose', self.robot_pose_callback, 10)
        
        self.parcel_pose_sub = None
        self.update_parcel_subscription()
            
        # Add start_pushing service server
        self.start_pushing_service = self.create_service(
            Trigger, 
            f'/{self.namespace}/start_pushing', 
            self.start_pushing_callback
        )
        self.get_logger().info(f'/{self.namespace}/start_pushing service created.')
        
        json_file_path = f'/root/workspace/data/{self.case}/{self.namespace}_Trajectory.json'
        with open(json_file_path, 'r') as json_file:
            self.data = json.load(json_file)['Trajectory']
            self.goal = self.data[-1].copy()  # Initialize goal from the last trajectory point
        # Initialize current state (x, y, theta, v, omega)
        self.current_state = np.zeros(5)
        self.parcel_state = np.zeros(3)
        self.last_time = self.get_clock().now()
        self.index = 0
        timer_period = 0.1# seconds
        self.ref_traj= Path()
        self.ref_traj.header.frame_id = 'world'
        self.ref_traj.header.stamp = self.get_clock().now().to_msg()
        self.pre_traj= Path()
        self.pre_traj.header.frame_id = 'world'
        self.pre_traj.header.stamp = self.get_clock().now().to_msg()
        self.MPC=MobileRobotMPC()
        self.P_HOR = self.MPC.N
        
        # Track pushing state and pickup transition
        self.pushing_complete = False
        self.backing_away = False
        self.safe_distance_reached = False
        self.prev_picking_flag = False
        self.trigger_sent_this_cycle = False # Flag to ensure trigger is sent once per cycle
        
        # Store the sequence of control inputs from MPC
        self.control_sequence = None
        self.control_step = 0
        
        # Timing control
        self.timer = self.create_timer(timer_period, self.control_loop)
        
    def parcel_index_callback(self, msg):
        """Handle updates to the current parcel index"""
        new_index = msg.data
        if new_index != self.current_parcel_index:
            with self.state_lock:
                old_index = self.current_parcel_index
                self.current_parcel_index = new_index
                self.get_logger().info(f'Follow controller updated parcel index: {old_index} -> {self.current_parcel_index}')
                # Reset state flags on new parcel
                self.pushing_ready_flag = False
                # Removed self.internal_pushing_status logic
                self.pushing_complete = False
                self.backing_away = False
                self.safe_distance_reached = False
                self.index = 0  # Reset trajectory index
                self.trigger_sent_this_cycle = False # Reset trigger flag for new parcel
                self.pushing_complete_flag_pub.publish(Bool(data=False)) # Reset Pushing_Flag for next robot
                # Update subscription to the correct parcel
                self.update_parcel_subscription()
    
    def update_parcel_subscription(self):
        """Update subscription to the correct parcel topic - always use parcel0"""
        # If we already have a subscription, destroy it
        if self.parcel_pose_sub is not None:
            self.destroy_subscription(self.parcel_pose_sub)
            
        # Always create subscription for parcel0
        self.parcel_pose_sub = self.create_subscription(
            PoseStamped,
            f'/parcel{self.current_parcel_index}/pose',
            self.parcel_pose_callback,
            10
        )
        self.get_logger().info(f'Now tracking parcel{self.current_parcel_index} for pushing')

    def extract_namespace_number(self, namespace):
        """Extract numerical index from namespace string using regex"""
        match = re.search(r'tb(\d+)', namespace)
        if (match):
            return int(match.group(1))
        return 0  # Default to 0 if no number found
    
    def relay_point_callback(self, msg):
        """Process the relay point pose as the goal"""
        with self.state_lock:
            self.relay_point_pose = msg.pose
            
            # Update the goal position based on the relay point
            self.goal[0] = self.relay_point_pose.position.x
            self.goal[1] = self.relay_point_pose.position.y
            
            # Extract yaw from quaternion
            # transforms3d expects [w, x, y, z]
            quat = [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]
            euler = tf.euler_from_quaternion(quat)
            self.goal[2]=euler[2]
            
            self.get_logger().debug(f'Updated goal from relay point: x={self.goal[0]:.2f}, y={self.goal[1]:.2f}, theta={self.goal[2]:.2f}')

    def parcel_pose_callback(self, msg):
        self.parcel_state[0] = msg.pose.position.x
        self.parcel_state[1] = msg.pose.position.y
        quat = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        euler = tf.euler_from_quaternion(quat)
        self.parcel_state[2] = euler[2]

            
    def start_pushing_callback(self, request, response):
        # self.get_logger().info(f'Start pushing service called for {self.namespace}. Current pushing_ready_flag: {self.pushing_ready_flag}')
        if not self.pushing_ready_flag: # Only act if not already ready
            self.pushing_ready_flag = True
            # Reset other relevant state variables for a new pushing task
            self.pushing_complete = False
            self.backing_away = False
            self.safe_distance_reached = False
            self.trigger_sent_this_cycle = False # Reset trigger flag
            # Removed call related to internal_pushing_status
            self.pushing_complete_flag_pub.publish(Bool(data=False)) # Ensure Pushing_Flag is False at start
            self.index = 0 # Reset trajectory index
            self.get_logger().info('Pushing task enabled.')
            response.success = True
            response.message = 'Pushing started.'
        else:
            self.get_logger().warn('Pushing task already enabled or in progress.') # Added warning for else case
            response.success = False
            response.message = 'Pushing already enabled.'
        return response

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
            self.current_state[2] = euler[2]  # yaw
            self.current_state[3] = msg.twist.twist.linear.x   # Linear velocity
            self.current_state[4] = msg.twist.twist.angular.z  # Angular velocity

    def call_pushing_finish_service(self): # Renamed method
        if not self.pushing_finish_client.service_is_ready():
            self.get_logger().warn(f'Service /{self.namespace}/pushing_finish not ready, cannot report completion.')
            return

        if self.trigger_sent_this_cycle:
            self.get_logger().info(f'Pushing completion for /{self.namespace}/pushing_finish already triggered this cycle.')
            return

        request = Trigger.Request()
        self.get_logger().info(f"Calling /{self.namespace}/pushing_finish (Trigger) service to report completion.")
        future = self.pushing_finish_client.call_async(request)
        future.add_done_callback(self.pushing_finish_response_callback) # Renamed callback

    def pushing_finish_response_callback(self, future): # Renamed method
        self.trigger_sent_this_cycle = True 
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully reported pushing completion via /{self.namespace}/pushing_finish (Trigger): {response.message}')
            else:
                self.get_logger().warn(f'Failed to report pushing completion via /{self.namespace}/pushing_finish (Trigger): {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call to /{self.namespace}/pushing_finish failed: {str(e)}')

    def advance_control_step(self):
        """Advance to the next control step in the stored sequence"""
        if self.control_sequence is None:
            return False
        
        self.control_step += 1
        
        # When using a control step from our sequence, we effectively move forward in our trajectory
        # So we should increment our trajectory index accordingly
        self.index += 1
        
        # If we've used all our control steps, need a new MPC solve
        if self.control_step >= self.MPC.N_c:
            self.control_step = 0
            self.control_sequence = None
            return False
        
        # Otherwise we can use the next step from our stored sequence
        return True
    
    def apply_stored_control(self):
        """Apply the current step from the stored control sequence"""
        if self.control_sequence is None or self.control_step >= self.MPC.N_c:
            return False
        
        cmd_msg = Twist()
        cmd_msg.linear.x = self.control_sequence[0, self.control_step]
        cmd_msg.angular.z = self.control_sequence[1, self.control_step]
        self.publisher_.publish(cmd_msg)
        
        self.get_logger().debug(f'Applied stored control step {self.control_step+1}/{self.MPC.N_c}: '
                                f'v={cmd_msg.linear.x:.2f}, ω={cmd_msg.angular.z:.2f}')
        return True

    def control_loop(self):
        # print(f'Control loop called with index: {self.index}, pushing_ready_flag: {self.pushing_ready_flag}', flush=True)
        if not self.pushing_ready_flag:
            # No service call or status change needed here if not ready.
            # Pushing_Flag is managed by start_pushing or parcel_index callbacks.
            return
            
        # Check if we have a valid stored control sequence we can use
        if self.control_sequence is not None:
            # Use the stored control sequence if available
            if self.apply_stored_control():
                self.advance_control_step()
                return
        
        if self.goal is None or self.parcel_state is None:
            self.get_logger().warn("Goal or parcel state not yet available.")
            return

        try:
            # 1. Update reference trajectory
            distance_error = np.sqrt((self.current_state[0]-self.goal[0])**2+(self.current_state[1]-self.goal[1])**2)
            object_distance = np.sqrt((self.goal[0]-self.parcel_state[0])**2+(self.goal[1]-self.parcel_state[1])**2)
            robot_parcel_distance = np.sqrt((self.current_state[0]-self.parcel_state[0])**2+(self.current_state[1]-self.parcel_state[1])**2)
            
            # Check if parcel reached the goal position and start backing away immediately
            if distance_error <= 0.15 and not self.pushing_complete:
                # self.get_logger().info(f'Parcel{self.current_parcel_index} reached goal. Distance to goal: {object_distance:.2f}m')
                self.pushing_complete = True
                self.backing_away = True
                self.pushing_complete_flag_pub.publish(Bool(data=self.pushing_complete))  # Corrected: Publish Bool message, data is True
                # Removed internal_pushing_status logic and premature service call
                
                # Start backing away immediately without waiting for next loop iteration
                cmd_msg = Twist()
                cmd_msg.linear.x = -0.08  # Faster reverse speed for immediate response
                cmd_msg.angular.z = 0.0
                self.publisher_.publish(cmd_msg)
                self.get_logger().info('Backing away immediately!')
                
                # Removed spawn_next_parcel publication from here - now handled in PickUp_controller.py
                return
            
            if self.backing_away:
                # Continue backing away from the parcel to a safe distance
                distance_from_goal = np.sqrt((self.current_state[0]-self.goal[0])**2 + (self.current_state[1]-self.goal[1])**2)
                cmd_msg = Twist()
                # Pushing status is still true during backing away

                if distance_from_goal < 0.35:  # Still need to back up more
                    cmd_msg.linear.x = -0.08
                    cmd_msg.angular.z = 0.0
                    self.publisher_.publish(cmd_msg)
                    self.pushing_complete_flag_pub.publish(Bool(data=self.pushing_complete)) 
                    # self.get_logger().info(f'Backing to safe distance: current={distance_from_goal:.2f}m, target=0.35m')
                    
                else:
                    # Safe distance reached, stop backing
                    self.backing_away = False
                    self.safe_distance_reached = True
                    cmd_msg.linear.x = 0.0
                    cmd_msg.angular.z = 0.0
                    self.publisher_.publish(cmd_msg)
                    self.get_logger().info('Safe distance reached. Pushing phase concluded.')
                    
                    # Report completion to own State_switch
                    self.call_pushing_finish_service() # Corrected: No arguments, and renamed method
                    # Signal this robot's pushing phase for this parcel is over
                    self.pushing_complete_flag_pub.publish(Bool(data=True)) 
                    
                    # Reset flags for next cycle
                    self.pushing_ready_flag = False #
                    self.index = 0 # Reset trajectory index
                return # Exit control loop iteration as backing away is handled                # Normal path following with MPC
            # This part should only run if not pushing_complete (i.e. not backing_away and not safe_distance_reached from this push cycle)
            if not self.pushing_complete and self.index < len(self.data):  
                # Calculate error to current reference point
                curr_ref_idx = min(self.index, len(self.data)-1)
                curr_pos = np.array([self.current_state[0], self.current_state[1]])
                ref_pos = np.array([self.data[curr_ref_idx][0], self.data[curr_ref_idx][1]])
                curr_pos_error = np.linalg.norm(curr_pos - ref_pos)
                
                # Calculate distance to final point for displaying progress
                final_pos = np.array([self.data[-1][0], self.data[-1][1]])
                dist_to_final = np.linalg.norm(curr_pos - final_pos)
        
                needs_replanning = (
                    self.control_sequence is None or 
                    self.control_step >= self.MPC.N_c - 1
                )
                
                # For debugging
                if curr_pos_error > 0.05:
                    self.get_logger().debug(f"Current error: {curr_pos_error:.4f}m, Distance to goal: {dist_to_final:.2f}m")
                
                if needs_replanning:
                    # Find the closest point in trajectory to our current position before replanning
                    curr_pos = np.array([self.current_state[0], self.current_state[1]])
                    best_idx = self.index
                    min_dist = float('inf')
                    
                    # Look around our current index to find the closest reference point
                    search_radius = 5  # Search radius
                    search_start = max(0, self.index - search_radius)
                    search_end = min(len(self.data), self.index + search_radius)
                    
                    for idx in range(search_start, search_end):
                        ref_pos = np.array([self.data[idx][0], self.data[idx][1]])
                        dist = np.linalg.norm(curr_pos - ref_pos)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx
                    
                    # Update index based on closest point to ensure accurate reference
                    # But don't let it go backwards too much
                    self.index = max(self.index - 2, best_idx)
                    
                    # Prepare reference trajectory arrays and visualization
                    ref_array = np.zeros((5, self.P_HOR+1))  # N+1 points for reference trajectory
                    self.ref_traj.poses = []  # Clear existing poses
                    current_timestamp = self.get_clock().now().to_msg()
                    self.ref_traj.header.stamp = current_timestamp
                    
                    # First, interpolate the trajectory for smoother reference
                    interpolated_traj = []
                    
                    # Only interpolate if we have enough points and not at the end
                    if self.index < len(self.data) - 1:
                        # Get number of segments based on our horizon
                        segments = min(self.P_HOR, len(self.data) - self.index - 1)
                        
                        for i in range(segments):
                            # Get original waypoints
                            idx = self.index + i
                            p1 = np.array([self.data[idx][0], self.data[idx][1], self.data[idx][2]])
                            p2 = np.array([self.data[idx+1][0], self.data[idx+1][1], self.data[idx+1][2]])
                            v1 = self.data[idx][3]
                            v2 = self.data[idx+1][3]
                            
                            # Normalize theta difference
                            p2[2] = p1[2] + ((p2[2] - p1[2] + np.pi) % (2 * np.pi) - np.pi)
                            
                            # Add first point
                            interpolated_traj.append(
                                [p1[0], p1[1], p1[2], v1, self.data[idx][4]]
                            )
                            
                            # Add interpolated points (only if segments are large enough)
                            dist = np.linalg.norm(p2[:2] - p1[:2])
                            if dist > 0.05:  # Only interpolate if points are far enough
                                num_interp = min(2, int(dist / 0.05))  # Adaptive interpolation
                                for j in range(1, num_interp):
                                    t = j / num_interp
                                    # Linear interpolation
                                    interp_p = p1 + t * (p2 - p1)
                                    interp_v = v1 + t * (v2 - v1)
                                    interp_omega = (p2[2] - p1[2]) / (self.MPC.dt * num_interp)
                                    
                                    interpolated_traj.append(
                                        [interp_p[0], interp_p[1], interp_p[2], interp_v, interp_omega]
                                    )
                        
                        # Add the final point from the original trajectory
                        last_idx = min(self.index + segments, len(self.data) - 1)
                        interpolated_traj.append(
                            [self.data[last_idx][0], self.data[last_idx][1], 
                             self.data[last_idx][2], self.data[last_idx][3], self.data[last_idx][4]]
                        )
                        
                        # Fill any remaining positions with the final point (if needed)
                        while len(interpolated_traj) < self.P_HOR + 1:
                            final_point = interpolated_traj[-1].copy()
                            final_point[3] = 0.0  # Zero velocity at end
                            final_point[4] = 0.0  # Zero angular velocity at end
                            interpolated_traj.append(final_point)
                    else:
                        # We're at the end of the trajectory, just use the final point
                        final_idx = len(self.data) - 1
                        final_point = [
                            self.data[final_idx][0], 
                            self.data[final_idx][1], 
                            self.data[final_idx][2], 
                            0.0,  # Zero velocity at end
                            0.0   # Zero angular velocity at end
                        ]
                        
                        for i in range(self.P_HOR + 1):
                            interpolated_traj.append(final_point.copy())
                    
                    # Now build reference trajectory from interpolated points
                    for i in range(min(self.P_HOR + 1, len(interpolated_traj))):
                        # Create visualization message
                        pose_msg = PoseStamped()
                        pose_msg.header.frame_id = "world"
                        pose_msg.header.stamp = current_timestamp
                        pose_msg.pose.position.x = interpolated_traj[i][0]
                        pose_msg.pose.position.y = interpolated_traj[i][1]
                        pose_msg.pose.position.z = 0.0
                        
                        yaw = interpolated_traj[i][2]
                        quat = tf.quaternion_from_euler(0.0, 0.0, yaw)
                        pose_msg.pose.orientation.x = quat[1]
                        pose_msg.pose.orientation.y = quat[2]
                        pose_msg.pose.orientation.z = quat[3]
                        pose_msg.pose.orientation.w = quat[0]
                        
                        self.ref_traj.poses.append(pose_msg)
                        
                        # Update reference array for MPC
                        ref_array[0, i] = interpolated_traj[i][0]  # x
                        ref_array[1, i] = interpolated_traj[i][1]  # y
                        ref_array[2, i] = interpolated_traj[i][2]  # theta
                        
                        # Use appropriate velocity references
                        if i < len(interpolated_traj) - 2:  # Not near end
                            ref_array[3, i] = interpolated_traj[i][3]  # v
                            ref_array[4, i] = interpolated_traj[i][4]  # omega
                        else:
                            # Gradually reduce velocities near the end
                            end_factor = (len(interpolated_traj) - i) / 2.0
                            ref_array[3, i] = interpolated_traj[i][3] * end_factor  # v
                            ref_array[4, i] = interpolated_traj[i][4] * end_factor  # omega
                
                self.ref_publisher.publish(self.ref_traj)

                current_state = self.current_state.copy()
                
                try:
                    self.MPC.set_reference_trajectory(ref_array)
                    u_sequence = self.MPC.update(current_state)  # This now contains N_c control steps
                    
                    # Check if controls are valid
                    if np.isnan(u_sequence).any():
                        self.get_logger().error("MPC returned NaN control values - using safe fallback controls")
                        # Safe fallback - slow forward movement
                        cmd_msg = Twist()
                        cmd_msg.linear.x = 0.05
                        cmd_msg.angular.z = 0.0
                        self.publisher_.publish(cmd_msg)
                        return
                    
                    # Store the N_c control steps for future use
                    self.control_sequence = u_sequence
                    self.control_step = 0
                    
                    # Apply first control command 
                    cmd_msg = Twist()
                    cmd_msg.linear.x = u_sequence[0, 0]  # First control step, linear velocity
                    cmd_msg.angular.z = u_sequence[1, 0]  # First control step, angular velocity
                    
                    # Check for goal proximity - slow down when close to final waypoint
                    curr_pos = np.array([self.current_state[0], self.current_state[1]])
                    final_pos = np.array([self.data[-1][0], self.data[-1][1]])
                    dist_to_final = np.linalg.norm(curr_pos - final_pos)
                    
                    # Calculate heading error to nearest target
                    nearest_idx = min(self.index, len(self.data)-1)
                    target_theta = self.data[nearest_idx][2]
                    current_theta = self.current_state[2]
                    theta_error = ((target_theta - current_theta + np.pi) % (2 * np.pi)) - np.pi
                    abs_theta_error = abs(theta_error)
                    
                    # Prioritize angular velocity correction when heading error is large
                    if abs_theta_error > 0.3:  # ~17 degrees
                        # Amplify angular velocity when heading error is significant
                        heading_correction_factor = min(2.0, 1.0 + abs_theta_error)
                        cmd_msg.angular.z *= heading_correction_factor
                        # Reduce linear velocity to allow better turning
                        cmd_msg.linear.x *= max(0.4, 1.0 - abs_theta_error/2.0)
                        self.get_logger().debug(f"Large heading error: {abs_theta_error:.2f}rad, adjusting controls")
                    
                    # Slow down when close to goal for smoother convergence
                    if dist_to_final < 0.25:
                        slow_factor = max(0.3, dist_to_final / 0.25)  # Scale from 0.3 to 1.0
                        cmd_msg.linear.x *= slow_factor
                        # Don't slow down angular velocity as much to maintain heading accuracy
                        cmd_msg.angular.z *= min(0.8, slow_factor * 1.2)
                        self.get_logger().debug(f"Close to goal ({dist_to_final:.2f}m), slowing down by factor {slow_factor:.2f}")
                    
                    self.publisher_.publish(cmd_msg)
                    
                    # Log the control action
                    self.get_logger().debug(f'MPC control: v={cmd_msg.linear.x:.3f}, ω={cmd_msg.angular.z:.3f}')
                    
                    # Adjust index based on progress along trajectory
                    # This helps ensure we track the reference better by synchronizing with actual progress
                    min_dist = float('inf')
                    best_idx = self.index
                    
                    # Look in a window around current index to find closest reference point
                    search_start = max(0, self.index - 2)
                    search_end = min(len(self.data), self.index + 5)
                    
                    for i in range(search_start, search_end):
                        ref_pos = np.array([self.data[i][0], self.data[i][1]])
                        dist = np.linalg.norm(curr_pos - ref_pos)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = i
                    
                    # Update index but don't go backwards
                    self.index = max(self.index, best_idx)
                    # Progress by at least 1
                    self.index += 1
                    
                    # Publish predicted trajectory with synchronized timestamps
                    pred_traj = self.MPC.get_predicted_trajectory()
                    if pred_traj is None:
                        self.get_logger().warn("Failed to get predicted trajectory")
                    else:
                        # Use the same timestamp as the reference trajectory for proper alignment
                        self.pre_traj.header.stamp = self.ref_traj.header.stamp
                        self.pre_traj.poses = []
                        
                        # Check for valid prediction trajectory
                        if isinstance(pred_traj, np.ndarray) and not np.isnan(pred_traj).any():
                            for i in range(min(self.P_HOR+1, pred_traj.shape[1])):
                                pose_msg = PoseStamped()
                                pose_msg.header.frame_id = "world"
                                pose_msg.header.stamp = self.ref_traj.header.stamp
                                pose_msg.pose.position.x = pred_traj[0, i]
                                pose_msg.pose.position.y = pred_traj[1, i]
                                pose_msg.pose.position.z = 0.0
                                
                                # Ensure we have valid orientation
                                yaw = pred_traj[2, i]
                                if not np.isnan(yaw):
                                    quat = tf.quaternion_from_euler(0.0, 0.0, yaw)
                                    pose_msg.pose.orientation.x = quat[1]
                                    pose_msg.pose.orientation.y = quat[2]
                                    pose_msg.pose.orientation.z = quat[3]
                                    pose_msg.pose.orientation.w = quat[0]
                                else:
                                    # Default orientation if invalid
                                    pose_msg.pose.orientation.w = 1.0
                                
                                self.pre_traj.poses.append(pose_msg)
                            
                            # Publish the predicted trajectory
                            self.prediction_publisher.publish(self.pre_traj)
                            
                            # Debug output to check alignment of trajectories
                            if self.pre_traj.poses and self.ref_traj.poses and len(self.pre_traj.poses) > 0 and len(self.ref_traj.poses) > 0:
                                # Calculate and log the maximum deviation between reference and predicted
                                max_deviation = 0.0
                                for i in range(min(len(self.pre_traj.poses), len(self.ref_traj.poses))):
                                    pred_pos = np.array([self.pre_traj.poses[i].pose.position.x, 
                                                        self.pre_traj.poses[i].pose.position.y])
                                    ref_pos = np.array([self.ref_traj.poses[i].pose.position.x, 
                                                       self.ref_traj.poses[i].pose.position.y])
                                    deviation = np.linalg.norm(pred_pos - ref_pos)
                                    max_deviation = max(max_deviation, deviation)
                                
                                # Only log significant deviations to avoid console spam
                                if max_deviation > 0.1:
                                    self.get_logger().debug(f"Max deviation between predicted and reference: {max_deviation:.4f}m")
                    
                except Exception as e:
                    self.get_logger().error(f'Control error: {str(e)}')
            
            elif self.safe_distance_reached:
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.publisher_.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            self.timer.cancel()

    
def main(args=None):
    # Start debugpy listener
    print(f"[FollowController main START] Called with args: {args}", flush=True)
    rclpy.init(args=args)
    
    cmd_vel_publisher_node = None # Initialize to None
    print(f"[FollowController main] Creating CmdVelPublisher instance.", flush=True)
    cmd_vel_publisher_node = CmdVelPublisher()
    current_node_name = cmd_vel_publisher_node.get_fully_qualified_name()
    print(f"[FollowController main] CmdVelPublisher instance created. Node name: {current_node_name}. Spinning...", flush=True)
    rclpy.spin(cmd_vel_publisher_node)
    cmd_vel_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
        main()