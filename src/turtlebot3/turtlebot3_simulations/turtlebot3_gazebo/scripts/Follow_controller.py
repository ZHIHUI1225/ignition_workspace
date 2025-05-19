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
        self.N = 25          # Prediction horizon
        self.N_c=10        # Control horizon
        self.dt = 0.1        # Time step
        self.Q = np.diag([12, 12, 5, 1, 1])  # State weights (x, y, theta, v, omega)
        self.R = np.diag([0.1, 0.1])         # Control input weights
        self.F = np.diag([12, 12, 5, 1, 1])           # Terminal cost weights
        
        # Velocity constraints
        self.max_vel = 0.1      # m/s
        self.min_vel = -0.1     # m/s (allow reverse)
        self.max_omega = np.pi/2 # rad/s
        self.min_omega = -np.pi/2 # rad/s
        
        # System dimensions
        self.nx = 5   # Number of states (x, y, theta, v, omega)
        self.nu = 2    # Number of controls (v, omega)
        
        # Reference trajectory
        self.ref_traj = None
        
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

        # Cost function
        cost = 0
        for k in range(self.N):
            state_error = self.X[:, k] - self.ref[:, k]
            cost += ca.mtimes([state_error.T, self.Q, state_error])
            
            control_error = self.U[:, k]
            cost += ca.mtimes([control_error.T, self.R, control_error])
            
            # Dynamics constraints - this was wrong in the original code
            x_next = self.robot_model(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal cost
        terminal_error = self.X[:, -1] - self.ref[:, -1]
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
            x[2] + u[1] * self.dt,                 # theta
            u[0],                                  # velocity (directly set)
            u[1]                                   # angular velocity (directly set)
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
    
class CmdVelPublisher(Node):
    def __init__(self): # Removed namespace argument
        super().__init__('follow_controller_node') # Default node name
        # Declare and get the namespace parameter
        self.declare_parameter('namespace', 'tb_default') # Provide a default value
        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
        
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
        
        json_file_path = f'/root/workspace/data/{self.namespace}_DiscreteTrajectory.json'
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

    def control_loop(self):
        # print(f'Control loop called with index: {self.index}, pushing_ready_flag: {self.pushing_ready_flag}', flush=True)
        if not self.pushing_ready_flag:
            # No service call or status change needed here if not ready.
            # Pushing_Flag is managed by start_pushing or parcel_index callbacks.
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
            if object_distance <= 0.06 and not self.pushing_complete:
                self.get_logger().info(f'Parcel{self.current_parcel_index} reached goal. Distance to goal: {object_distance:.2f}m')
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
                    cmd_msg.linear.x = -0.05
                    cmd_msg.angular.z = 0.0
                    self.publisher_.publish(cmd_msg)
                    self.pushing_complete_flag_pub.publish(Bool(data=self.pushing_complete)) 
                    self.get_logger().info(f'Backing to safe distance: current={distance_from_goal:.2f}m, target=0.35m')
                    
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
                return # Exit control loop iteration as backing away is handled
            
            # Normal path following with MPC
            # This part should only run if not pushing_complete (i.e. not backing_away and not safe_distance_reached from this push cycle)
            if not self.pushing_complete and self.index < len(self.data):  
                # Ensure status is false if we are in normal path following before pushing starts
                # Removed internal_pushing_status logic and service call

                # print(f'index: {self.index}, distance_error: {distance_error:.2f}')
                ref_array = np.zeros((5, self.P_HOR+1))  # N+1 points for reference trajectory
                self.ref_traj.poses = []  # Clear existing poses
                
                for i in range(self.P_HOR+1):  # +1 for initial state
                    pose_msg = PoseStamped()
                    pose_msg.header = Header(frame_id="world")
                    pose_msg.pose.position.x = self.data[min(self.index+i,len(self.data)-1)][0]
                    pose_msg.pose.position.y = self.data[min(self.index+i,len(self.data)-1)][1]
                    pose_msg.pose.position.z = 0.0
                    
                    yaw = self.data[min(self.index+i,len(self.data)-1)][2]
                    # Convert yaw to quaternion
                    quat = tf.quaternion_from_euler(0.0, 0.0, yaw)
                    pose_msg.pose.orientation.x = quat[1]
                    pose_msg.pose.orientation.y = quat[2]
                    pose_msg.pose.orientation.z = quat[3]
                    pose_msg.pose.orientation.w = quat[0]
                    
                    self.ref_traj.poses.append(pose_msg)
                    
                    # Update reference trajectory for MPC
                    ref_array[0, i] = self.data[min(self.index+i,len(self.data)-1)][0]  # x
                    ref_array[1, i] = self.data[min(self.index+i,len(self.data)-1)][1]  # y
                    ref_array[2, i] = self.data[min(self.index+i,len(self.data)-1)][2]  # theta
                    if self.index+i<len(self.data)-1:
                        ref_array[3, i] = self.data[self.index+i][3]  # v
                        ref_array[4, i] = self.data[self.index+i][4]  # omega
                    else:
                        ref_array[3, i] = 0
                        ref_array[4, i] = 0
                
                self.ref_publisher.publish(self.ref_traj)

                current_state = self.current_state.copy()
                
                try:
                    self.MPC.set_reference_trajectory(ref_array)
                    u = self.MPC.update(current_state)
                    
                    # 4. Publish control command based on MPC output
                    cmd_msg = Twist()
                    cmd_msg.linear.x = u[0]
                    cmd_msg.angular.z = u[1]
                    self.publisher_.publish(cmd_msg)
                    # print(f'Publishing cmd_vel: linear.x={cmd_msg.linear.x:.2f}, angular.z={cmd_msg.angular.z:.2f}')
                    
                    # Publish predicted trajectory
                    pred_traj = self.MPC.get_predicted_trajectory()
                    self.pre_traj.poses = []
                    for i in range(self.P_HOR+1):
                        pose_msg = PoseStamped()
                        pose_msg.header = Header(frame_id="world")
                        pose_msg.pose.position.x = pred_traj[0, i]
                        pose_msg.pose.position.y = pred_traj[1, i]
                        pose_msg.pose.position.z = 0.0
                        yaw = pred_traj[2, i]
                        # Convert yaw to quaternion
                        quat = tf.quaternion_from_euler(0.0, 0.0, yaw)
                        pose_msg.pose.orientation.x = quat[1]
                        pose_msg.pose.orientation.y = quat[2]
                        pose_msg.pose.orientation.z = quat[3]
                        pose_msg.pose.orientation.w = quat[0]
                        self.pre_traj.poses.append(pose_msg)
                    self.prediction_publisher.publish(self.pre_traj)
                    self.index += 1
                    
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