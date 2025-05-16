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
import threading
from std_msgs.msg import Bool
import os

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
    def __init__(self):
        super().__init__('follow_controller_node')
        # Declare and get the namespace parameter
        self.declare_parameter('namespace', 'tb0')  # Default to tb0
        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
        
        self.get_logger().info(f'Initializing follow controller for robot: {self.namespace}')
        
        self.Pushing_flag = Bool()
        self.Pushing_flag.data = False
        self.Ready_flag = False
        self.state_lock = threading.Lock()
        
        # Create publishers with proper namespaces
        cmd_vel_topic = f'/{self.namespace}/cmd_vel'
        self.publisher_ = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.get_logger().info(f'Publishing cmd_vel on: {cmd_vel_topic}')
        
        pred_path_topic = f'/{self.namespace}/pred_path'
        self.prediction_publisher = self.create_publisher(Path, pred_path_topic, 10)
        
        ref_path_topic = f'/{self.namespace}/ref_path'
        self.ref_publisher = self.create_publisher(Path, ref_path_topic, 10)
        
        # Create robot-specific Pushing_flag topic
        pushing_flag_topic = f'/{self.namespace}/Pushing_flag'
        self.pushing_flag_pub = self.create_publisher(Bool, pushing_flag_topic, 10)
        self.get_logger().info(f'Publishing Pushing_flag on: {pushing_flag_topic}')
        
        # Add subscribers with proper namespaces
        robot_pose_topic = f'/{self.namespace}/odom'
        self.robot_pose_sub = self.create_subscription(
            Odometry, robot_pose_topic, self.robot_pose_callback, 10)
        self.get_logger().info(f'Subscribing to robot pose on: {robot_pose_topic}')
        
        # Object pose is shared across robots
        self.parcel_pose_sub = self.create_subscription(
            PoseStamped, '/parcel/pose', self.parcel_pose_callback, 10)
        
        # Subscribe to robot-specific Ready_flag
        ready_flag_topic = f'/{self.namespace}/Ready_flag'
        self.Ready_flag_sub = self.create_subscription(
            Bool, ready_flag_topic, self.Ready_flag_callback, 10)
        self.get_logger().info(f'Subscribing to Ready_flag on: {ready_flag_topic}')
        
        # Load robot-specific trajectory
        json_file_path = f'/root/workspace/data/{self.namespace}_DiscreteTrajectory.json'
        self.get_logger().info(f'Loading trajectory from: {json_file_path}')
        
        try:
            with open(json_file_path, 'r') as json_file:
                self.data = json.load(json_file)['Trajectory']
                self.goal = self.data[-1]
                self.get_logger().info(f'Loaded trajectory with {len(self.data)} points. Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
        except Exception as e:
            self.get_logger().error(f'Failed to load trajectory file: {e}')
            # Provide a minimal fallback trajectory
            self.data = [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.1, 0.0]]
            self.goal = self.data[-1]
        
        # Initialize current state (x, y, theta, v, omega)
        self.current_state = np.zeros(5)
        self.parcel_state = np.zeros(3)
        self.last_time = self.get_clock().now()
        self.index = 0
        timer_period = 0.1  # seconds
        
        # Setup path messages
        self.ref_traj = Path()
        self.ref_traj.header.frame_id = 'map'
        self.ref_traj.header.stamp = self.get_clock().now().to_msg()
        
        self.pre_traj = Path()
        self.pre_traj.header.frame_id = 'map'
        self.pre_traj.header.stamp = self.get_clock().now().to_msg()
        
        # Initialize MPC controller
        self.MPC = MobileRobotMPC()
        self.P_HOR = self.MPC.N
        
        # Start control loop timer
        self.timer = self.create_timer(timer_period, self.control_loop)
        self.get_logger().info(f'Follow controller for {self.namespace} initialized and ready')
    
    def parcel_pose_callback(self, msg):
        # Store parcel state from message
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
    
    def Ready_flag_callback(self, msg):
        old_flag = self.Ready_flag
        self.Ready_flag = msg.data
        if old_flag != self.Ready_flag:
            self.get_logger().info(f'Ready_flag changed: {old_flag} -> {self.Ready_flag}')
    
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
    
    def control_loop(self):
        # Skip control if not ready
        if self.Ready_flag is False:
            return
            
        try:
            # Calculate distances
            distance_error = np.sqrt((self.current_state[0]-self.goal[0])**2 + 
                                    (self.current_state[1]-self.goal[1])**2)
            object_distance = np.sqrt((self.goal[0]-self.parcel_state[0])**2 + 
                                     (self.goal[1]-self.parcel_state[1])**2)
            
            # MPC controller works when:
            # 1. We still have trajectory points to follow (index < len(data))
            # 2. Object is not already at destination (object_distance > threshold)
            # 3. We're not in pushing mode
            if self.index < len(self.data) and object_distance > 0.06 and self.Pushing_flag.data is False:
                # In following mode
                self.Pushing_flag.data = False
                self.pushing_flag_pub.publish(self.Pushing_flag)
                
                self.get_logger().debug(f'Following trajectory: index={self.index}, distance={distance_error:.2f}')
                
                # Prepare reference trajectory for MPC
                ref_array = np.zeros((5, self.P_HOR+1))
                self.ref_traj.poses = []
                
                # Fill trajectory points
                for i in range(self.P_HOR+1):
                    idx = min(self.index+i, len(self.data)-1)
                    
                    # Create pose message
                    pose_msg = PoseStamped()
                    pose_msg.header = Header(frame_id="map")
                    pose_msg.pose.position.x = self.data[idx][0]
                    pose_msg.pose.position.y = self.data[idx][1]
                    pose_msg.pose.position.z = 0.0
                    
                    # Convert yaw to quaternion
                    yaw = self.data[idx][2]
                    quat = tf.quaternion_from_euler(0.0, 0.0, yaw)
                    pose_msg.pose.orientation.x = quat[0]
                    pose_msg.pose.orientation.y = quat[1]
                    pose_msg.pose.orientation.z = quat[2]
                    pose_msg.pose.orientation.w = quat[3]
                    
                    self.ref_traj.poses.append(pose_msg)
                    
                    # Update reference trajectory for MPC
                    ref_array[0, i] = self.data[min(self.index+i,len(self.data)-1)][0]  # x
                    ref_array[1, i] = self.data[min(self.index+i,len(self.data)-1)][1]  # y
                    ref_array[2, i] = self.data[min(self.index+i,len(self.data)-1)][2]  # theta
                    if self.index+i<len(self.data)-1:
                        ref_array[3, i] = self.data[self.index+i][3]  # v
                        ref_array[4, i] = self.data[self.index+i][4]  # omega
                    else:
                        ref_array[3, i] = 0.0
                        ref_array[4, i] = 0.0
                
                # Publish reference trajectory
                self.ref_publisher.publish(self.ref_traj)
                
                # Get current state snapshot
                current_state = self.current_state.copy()
                
                # Run MPC update and get control inputs
                try:
                    self.MPC.set_reference_trajectory(ref_array)
                    u = self.MPC.update(current_state)
                    
                    # Create and publish control command
                    cmd_msg = Twist()
                    cmd_msg.linear.x = u[0]
                    cmd_msg.angular.z = u[1]
                    self.publisher_.publish(cmd_msg)
                    
                    self.get_logger().debug(f'Published cmd_vel: v={cmd_msg.linear.x:.2f}, ω={cmd_msg.angular.z:.2f}')
                    
                    # Publish predicted trajectory for visualization
                    pred_traj = self.MPC.get_predicted_trajectory()
                    self.pre_traj.poses = []
                    
                    for i in range(self.P_HOR+1):
                        pose_msg = PoseStamped()
                        pose_msg.header = Header(frame_id="map")
                        pose_msg.pose.position.x = pred_traj[0, i]
                        pose_msg.pose.position.y = pred_traj[1, i]
                        pose_msg.pose.position.z = 0.0
                        
                        # Convert yaw to quaternion
                        yaw = pred_traj[2, i]
                        quat = tf.quaternion_from_euler(0.0, 0.0, yaw)
                        pose_msg.pose.orientation.x = quat[0]
                        pose_msg.pose.orientation.y = quat[1]
                        pose_msg.pose.orientation.z = quat[2]
                        pose_msg.pose.orientation.w = quat[3]
                        
                        self.pre_traj.poses.append(pose_msg)
                    
                    self.prediction_publisher.publish(self.pre_traj)
                    self.index += 1
                    
                except Exception as e:
                    self.get_logger().error(f'MPC error: {str(e)}')
            else:
                # In pushing mode or finished
                distance = np.sqrt((self.current_state[0]-self.goal[0])**2 + 
                                  (self.current_state[1]-self.goal[1])**2)
                
                # Set pushing flag to true and publish
                self.Pushing_flag.data = True
                self.pushing_flag_pub.publish(self.Pushing_flag)
                
                cmd_msg = Twist()
                
                if distance < 0.35:
                    # Back away from object
                    cmd_msg.linear.x = -0.05
                    cmd_msg.angular.z = 0.0
                    self.publisher_.publish(cmd_msg)
                    self.get_logger().info(f'Backing away to safe distance: {distance:.2f}m')
                else:
                    # Stop moving when at safe distance
                    cmd_msg.linear.x = 0.0
                    cmd_msg.angular.z = 0.0
                    self.publisher_.publish(cmd_msg)
                    self.get_logger().info(f'Robot {self.namespace}: Finished pushing')
                    self.timer.cancel()
                    return
                    
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            # Stop robot for safety
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.publisher_.publish(cmd_msg)
            self.timer.cancel()
    
def main(args=None):
    print(f"[FollowController main START] Called with args: {args}", flush=True)
    rclpy.init(args=args)
    
    try:
        cmd_vel_publisher_node = CmdVelPublisher()
        current_node_name = cmd_vel_publisher_node.get_fully_qualified_name()
        print(f"[FollowController] Node created: {current_node_name}", flush=True)
        rclpy.spin(cmd_vel_publisher_node)
    except Exception as e:
        print(f"[FollowController] Error: {e}", flush=True)
    finally:
        if 'cmd_vel_publisher_node' in locals():
            cmd_vel_publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()