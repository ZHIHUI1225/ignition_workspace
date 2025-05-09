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
    def __init__(self): # Removed namespace argument
        super().__init__('follow_controller_node') # Default node name
        # Declare and get the namespace parameter
        self.declare_parameter('namespace', 'tb_default') # Provide a default value
        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
                
        # self.declare_parameter('use_sim_time', True) # This was already correctly removed
        self.Pushing_flag=Bool()
        self.Pushing_flag.data=False
        self.Ready_flag=False
        self.state_lock = threading.Lock()
        self.publisher_ = self.create_publisher(Twist, 'pushing/cmd_vel', 10)
        self.prediction_publisher = self.create_publisher(Path, 'pushing/pred_path', 10)
        self.ref_publisher = self.create_publisher(Path, 'pushing/ref_path', 10)
        self.pushing_flag_pub=self.create_publisher(Bool, 'Pushing_flag', 10)
        # Add a subscriber for robot pose
        self.robot_pose_sub = self.create_subscription(
            Odometry, 'pushing/robot_pose', self.robot_pose_callback, 10)
        # self.robot_twist_sub = self.create_subscription(
        #     TwistStamped, 'pushing/robot_twist', self.robot_twist_callback, 10)
        self.parcel_pose_sub = self.create_subscription(
            PoseStamped, 'pushing/object_pose', self.parcel_pose_callback, 10)
        # self.Pushing_flag_sub = self.create_subscription(
        #     Bool, 'Pushing_flag', self.Pushing_flag_callback, 10)
        self.Ready_flag_sub = self.create_subscription(
            Bool, 'Ready_flag', self.Ready_flag_callback, 10)
        
        # Use self.namespace which is now a class member
        json_file_path = f'/root/workspace/data/{self.namespace}_DiscreteTrajectory.json'
        with open(json_file_path, 'r') as json_file:
            self.data = json.load(json_file)['Trajectory']
            self.goal=self.data[-1]
        # Initialize current state (x, y, theta, v, omega)
        self.current_state = np.zeros(5)
        self.parcel_state = np.zeros(3)
        self.last_time = self.get_clock().now()
        self.index = 0
        timer_period = 0.1# seconds
        self.ref_traj= Path()
        self.ref_traj.header.frame_id = 'map'
        self.ref_traj.header.stamp = self.get_clock().now().to_msg()
        self.pre_traj= Path()
        self.pre_traj.header.frame_id = 'map'
        self.pre_traj.header.stamp = self.get_clock().now().to_msg()
        self.MPC=MobileRobotMPC()
        self.P_HOR = self.MPC.N
         # Timing control
        self.timer = self.create_timer(timer_period, self.control_loop)
        
        
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
        # self.get_logger().info(f'Parcel pose: {self.parcel_state[0]:.2f}, {self.parcel_state[1]:.2f}, {self.parcel_state[2]:.2f}')

    # def Pushing_flag_callback(self, msg):
    #     self.Pushing_flag=msg.data
        # self.get_logger().info(f'Pushing_flag: {self.Pushing_flag}')
    def Ready_flag_callback(self, msg):
        old_flag = self.Ready_flag
        self.Ready_flag = msg.data
        if old_flag != self.Ready_flag:
            self.get_logger().info(f'Ready_flag changed: {old_flag} -> {self.Ready_flag}')
    def robot_pose_callback(self, msg):
        """Update position and orientation from pose message"""
        # print(f"robot_pose_callback: {msg}") # Optional: Keep or remove the raw message print
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
            # Example using more decimal places:
            # self.get_logger().info(f'Robot pose: {self.current_state[0]:.2e}, {self.current_state[1]:.2e}, {self.current_state[2]:.2e}')

    def control_loop(self):
        self.get_logger().info('Entering control_loop')
        if self.Ready_flag is False:
            self.get_logger().info('Ready_flag is False, skipping control loop')
            return
        try:
            # 1. Update reference trajectory
            distance_error=np.sqrt((self.current_state[0]-self.goal[0])**2+(self.current_state[1]-self.goal[1])**2)
            object_distance=np.sqrt((self.goal[0]-self.parcel_state[0])**2+(self.goal[1]-self.parcel_state[1])**2)
            # MPC controller works and publishes cmd_vel when:
            # 1. We still have trajectory points to follow (index < len(data))
            # 2. We're not close enough to the goal (distance_error > 0.1)
            # 3. We're not in pushing mode (Pushing_flag is False)
            if self.index < len(self.data) and object_distance > 0.05 and self.Pushing_flag.data is False:  
                self.Pushing_flag.data=False
                self.pushing_flag_pub.publish(self.Pushing_flag)
                print(f'index: {self.index}, distance_error: {distance_error:.2f}')
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
                        ref_array[3, i] = 0
                        ref_array[4, i] = 0
                
                # self.MPC.set_reference_trajectory(ref_array)
                self.ref_publisher.publish(self.ref_traj)

            
                # 2. Get current state snapshot
                # with self.state_lock:
                # if self.ref_traj!=None:
                current_state = self.current_state.copy()
                
                # 3. MPC Update - this is when the controller calculates optimal control inputs
                try:
                    self.MPC.set_reference_trajectory(ref_array)
                    u = self.MPC.update(current_state)
                    
                    # 4. Publish control command based on MPC output
                    cmd_msg = Twist()
                    cmd_msg.linear.x = u[0]
                    cmd_msg.angular.z = u[1]
                    self.publisher_.publish(cmd_msg)
                    print(f'Publishing cmd_vel: linear.x={cmd_msg.linear.x:.2f}, angular.z={cmd_msg.angular.z:.2f}')
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
                        pose_msg.pose.orientation.x = quat[0]
                        pose_msg.pose.orientation.y = quat[1]
                        pose_msg.pose.orientation.z = quat[2]
                        pose_msg.pose.orientation.w = quat[3]
                        self.pre_traj.poses.append(pose_msg)
                    self.prediction_publisher.publish(self.pre_traj)
                    self.index += 1
                    
                except Exception as e:
                    self.get_logger().error(f'Control error: {str(e)}')
            else:
                distance=np.sqrt((self.current_state[0]-self.goal[0])**2+(self.current_state[1]-self.goal[1])**2)
                cmd_msg = Twist()
                self.Pushing_flag.data=True
                if distance<0.3 :
                    cmd_msg.linear.x =-0.05
                    cmd_msg.angular.z = 0.0
                    self.publisher_.publish(cmd_msg)
                    self.get_logger().info(f'back to safe distance: {distance:.2f}m')
                    self.pushing_flag_pub.publish(self.Pushing_flag)
                else:
                    self.pushing_flag_pub.publish(self.Pushing_flag)
                    cmd_msg.linear.x = 0.0
                    cmd_msg.angular.z = 0.0
                    self.publisher_.publish(cmd_msg)
                    self.get_logger().info('Finished pushing.')
                    self.timer.cancel()
                    return
            print(f"Current state: {self.current_state}")
            print(f"Goal state: {self.goal}")
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