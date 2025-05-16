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


class PickupController(Node):
    def __init__(self):
        super().__init__('pickup_controller')
        
        # Declare parameters
        self.declare_parameter('namespace', 'tb0')
        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
        
        self.get_logger().info(f'Initializing pickup controller for robot: {self.namespace}')
        
        # Initialize flags
        self.pickup_flag = Bool()
        self.pickup_flag.data = False
        self.ready_flag = False
        self.pickup_done = False
        self.target_reached = False
        
        # Threading lock for state protection
        self.state_lock = threading.Lock()
        
        # Publishers with proper namespaces
        cmd_vel_topic = f'/{self.namespace}/cmd_vel'
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.get_logger().info(f'Publishing cmd_vel on: {cmd_vel_topic}')
        
        pred_path_topic = f'/{self.namespace}/pickup_pred_path'
        self.predicted_path_pub = self.create_publisher(Path, pred_path_topic, 10)
        
        target_path_topic = f'/{self.namespace}/pickup_target_path'
        self.target_path_pub = self.create_publisher(Path, target_path_topic, 10)
        
        # Publish PickUpDone flag for behavior tree to consume
        pickup_done_topic = f'/{self.namespace}/PickUpDone'
        self.pickup_done_pub = self.create_publisher(Bool, pickup_done_topic, 10)
        self.get_logger().info(f'Publishing PickUpDone on: {pickup_done_topic}')
        
        # Subscribers
        # Use the robot's odom topic directly 
        robot_pose_topic = f'/{self.namespace}/odom'
        self.robot_pose_sub = self.create_subscription(
            Odometry, robot_pose_topic, self.robot_pose_callback, 10)
        self.get_logger().info(f'Subscribing to robot pose on: {robot_pose_topic}')
        
        # The object pose (parcel) is shared across robots
        self.object_pose_sub = self.create_subscription(
            PoseStamped, '/parcel/pose', self.object_pose_callback, 10)
        
        # Subscribe to robot-specific Ready_flag from behavior tree
        ready_flag_topic = f'/{self.namespace}/Ready_flag'
        self.ready_flag_sub = self.create_subscription(
            Bool, ready_flag_topic, self.ready_flag_callback, 10)
        self.get_logger().info(f'Subscribing to Ready_flag on: {ready_flag_topic}')
        
        # Load robot-specific trajectory from workspace data directory
        json_file_path = f'/root/workspace/data/{self.namespace}_DiscreteTrajectory.json'
        self.get_logger().info(f'Loading trajectory from: {json_file_path}')
        
        try:
            with open(json_file_path, 'r') as json_file:
                self.data = json.load(json_file)['Trajectory']
                self.trajectory_data = self.data[::-1]  # Reverse the trajectory for pickup
                self.goal = self.data[0]
                self.get_logger().info(f'Loaded trajectory with {len(self.data)} points. Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
        except Exception as e:
            self.get_logger().error(f'Failed to load trajectory file: {e}')
            # Provide a minimal fallback trajectory
            self.data = [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.1, 0.0]]
            self.trajectory_data = self.data[::-1]
            self.goal = self.data[0]
        
        # Initialize state variables
        self.current_state = np.zeros(3)  # x, y, theta
        self.object_state = np.zeros(3)   # object x, y, theta
        
        # Calculate target pose - approach the object from a certain direction
        self.target_pose = np.zeros(3)    # x, y, theta
        self.target_pose[0] = self.goal[0] - 0.35 * np.cos(self.goal[2])
        self.target_pose[1] = self.goal[1] - 0.35 * np.sin(self.goal[2])
        self.target_pose[2] = self.goal[2]
        
        # Interpolate between target_pose and goal position
        # This creates a smoother approach to the final position
        num_interp_points = 10  # Number of interpolation points
        interp_points = []

        # Calculate the linear interpolation points
        for i in range(num_interp_points):
            alpha = i / (num_interp_points - 1)  # Interpolation factor (0 to 1)
            interp_x = self.target_pose[0] * alpha + self.goal[0] * (1 - alpha) 
            interp_y = self.target_pose[1] * alpha + self.goal[1] * (1 - alpha) 
            interp_theta = self.target_pose[2]  # Keep orientation constant for smoother approach
            
            v = 0.1  # Low linear velocity for final approach
            omega = 0.0  # No rotation
            
            interp_points.append([interp_x, interp_y, interp_theta, v, omega])

        # Add interpolation points to the beginning of trajectory_data
        # This means they'll be executed last in the reverse trajectory
        self.trajectory_data = self.trajectory_data + interp_points 
        
        # Create MPC controller
        self.mpc = MobileRobotMPC()
        self.prediction_horizon = self.mpc.N
        
        # Paths for visualization
        self.target_path = Path()
        self.target_path.header.frame_id = 'map'
        self.predicted_path = Path()
        self.predicted_path.header.frame_id = 'map'
        
        # Control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Update and publish target path once
        self.update_target_path()
        
        self.get_logger().info(f'Pickup controller initialized for {self.namespace}')
    
    def robot_pose_callback(self, msg):
        """Update position and orientation from odometry message"""
        with self.state_lock:
            # Position
            self.current_state[0] = msg.pose.position.x
            self.current_state[1] = msg.pose.position.y
            
            # Orientation (yaw)
            quat = [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]
            euler = tf.euler_from_quaternion(quat)
            self.current_state[2] = euler[2]  # yaw
    
    def object_pose_callback(self, msg):
        """Update object position and orientation"""
        self.object_state[0] = msg.pose.position.x
        self.object_state[1] = msg.pose.position.y
        
        quat = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        euler = tf.euler_from_quaternion(quat)
        self.object_state[2] = euler[2]
    
    def ready_flag_callback(self, msg):
        """Update ready flag from behavior tree"""
        old_flag = self.ready_flag
        self.ready_flag = msg.data
        if old_flag != self.ready_flag:
            self.get_logger().info(f'Ready_flag changed: {old_flag} -> {self.ready_flag}')
            if self.ready_flag:
                # Reset state when newly activated
                self.target_reached = False
                self.pickup_done = False
    
    def update_target_path(self):
        """Update the target path for visualization"""
        self.target_path.poses = []
        pose_msg = PoseStamped()
        pose_msg.header = Header(frame_id="map")
        pose_msg.pose.position.x = self.target_pose[0]
        pose_msg.pose.position.y = self.target_pose[1]
        pose_msg.pose.position.z = 0.0
        
        quat = tf.quaternion_from_euler(0.0, 0.0, self.target_pose[2])
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.target_path.poses.append(pose_msg)
        self.target_path_pub.publish(self.target_path)
    
    def control_loop(self):
        """Main control loop running at fixed frequency"""
        if not self.ready_flag:
            # Not activated by behavior tree yet
            return
        
        if self.pickup_done:
            # Already completed pickup
            return

        # Calculate distance to target
        distance_error = np.sqrt((self.current_state[0] - self.target_pose[0])**2 + 
                                (self.current_state[1] - self.target_pose[1])**2)
        
        if distance_error > 0.08:
            # Still navigating to target
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
                    pose_msg.header = Header(frame_id="map")
                    pose_msg.pose.position.x = predicted_traj[0, i]
                    pose_msg.pose.position.y = predicted_traj[1, i]
                    pose_msg.pose.position.z = 0.0
                    
                    quat = tf.quaternion_from_euler(0.0, 0.0, predicted_traj[2, i])
                    pose_msg.pose.orientation.x = quat[0]
                    pose_msg.pose.orientation.y = quat[1]
                    pose_msg.pose.orientation.z = quat[2]
                    pose_msg.pose.orientation.w = quat[3]
                    
                    self.predicted_path.poses.append(pose_msg)
                
                self.predicted_path_pub.publish(self.predicted_path)
                
            except Exception as e:
                self.get_logger().error(f'MPC error: {str(e)}')
                
                # Safety: stop robot if MPC fails
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_msg)
        else:
            # Target reached, perform pickup
            if not self.target_reached:
                self.get_logger().info(f'Target reached. Error: {distance_error:.4f}m')
                self.target_reached = True
            
            # Stop the robot
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)
            
            # Simulate a pickup procedure that takes some time
            if not self.pickup_done:
                # In a real system, you would trigger actuators here
                self.get_logger().info('Performing pickup operation...')
                
                # Mark pickup as done
                self.pickup_done = True
                
                # Publish PickUpDone flag for behavior tree to detect
                done_msg = Bool()
                done_msg.data = True
                self.pickup_done_pub.publish(done_msg)
                self.get_logger().info(f'Published PickUpDone=True to {self.namespace}/PickUpDone')


class MobileRobotMPC:
    def __init__(self):
        # MPC parameters
        self.N = 10           # Prediction horizon
        self.dt = 0.1         # Time step
        self.Q = np.diag([10, 10, 5])  # State weights (x, y, theta, v, omega)
        self.R = np.diag([0.1, 0.1])        # Control input weights
        self.F = np.diag([20, 20, 10]) # Terminal cost weights
        
        # Velocity constraints
        self.max_vel = 0.05     # m/s
        self.min_vel = -0.15    # m/s 
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
    print(f"[PickupController main START] Called with args: {args}", flush=True)
    rclpy.init(args=args)
    
    try:
        pickup_controller = PickupController()
        current_node_name = pickup_controller.get_fully_qualified_name()
        print(f"[PickupController] Node created: {current_node_name}", flush=True)
        rclpy.spin(pickup_controller)
    except Exception as e:
        print(f"[PickupController] Error: {e}", flush=True)
    finally:
        if 'pickup_controller' in locals():
            pickup_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
