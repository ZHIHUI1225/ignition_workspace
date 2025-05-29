#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger, SetBool
import math
import threading
import re
import numpy as np
import casadi as ca
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
import tf_transformations as tf
from nav_msgs.msg import Path, Odometry

class MobileRobotMPC:
    def __init__(self):
        # MPC parameters
        self.N = 10         # Extended prediction horizon for smoother approach
        self.dt = 0.1        # Time step
        self.wx = 2.0        # Increased position error weight for better position convergence
        self.wtheta = 1.5    # Increased orientation error weight for better alignment
        self.wu = 0.08       # Slightly reduced control effort weight for more responsive control
        
        # Control constraints - further reduced max velocity for even slower approach
        self.v_max = 0.02     # m/s (reduced from 0.025)
        self.w_max = 0.6      # rad/s (reduced from 0.8)
        
        # State and control dimensions
        self.nx = 3          # [x, y, theta]
        self.nu = 2          # [v, w]
        
        # Initialize CasADi optimizer
        self.setup_optimizer()
        
    def setup_optimizer(self):
        # Define symbolic variables
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N+1)
        self.U = self.opti.variable(self.nu, self.N)
        
        # Parameters (initial state and reference)
        self.x0 = self.opti.parameter(self.nx)
        self.x_ref = self.opti.parameter(self.nx)
        
        # Cost function
        cost = 0
        for k in range(self.N):
            # Tracking cost with increasing weights as we approach the end
            # Stronger progression factor for more aggressive convergence
            progress_factor = 1.0 + 2.0 * (k + 1) / self.N  # Increases from 1.0 to 3.0
            
            # Position cost - higher weight for xy position tracking
            pos_error = self.X[:2, k] - self.x_ref[:2]
            cost += progress_factor * self.wx * ca.sumsqr(pos_error)
            
            # Orientation cost with angle normalization to handle wraparound
            theta_error = self.X[2, k] - self.x_ref[2]
            # Normalize angle difference to [-pi, pi]
            theta_error = ca.fmod(theta_error + ca.pi, 2*ca.pi) - ca.pi
            cost += progress_factor * self.wtheta * theta_error**2
            
            # Special emphasis on final portion of trajectory
            if k >= self.N - 5:  # Last 5 steps
                # Extra emphasis on final approach
                cost += 1.5 * self.wx * ca.sumsqr(pos_error)
                cost += 2.0 * self.wtheta * theta_error**2
            
            # Control effort cost with smoother transitions
            if k > 0:
                # Penalize control changes for smoother motion
                control_change = self.U[:, k] - self.U[:, k-1]
                cost += 0.1 * ca.sumsqr(control_change)
            
            # Base control effort penalty
            cost += self.wu * ca.sumsqr(self.U[:, k])
        
        # Terminal cost - much stronger to ensure convergence at endpoint
        terminal_pos_error = self.X[:2, self.N] - self.x_ref[:2]
        cost += 10.0 * self.wx * ca.sumsqr(terminal_pos_error)
        
        # Terminal orientation with normalization for angle wraparound
        terminal_theta_error = self.X[2, self.N] - self.x_ref[2]
        terminal_theta_error = ca.fmod(terminal_theta_error + ca.pi, 2*ca.pi) - ca.pi
        cost += 30.0 * self.wtheta * terminal_theta_error**2
        self.opti.minimize(cost)
        # Dynamics constraints
        for k in range(self.N):
            x_next = self.X[:, k] + self.robot_model(self.X[:, k], self.U[:, k]) * self.dt
            self.opti.subject_to(self.X[:, k+1] == x_next)
        
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # Control constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max, self.U[0, :], self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.w_max, self.U[1, :], self.w_max))
        
        # Strict terminal constraints to ensure convergence and smooth stopping
        # Terminal velocity constraints - must approach zero at the end
        self.opti.subject_to(self.U[0, -1] <= 0.0005)  # Final linear velocity virtually zero
        self.opti.subject_to(self.U[0, -1] >= 0.0)     # No negative velocity at end
        self.opti.subject_to(self.U[1, -1] <= 0.0005)  # Final angular velocity virtually zero
        self.opti.subject_to(self.U[1, -1] >= -0.0005) # Final angular velocity virtually zero
        
        # Smooth deceleration in last few steps
        for k in range(self.N-3, self.N):
            # Progressive velocity reduction for final steps
            max_vel_factor = (self.N - k) / 4.0  # Ranges from 0.75 to 0.25
            self.opti.subject_to(self.U[0, k] <= self.v_max * max_vel_factor)
        
        # Solver settings with improved convergence parameters
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.tol': 1e-5,           # Even tighter tolerance
            'ipopt.acceptable_tol': 1e-4, # More precise solution
            'ipopt.max_iter': 200,       # More iterations allowed
            'ipopt.warm_start_init_point': 'yes' # Use warm starting for stability
        }
        self.opti.solver('ipopt', opts)
        
    def robot_model(self, x, u):
        # Differential drive kinematics
        dx = ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1]
        )
        return dx
        
    def update_control(self, current_state, target_state):
        # Check how close we are to the target
        dist_to_target = np.sqrt((current_state[0] - target_state[0])**2 + 
                                (current_state[1] - target_state[1])**2)
        
        # Check orientation alignment
        angle_diff = abs((current_state[2] - target_state[2] + np.pi) % (2 * np.pi) - np.pi)
        
        # If we're very close to the target and well-aligned, stop completely
        if dist_to_target < 0.015 and angle_diff < 0.05:  # 1.5cm and ~3 degrees
            return np.array([0.0, 0.0])  # Stop completely
            
        # If we're close but not perfectly aligned, prioritize orientation
        elif dist_to_target < 0.03 and angle_diff > 0.05:
            # Just rotate to align with target, very slowly
            return np.array([0.0, 0.1 * np.sign(target_state[2] - current_state[2])])
        
        # Set initial state and reference
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.x_ref, target_state)
        
        # Solve optimization problem
        try:
            sol = self.opti.solve()
            x_opt = sol.value(self.X)
            u_opt = sol.value(self.U)
            return u_opt[:, 0]  # Return first control input
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None
        
class ProximityChecker(Node):
    def __init__(self): # Removed namespace argument
        super().__init__('state_switch_node') # Default node name
        # Declare and get namespace parameter
        self.declare_parameter('namespace', 'tb_default') # Provide a default value
        self.namespace = self.get_parameter('namespace').get_parameter_value().string_value
        
        # Extract number from namespace using regex
        self.namespace_number = self.extract_namespace_number(self.namespace)
        self.get_logger().info(f'Operating in namespace: {self.namespace}, Extracted number: {self.namespace_number}')
        
        # Parameters
        self.declare_parameter('distance_threshold', 0.08)  
        self.threshold = self.get_parameter('distance_threshold').value
        
        # Each robot has its own parcel index, starting from 0
        self.current_parcel_index = 0
        self.approaching_target = False
        # Subscribers
        self.parcel_pose = None
        self.target = None #the relay point i+1
        self.start= None # the relay point i 
        self.lock = threading.Lock()
        self.last_flag = False
        self.last_flag_sub=self.create_subscription(
            Bool,
            'Last_Flag',  # Corrected case to match launch file remapping
            self.last_flag_callback,
            10)
            
        
        # Dynamic parcel topic subscription based on the current index
        self.parcel_sub = None
        self.update_parcel_subscription()
        
        # nub_relay = self.namespace_number+1
        self.relay_sub = self.create_subscription(
            PoseStamped,
            f'Target/pose',
            self.relay_callback,
            10)
        self.RP_sub = self.create_subscription(
            PoseStamped,
            f'Start/pose',
            self.start_callback,
            10)
        self.last_robot = None # the turtlebot i-1
        self.last_robot_pose = self.create_subscription(
            Odometry,
            'Last/pose',
            self.last_robot_callback,
            10)
        
        self.robot= None # the turtlebot i
        self.robot_pose=self.create_subscription(
            Odometry,
            'Robot/pose',
            self.robot_callback, # Added the callback
            10 
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)      # Command velocity publisher
        self.parcel_index_pub = self.create_publisher(Int32, f'/{self.namespace}/current_parcel_index', 10)

        self.index_sub = self.create_subscription(
            Int32,
            f'/{self.namespace}/current_parcel_index',
            self.current_index_callback,
            10
        )
        self.get_logger().info(f'Subscribed to parcel index topic: /{self.namespace}/current_parcel_index')
        # Service clients for activating controllers
        self.start_pushing_client = self.create_client(Trigger, f'/{self.namespace}/start_pushing')
        self.start_picking_client = self.create_client(Trigger, f'/{self.namespace}/start_picking')
        
        # Service server to receive pushing status from FollowController
        self.pushing_finish_server = self.create_service(
            Trigger, 
            f'/{self.namespace}/pushing_finish', 
            self.pushing_finish_trigger_callback 
        )
        self.get_logger().info(f'Service /{self.namespace}/pushing_finish (Trigger) server created.')
        
        # Initialize flags for tracking state
        self.pushing_ready_flag = False
        self.picking_ready_flag = False
        
        # REMOVED: self.current_pushing_status_from_follow = False 
        
        # States to track the workflow
        self.pushing_complete = False # Will be set True by trigger
        self.pushing_completion_triggered_for_current_parcel = False
        
        # MPC controller
        self.mpc = MobileRobotMPC()
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        # Timer for periodic checking
        self.timer = self.create_timer(0.1, self.check_proximity)  # 10Hz
        self.get_logger().info(f"State_switch node initialized for namespace: {self.namespace}") # Log namespace
    
    
    def update_parcel_subscription(self):
        """Update subscription to the correct parcel topic based on current index"""
        # If we already have a subscription, destroy it
        if self.parcel_sub is not None:
            self.destroy_subscription(self.parcel_sub)
            
        # Create new subscription with updated topic name
        self.parcel_sub = self.create_subscription(
            PoseStamped,
            f'/parcel{self.current_parcel_index}/pose',
            self.parcel_callback,
            10
        )
        self.get_logger().info(f'Now tracking parcel{self.current_parcel_index}')
    
        
    def last_robot_callback(self, msg):
        with self.lock:
            self.last_robot = msg.pose
            # self.get_logger().debug('Updated last robot pose')
    def robot_callback(self, msg):
        with self.lock:
            self.robot = msg.pose
            # self.get_logger().debug('Updated robot pose')

    def start_callback(self, msg):
        with self.lock:
            self.start = msg.pose
            # self.get_logger().debug('Updated start RP pose')
    def last_flag_callback(self,msg):
        with self.lock:
            self.last_flag = msg.data
            # self.get_logger().info(f'Last flag: {self.last_flag}')
            # if self.namespace_number == 0:
            #     self.last_flag = True


    def extract_namespace_number(self, namespace):
        """Extract numerical index from namespace string using regex"""
        match = re.match(r"^tb(\d+)$", namespace)
        if match:
            return int(match.group(1))
        self.get_logger().error(f"Invalid namespace format: {namespace}. Using default 0")
        return 0
    
    def parcel_callback(self, msg):
        with self.lock:
            self.parcel_pose = msg.pose
            # self.get_logger().debug(f'Updated parcel{self.current_parcel_index} pose')

    def relay_callback(self, msg):
        with self.lock:
            self.target = msg.pose
            # self.get_logger().debug('Updated target relay point pose')

    def calculate_distance(self, pose1, pose2):
        if hasattr(pose1, 'pose'):
            pose1 = pose1.pose
        if hasattr(pose2, 'pose'):
            pose2 = pose2.pose
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx**2 + dy**2)

    def quaternion_to_yaw(self, quat):
        # Convert quaternion to yaw angle
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        quat = [x, y, z, w]
        euler = tf.euler_from_quaternion(quat)
        return euler[2]

    def pushing_finish_trigger_callback(self, request: Trigger.Request, response: Trigger.Response): # Renamed callback
        """Callback for the Trigger service indicating FollowController has completed pushing."""
        with self.lock:
            if self.pushing_completion_triggered_for_current_parcel:
                self.get_logger().info(f"Pushing completion trigger (pushing_finish) received for {self.namespace}, but already processed for current parcel.")
                response.success = False # Indicate no state change / already done
                response.message = "Pushing completion already processed for this parcel cycle."
                return response

            self.get_logger().info(f"Pushing completion trigger (pushing_finish) received for {self.namespace}. Updating state.")
            self.pushing_complete = True
            self.pushing_completion_triggered_for_current_parcel = True

            response.success = True
            response.message = "Pushing status successfully updated to complete via trigger (pushing_finish)."
        return response
    
    def get_direction(self, robot_theta, parcel_theta):
        # 规范化输入角度到[-π, π]
        def normalize(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        
        robot_theta = normalize(robot_theta)
        parcel_theta = normalize(parcel_theta)
        
        # 生成候选角度并规范化
        candidates = [
            parcel_theta,
            normalize(parcel_theta + np.pi/2),  # 向右偏转90度
            normalize(parcel_theta - np.pi/2),   # 向左偏转90度
            normalize(parcel_theta + np.pi),     # 180度
        ]
        
        # 计算最小圆周角差
        diffs = [abs(normalize(c - robot_theta)) for c in candidates]
        
        index_min = np.argmin(diffs)

        return candidates[index_min]
    
    def check_proximity(self):
        with self.lock:
            # Ensure all necessary data is available
            if self.parcel_pose is None or self.robot is None or self.target is None or self.start is None:
                # self.get_logger().info("Waiting for pose data...")
                return

            # Instead of publishing to a topic, we'll use service calls when needed
            # Calculate distances
            distance_parcel_to_target = self.calculate_distance(self.parcel_pose, self.target)
            
            if self.last_robot is None:
                self.get_logger().warn('Waiting for last robot poses...')
                return
            if self.last_flag is None:
                self.get_logger().warn('Waiting for last flag...')
                return
                
            # Calculate distances between key elements
            dis_last_parcel = self.calculate_distance(self.parcel_pose, self.last_robot)
            dis_robot_parcel = self.calculate_distance(self.parcel_pose, self.robot)
            
            # Check if parcel is in relay point range and last robot is not
            parcel_in_relay = distance_parcel_to_target < self.threshold
            last_robot_away = dis_last_parcel > 0.2
            
            # Calculate distance to target position for approaching mode
            if hasattr(self, 'target_state'):
                target_pos = np.array([self.target_state[0], self.target_state[1]])
                current_pos = np.array([self.robot.pose.position.x, self.robot.pose.position.y])
                distance_to_target = np.linalg.norm(target_pos - current_pos)
                distance_parcel_start= self.calculate_distance(self.parcel_pose, self.start)
                robot_start= distance_parcel_start <0.3
            else:
                distance_to_target = float('inf')
            if self.namespace_number == 0:
                last_robot_away = True
                self.last_flag = True
                # self.approaching_target = True

            
            if self.last_flag is True and last_robot_away and not self.pushing_ready_flag and not self.pushing_complete and robot_start:
                # Condition 1: Parcel is in relay point range and last robot isn't
                if dis_robot_parcel > 0.25:  # Robot needs to approach
                    print(f"parcel_index: {self.current_parcel_index}, parcel_y: {self.parcel_pose.position.y}, robot_y: {self.robot.pose.position.y}, dist: {dis_robot_parcel}m") # Corrected parcel_pose access
                    self.approaching_target = True
                    # Compute control commands using MPC
                    self.current_state = np.array([
                        self.robot.pose.position.x, 
                        self.robot.pose.position.y, 
                        self.quaternion_to_yaw(self.robot.pose.orientation)
                    ])
                    
                    self.target_state = np.array([
                        self.parcel_pose.position.x, 
                        self.parcel_pose.position.y, 
                        self.quaternion_to_yaw(self.parcel_pose.orientation)
                    ])
                    
                    # Get optimal direction and offset
                    self.target_state[2] = self.get_direction(
                        self.current_state[2],
                        self.target_state[2]
                    )
                    self.target_state[0] = self.target_state[0] - 0.12 * math.cos(self.target_state[2])
                    self.target_state[1] = self.target_state[1] - 0.12 * math.sin(self.target_state[2])
                    
                    # Generate and apply control
                    u = self.mpc.update_control(self.current_state, self.target_state)
                    # print(f"Current state: {self.current_state}, Target state: {self.target_state}")
                    if u is not None:
                        cmd = Twist()
                        cmd.linear.x = float(u[0])
                        cmd.angular.z = float(u[1])
                        self.cmd_pub.publish(cmd)
                        # self.get_logger().info(f'Approaching target. Distance: {distance_to_target:.2f}m')
                else:
                    # Robot reached target position, ready to start pushing
                    # print(f"approaching_target: {self.approaching_target}, distance_to_target: {distance_to_target}")
                    if self.approaching_target:
                        self.approaching_target = False
                        self.get_logger().info('Target position reached, starting pushing task')
                        
                        # Stop the robot
                        cmd = Twist()
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
                        self.cmd_pub.publish(cmd)
                    
                    # Signal to start pushing task by calling the service
                    self.pushing_ready_flag = True
                    self.call_start_pushing_service()
            
            # When pushing is complete, wait for robot to back away, then trigger pickup
            elif self.pushing_complete:
                # Pushing task is complete, robot should be backing away
                self.pushing_ready_flag = False 
                self.pushing_complete = True  
                
            # After a suitable delay, enable pickup
            if self.pushing_complete and not self.picking_ready_flag and not hasattr(self, 'pickup_active_service'):  
                self.get_logger().info('Robot at safe distance, activating pickup controller')
                self.picking_ready_flag = True
                self.pickup_active_service = True
                self.call_start_picking_service()


    def call_start_pushing_service(self):
        """Call the start_pushing service to activate the pushing controller"""
        request = Trigger.Request()
        future = self.start_pushing_client.call_async(request)
        future.add_done_callback(self.pushing_service_callback)
        
    def pushing_service_callback(self, future):
        """Callback for the start_pushing service response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully activated pushing controller: {response.message}')
            
            else:
                self.get_logger().warn(f'Failed to activate pushing controller: {response.message}')
                # If the service call fails, reset the flag
                self.pushing_ready_flag = False
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')
            # If the service call throws an exception, reset the flag
            self.pushing_ready_flag = False
    
    def call_start_picking_service(self):
        """Call the start_picking service to activate the pickup controller"""
        request = Trigger.Request()
        future = self.start_picking_client.call_async(request)
        future.add_done_callback(self.picking_service_callback)
    
    def picking_service_callback(self, future):
        """Callback for the start_picking service response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully activated pickup controller: {response.message}')
                # self.pushing_complete = False  # Reset pushing complete flag
            else:
                self.get_logger().warn(f'Failed to activate pickup controller: {response.message}')
                # If the service call fails, reset the flag
                self.picking_ready_flag = False

        #         self.get_logger().info('Reset pickup_active_service flag')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')
        #     # If the service call throws an exception, reset the flags
            self.picking_ready_flag = False

    def current_index_callback(self, msg):
        """Handle updates to the current parcel index and reset state flags"""
        with self.lock: # Ensure thread safety for flag updates
            old_index = self.current_parcel_index
            if msg.data != old_index:
                self.current_parcel_index = msg.data
                self.get_logger().info(f"Parcel index updated from {old_index} to {self.current_parcel_index} in {self.namespace}. Resetting completion flags and parcel pose.")
                self.parcel_pose = None # Explicitly clear parcel_pose
                self.pushing_complete = False
                self.pushing_completion_triggered_for_current_parcel = False
                self.picking_ready_flag = False 
                self.pushing_ready_flag = False 
                self.last_flag = False
                # Add any other flags that need to be reset for a new parcel workflow
                self.update_parcel_subscription() 


def main(args=None):
    rclpy.init(args=args)
    # Directly create and spin the ProximityChecker node
    proximity_checker_node = ProximityChecker()
    # The namespace parameter will be set by the launch file.
    
    try:
        rclpy.spin(proximity_checker_node)
    except KeyboardInterrupt:
        proximity_checker_node.get_logger().info('Shutting down...')
    finally:
        proximity_checker_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()