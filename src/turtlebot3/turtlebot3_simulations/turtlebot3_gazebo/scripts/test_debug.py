#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
import math
import threading
import re
import numpy as np
import casadi as ca
import tf_transformations as tf
from nav_msgs.msg import Path, Odometry

class MobileRobotMPC:
    def __init__(self):
        # MPC parameters
        self.N = 5          # Prediction horizon
        self.dt = 0.1        # Time step
        self.wx = 1.0        # Position error weight
        self.wtheta = 0.5    # Orientation error weight
        self.wu = 0.1        # Control effort weight
        
        # Control constraints
        self.v_max = 0.05     # m/s
        self.w_max = 0.5     # rad/s
        
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
            # Tracking cost
            cost += self.wx * ca.sumsqr(self.X[:2, k] - self.x_ref[:2])
            cost += self.wtheta * (self.X[2, k] - self.x_ref[2])**2
            # Control effort cost
            cost += self.wu * ca.sumsqr(self.U[:, k])
        
        # Terminal cost
        cost += 1 * self.wx * ca.sumsqr(self.X[:2, self.N] - self.x_ref[:2])
        cost += 1 * self.wtheta * (self.X[2, self.N] - self.x_ref[2])**2
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
        
        # Solver settings
        opts = {'ipopt.print_level': 0, 'print_time': 0}
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
        # Set initial state and reference
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.x_ref, target_state)
        
        # Solve optimization problem
        try:
            sol = self.opti.solve()
            x_opt = sol.value(self.X)
            u_opt = sol.value(self.U)
            return u_opt[:, 0]  # Return first control input
        except:
            return None
        
class ProximityChecker(Node):
    def __init__(self,namespace):
        super().__init__('Pushing_checker')
        # Declare and get namespace parameter
        # self.declare_parameter('namespace', 'tb0')
        # namespace = self.get_parameter('namespace').value
        
        # Extract number from namespace using regex
        self.namespace_number = self.extract_namespace_number(namespace)
        self.get_logger().info(f'Operating in namespace: {namespace}, Extracted number: {self.namespace_number}')
        
        # Parameters
        self.declare_parameter('distance_threshold', 0.08)  
        self.threshold = self.get_parameter('distance_threshold').value
        
        # Subscribers
        self.parcel_pose = None
        self.target = None #the relay point i+1
        self.start= None # the relay point i 
        self.lock = threading.Lock()
        self.last_flag = False
        self.last_flag_sub=self.create_subscription(
            Bool,
            'Last_flag',
            self.last_flag_callback,
            10)
        self.pushing_flag_pub=self.create_subscription(
            Bool,
            'Pushing_flag',
            self.pushing_flag_callback,
            10)
        self.parcel_sub = self.create_subscription(
            PoseStamped,
            f'parcel0/pose',
            self.parcel_callback,
            10)
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
            self.robot_callback,
            10)
        # Publisher
        # self.flag_pub = self.create_publisher(Bool, 'Pushing_flag', 10) # the parcel arrived at the relay point
        self.ready_pub = self.create_publisher(Bool, 'Ready_flag', 10)  # the robot is ready to push the parcel
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        if namespace == 'tb0':
            self.ready_flag = Bool()
            self.ready_flag.data = True
            self.ready_pub.publish(self.ready_flag)
        else:
            self.ready_flag = Bool()
            self.ready_flag.data = False
            self.ready_pub.publish(self.ready_flag)
        self.pushing_flag = False
        # self.flag_pub.publish(self.pushing_flag)
        # MPC controller
        self.mpc = MobileRobotMPC()
        self.current_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.target_state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        # Timer for periodic checking
        self.timer = self.create_timer(0.1, self.check_proximity)  # 10Hz
    def pushing_flag_callback(self,msg):
        with self.lock:
            self.pushing_flag = msg.data
            self.get_logger().debug('Updated pushing flag')
        
    def last_robot_callback(self, msg):
        with self.lock:
            self.last_robot = msg.pose
            self.get_logger().debug('Updated last robot pose')
    def robot_callback(self, msg):
        with self.lock:
            self.robot = msg.pose
            self.get_logger().debug('Updated robot pose')

    def start_callback(self, msg):
        with self.lock:
            self.start = msg.pose
            self.get_logger().debug('Updated start RP pose')
    def last_flag_callback(self,msg):
        with self.lock:
            self.last_flag = msg.data
            if self.namespace_number == 0:
                self.last_flag = True
            self.get_logger().debug('Updated last flag')

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
            self.get_logger().debug('Updated parcel pose')

    def relay_callback(self, msg):
        with self.lock:
            self.target = msg.pose
            self.get_logger().debug('Updated target relay point pose')

    def calculate_distance(self, pose1, pose2):
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

    
    def get_direction(self,robot_theta,parcel_theta):
        angle = [parcel_theta,parcel_theta+np.pi/2,parcel_theta-np.pi/2]
        angle_diff = [abs(robot_theta-angle[0]),abs(robot_theta-angle[1]),abs(robot_theta-angle[2])]
        index_min=np.argmin(angle_diff)
        return angle[index_min]
    
    def check_proximity(self):
        with self.lock:
            if self.parcel_pose is None:
                self.get_logger().warn('Waiting for parcel poses...')
                return
            if self.target is None:
                self.get_logger().warn('Waiting for target poses...')
                return
            if self.start is None:
                self.get_logger().warn('Waiting for start poses...')
                return
            if self.namespace_number == 0:
                self.ready_flag.data = True
                self.ready_pub.publish(self.ready_flag)
    
            distance = self.calculate_distance(self.parcel_pose, self.target )
            # flag = Bool()
            # flag.data = distance < self.threshold
            # self.flag_pub.publish(flag)

            # if flag.data:
                # self.get_logger().info(f'Flag: {flag.data}')
                # self.get_logger().info(f'Distance: {distance:.2f}m, Flag: {flag.data}')
            if self.last_robot is None:
                self.get_logger().warn('Waiting for last robot poses...')
                return
            if self.last_flag is None:
                self.get_logger().warn('Waiting for last flag...')
                return
            dis_last_parcel = self.calculate_distance(self.parcel_pose, self.last_robot)
            dis_robot_parcel = self.calculate_distance(self.parcel_pose, self.robot)
            if   self.namespace_number!=0 and self.last_flag is True and dis_last_parcel > 0.25 and self.pushing_flag is False:
                if dis_robot_parcel > 0.14: # the last robot has pushed the parcel to the relay point
                    # Compute control using MPC
                    self.current_state = np.array([self.robot.position.x, self.robot.position.y, self.quaternion_to_yaw(self.robot.orientation)])
                    self.target_state = np.array([self.parcel_pose.position.x, self.parcel_pose.position.y, self.quaternion_to_yaw(self.parcel_pose.orientation)])
                    self.target_state[2]=self.get_direction(self.current_state[2],self.target_state[2])
                    self.target_state[0]=self.target_state[0]-0.12*math.cos(self.target_state[2])
                    self.target_state[1]=self.target_state[1]-0.12*math.sin(self.target_state[2])
                    u = self.mpc.update_control(self.current_state, self.target_state)
                    if u is not None:
                        # Publish control command
                        cmd = Twist()
                        cmd.linear.x = float(u[0])
                        cmd.angular.z = float(u[1])
                        self.cmd_pub.publish(cmd)
                else:
                    self.ready_flag.data = True
                    self.ready_pub.publish(self.ready_flag)
                    # self.pushing_flag.data = False # reset to false
                    # self.flag_pub.publish(self.pushing_flag)



                

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('Publish_flag_node')
    node.declare_parameter('namespace', 'tb0')  # Declare the namespace parameter with a default value

    namespace = node.get_parameter('namespace').get_parameter_value().string_value 
    node = ProximityChecker(namespace)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()