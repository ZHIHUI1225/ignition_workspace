#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
import py_trees
from nav2_msgs.action import NavigateToPose
import py_trees_ros
from action_msgs.msg import GoalStatus
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Path, Odometry


class MoveToGoal(py_trees.behaviour.Behaviour):

    def __init__(self, name, robot_id):
        super().__init__(name)
        self.robot_id = robot_id
        self.nav_client = None
        self.goal_handle = None

    def setup(self, **kwargs):
        # 获取ROS2节点句柄（参考网页2的节点注册方法）
        self.node = kwargs['node']
        self.nav_client = ActionClient(self.node, NavigateToPose, f'/robot_{self.robot_id}/navigate_to_pose')

    def update(self):
        # 从黑板获取目标位姿（参考网页5的黑板通信机制）
        target_pose = self.blackboard.get(f'robot_{self.robot_id}_target_pose')
        
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            return py_trees.common.Status.RUNNING
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        self._send_goal_future = self.nav_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self._goal_response_callback)
        return py_trees.common.Status.RUNNING

    def _goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.node.get_logger().error('导航目标被拒绝')
            return
        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result().result
        if hasattr(result, 'result_code') and result.result_code == NavigateToPose.Result.SUCCESS: # Retain original logic but add check
            self.blackboard.set(f'parcel_at_relay_{self.robot_id}', True)
        elif hasattr(future.result(), 'status') and future.result().status == GoalStatus.STATUS_SUCCEEDED: # Fallback to status
             self.blackboard.set(f'parcel_at_relay_{self.robot_id}', True)
        # else:
            # self.node.get_logger().warn(f"Goal for robot {self.robot_id} did not succeed based on result_code or status.")

# --- custom checker class ---
class CheckBlackboardVariable(py_trees.behaviour.Behaviour):
    """Check that a Blackboard variable equals an expected value."""
    def __init__(self, name, key, expected_value):
        super().__init__(name)
        self.key = key
        self.expected_value = expected_value
        self.blackboard = py_trees.blackboard.Blackboard()

    def update(self):
        if self.blackboard.get(self.key) == self.expected_value:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

# --- Follow Control Behavior ---
class FollowControlBehavior(py_trees.behaviour.Behaviour):
    """Calls Follow_controller.py node to control robot following behavior"""
    def __init__(self, name, robot_id):
        super().__init__(name)
        self.robot_id = robot_id
        self.publisher = None
        self.ready_flag_publisher = None
        self.blackboard = py_trees.blackboard.Blackboard()  # Initialize blackboard

    def setup(self, **kwargs):
        self.node = kwargs['node']
        # Publisher for Ready_flag - use the original topic without namespace
        self.ready_flag_publisher = self.node.create_publisher(
            Bool, 'Ready_flag', 10
        )
        self.node.get_logger().info(f'Publishing to Ready_flag topic to trigger Follow_controller')
        
        # Subscribe to Pushing_flag to detect when following is complete
        self.pushing_flag_subscriber = self.node.create_subscription(
            Bool, 'Pushing_flag', self.pushing_flag_callback, 10
        )
        self.node.get_logger().info('Subscribing to Pushing_flag topic')

    def pushing_flag_callback(self, msg):
        """Callback for Pushing_flag messages"""
        if msg.data:
            self.node.get_logger().info(f'Robot {self.robot_id}: Received Pushing_flag=True')
            self.blackboard.set('FollowDone', True)

    def initialise(self):
        # Trigger follow controller by setting Ready_flag to True
        ready_msg = Bool()
        ready_msg.data = True
        self.ready_flag_publisher.publish(ready_msg)
        self.node.get_logger().info(f'Robot {self.robot_id}: Starting Follow Control')
        # Reset the FollowDone flag in the blackboard
        self.blackboard.set('FollowDone', False)

    def update(self):
        # Check if follow control has completed
        if self.blackboard.exists('FollowDone') and self.blackboard.get('FollowDone'):
            self.node.get_logger().info(f'Robot {self.robot_id}: Follow Control completed')
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.SUCCESS:
            self.node.get_logger().info(f'Robot {self.robot_id}: Follow Control terminated successfully')
            # Reset the Ready_flag to stop the follow controller
            ready_msg = Bool()
            ready_msg.data = False
            self.ready_flag_publisher.publish(ready_msg)

# --- Pick Up Behavior ---
class PickUpBehavior(py_trees.behaviour.Behaviour):
    """Calls the pick up node to handle package pickup"""
    def __init__(self, name, robot_id):
        super().__init__(name)
        self.robot_id = robot_id
        self.blackboard = py_trees.blackboard.Blackboard()
        self.pickup_flag_publisher = None
        self.pickup_done_subscriber = None

    def setup(self, **kwargs):
        self.node = kwargs['node']
        # Publisher for Pickup_flag to activate the pickup controller (using existing topic name)
        self.pickup_flag_publisher = self.node.create_publisher(
            Bool, 'Pickup_flag', 10
        )
        self.node.get_logger().info(f'Publishing to Pickup_flag topic to trigger pickup controller')
        
        # Create local subscriber to track PickUp completion
        # We'll simulate completion with a timeout for now, since there's no explicit done flag
        self.blackboard.set('PickUpDone', False)
        return True

    def initialise(self):
        self.node.get_logger().info(f'Robot {self.robot_id}: Starting Pick Up')
        # Trigger pickup controller by setting Pickup_flag to True
        pickup_msg = Bool()
        pickup_msg.data = True
        self.pickup_flag_publisher.publish(pickup_msg)
        self.node.get_logger().info(f'Published Pickup_flag=True to trigger pickup')
        
        # Start time for simulating pickup completion
        self.start_time = self.node.get_clock().now()

    def update(self):
        # For now, we'll simulate completion after a fixed delay
        # In a real system, you'd use a callback from a completion topic
        current_time = self.node.get_clock().now()
        if (current_time - self.start_time).nanoseconds / 1e9 > 5.0:  # 5 second timeout
            self.blackboard.set('PickUpDone', True)
            self.node.get_logger().info(f'Robot {self.robot_id}: Pick Up completed')
            return py_trees.common.Status.SUCCESS
        
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.SUCCESS:
            self.node.get_logger().info(f'Robot {self.robot_id}: Pick Up terminated successfully')
        # Reset the Pickup_flag to stop the pickup controller
        if self.pickup_flag_publisher:
            msg = Bool()
            msg.data = False
            self.pickup_flag_publisher.publish(msg)

# --- Check Robot Not In Range ---
class CheckRobotNotInRange(py_trees.behaviour.Behaviour):
    """Checks if the previous robot is not in the starting relay point range or is in pushing mode"""
    def __init__(self, name, robot_id):
        super().__init__(name)
        self.robot_id = robot_id
        self.previous_robot_id = robot_id - 1 if robot_id > 0 else None
        self.blackboard = py_trees.blackboard.Blackboard()

    def update(self):
        # If this is the first robot, no previous robot to check
        if self.previous_robot_id is None:
            return py_trees.common.Status.SUCCESS
        
        # Check if previous robot is in range and not in pushing mode
        # If previous robot is in pushing mode, we can proceed
        if self.blackboard.exists(f'robot_{self.previous_robot_id}_pushing_flag') and \
           self.blackboard.get(f'robot_{self.previous_robot_id}_pushing_flag') == True:
            # Previous robot is in pushing mode, safe to proceed
            return py_trees.common.Status.SUCCESS
            
        # Check position only if robot is not in pushing mode
        if self.blackboard.exists(f'robot_{self.previous_robot_id}_at_relay_start') and \
           self.blackboard.get(f'robot_{self.previous_robot_id}_at_relay_start'):
            # Previous robot is in range and not pushing, cannot proceed
            return py_trees.common.Status.FAILURE
        
        # Previous robot is not in range, can proceed
        return py_trees.common.Status.SUCCESS

def create_robot_subtree(robot_id: int, total_robots: int):
    """创建单个机器人行为子树"""
    # Main sequence for this robot's behavior
    robot_sequence = py_trees.composites.Sequence(
        name=f'Robot_{robot_id}_Sequence', 
        memory=True
    )
    
    # Conditions for starting package handling
    package_conditions = py_trees.composites.Sequence(
        name=f'Package_Conditions_R{robot_id}',
        memory=True
    )
    
    # Check if package is at the starting relay point
    package_at_start = CheckBlackboardVariable(
        name=f'Check_Package_At_Start_R{robot_id}',
        key='package_at_relay_start',
        expected_value=True
    )
    
    # Check if previous robot is not in range
    robot_not_in_range = CheckRobotNotInRange(
        name=f'Check_Robot_Not_In_Range_R{robot_id}',
        robot_id=robot_id
    )
    
    # Package handling sequence
    package_handling = py_trees.composites.Sequence(
        name=f'Package_Handling_R{robot_id}',
        memory=True
    )
    
    # Follow control behavior
    follow_control = FollowControlBehavior(
        name=f'Follow_Control_R{robot_id}',
        robot_id=robot_id
    )
    
    # Check if follow control is completed
    follow_done = CheckBlackboardVariable(
        name=f'Check_Follow_Done_R{robot_id}',
        key='FollowDone',
        expected_value=True
    )
    
    # Pick up behavior
    pick_up = PickUpBehavior(
        name=f'Pick_Up_R{robot_id}',
        robot_id=robot_id
    )
    
    # Check if pick up is completed
    pick_up_done = CheckBlackboardVariable(
        name=f'Check_PickUp_Done_R{robot_id}',
        key='PickUpDone',
        expected_value=True
    )
    
    # Reset for next package
    reset_sequence = py_trees.composites.Sequence(
        name=f'Reset_For_Next_Package_R{robot_id}',
        memory=True
    )
    
    reset_follow_done = py_trees.behaviours.SetBlackboardVariable(
        name=f'Reset_Follow_Done_R{robot_id}',
        variable_name='FollowDone',
        variable_value=False,
        overwrite=True
    )
    
    reset_pick_up_done = py_trees.behaviours.SetBlackboardVariable(
        name=f'Reset_PickUp_Done_R{robot_id}',
        variable_name='PickUpDone',
        variable_value=False,
        overwrite=True
    )
    
    # Build the tree structure
    package_conditions.add_children([package_at_start, robot_not_in_range])
    package_handling.add_children([follow_control, follow_done, pick_up, pick_up_done])
    reset_sequence.add_children([reset_follow_done, reset_pick_up_done])
    
    robot_sequence.add_children([package_conditions, package_handling, reset_sequence])
    
    # Use a Decorator to repeat the behavior
    repeater = py_trees.decorators.FailureIsRunning(
        name=f"Repeat_Behavior_R{robot_id}",
        child=robot_sequence
    )
    
    return repeater

def create_multi_robot_tree(num_robots=5):
    """Create multi-robot behavior tree"""
    # Initialize ROS2 node
    node = Node('bt_controller')
    
    # Define a default QoS profile for subscribers
    default_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1
    )
    
    # Build the tree structure with parallel policy
    root = py_trees.composites.Parallel(
        name="MultiRobotSystem", 
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    
    # Add subscriber for parcel pose - using the global topic
    root.add_child(py_trees_ros.subscribers.ToBlackboard(
        name="ParcelPoseSub",
        topic_name='/parcel/pose',
        topic_type=PoseStamped,
        qos_profile=default_qos,
        blackboard_variables={'obj_pose': None}
    ))
    
    # Subscribe to the single Pushing_flag topic (not per-robot)
    root.add_child(py_trees_ros.subscribers.ToBlackboard(
        name="PushingFlagSub",
        topic_name='Pushing_flag',
        topic_type=Bool,
        qos_profile=default_qos,
        blackboard_variables={'pushing_flag': None}
    ))
    
    # Add robot poses (if available)
    for i in range(num_robots):
        # Add robot pose subscriber if the topic exists
        root.add_child(py_trees_ros.subscribers.ToBlackboard(
            name=f"Robot{i}OdomSub",
            topic_name=f'/robot_{i}/odom',
            topic_type=Odometry,
            qos_profile=default_qos,
            blackboard_variables={f'robot_{i}_pose': None}
        ))
    
    # Add behavior subtrees for each robot
    for i in range(num_robots):
        root.add_child(create_robot_subtree(i, num_robots))

    # Initialize package availability
    init_package = py_trees.behaviours.SetBlackboardVariable(
        name='Init_Package_Availability',
        variable_name='package_at_relay_start',
        variable_value=True,
        overwrite=True
    )
    
    root.add_child(init_package)

    return root, node

def main(args=None):
    rclpy.init(args=args)
    
    # Initialize blackboard parameters
    blackboard = py_trees.blackboard.Blackboard()
    blackboard.set('package_at_relay_start', True)  # Initial package is available
    blackboard.set('FollowDone', False)             # Follow behavior not yet complete
    blackboard.set('PickUpDone', False)             # Pick up behavior not yet complete
    
    # For each robot, initialize that it's not at the starting relay point
    for i in range(5):
        blackboard.set(f'robot_{i}_at_relay_start', False)
    
    # Create behavior tree
    tree, node = create_multi_robot_tree(num_robots=5)
    
    # Set up node to provide context for trees - this is needed for behaviors that need node access
    for behavior in tree.iterate():
        if hasattr(behavior, 'setup') and callable(behavior.setup):
            behavior.setup(node=node)
    
    # Create behavior tree without passing node directly to constructor
    bt = py_trees_ros.trees.BehaviourTree(tree)
    
    # Set execution frequency
    bt.setup(timeout=15)
    timer = node.create_timer(0.1, lambda: bt.tick())  # Using tick() instead of tick_once()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()