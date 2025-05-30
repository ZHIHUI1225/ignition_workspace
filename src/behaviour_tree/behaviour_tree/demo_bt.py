import py_trees
import rclpy
import threading
import time
import py_trees.display
from py_trees_ros.trees import BehaviourTree
import py_trees_ros.visitors

class SayHello(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.logger = rclpy.logging.get_logger(name)

    def update(self):
        self.logger.info("Hello from Python Behavior Tree!")
        return py_trees.common.Status.SUCCESS

def create_tree(ros_node):  # Add ros_node argument
    # Create a dummy waiting behavior
    root = py_trees.composites.Sequence(name="DemoSequence", memory=True)
    root.add_child(SayHello("Greet"))
    # Use Success behavior instead of WaitForSignal which doesn't exist in py_trees 2.3.0
    root.add_child(py_trees.decorators.RunningIsFailure(
        name="HandleWait",
        child=py_trees.behaviours.Success(name="DummyWait")
    ))
    return BehaviourTree(root=root)  # Remove node=ros_node

def main():
    rclpy.init()
    
    try:
        # Create a ROS node for our behavior tree
        ros_node = rclpy.create_node("demo_behavior_tree")
        
        # Create the behavior tree
        tree = create_tree(ros_node)  # Pass ros_node here
        
        # Enable snapshot streams - this is the key for visualization!
        # According to py_trees_ros tutorials, we need to enable these parameters
        ros_node.declare_parameter('enable_snapshot_visitor', True)
        ros_node.declare_parameter('snapshot_visitor_publish_blackboard', True)
        ros_node.declare_parameter('snapshot_visitor_publish_activity', True)
        
        # Setup the tree (this will initialize the visitors and publishers)
        # Pass ros_node to tree.setup() so the tree uses it for publishing
        tree.setup(timeout=15, node=ros_node)
        
        # Diagnostic logging
        ros_node.get_logger().info("--- DIAGNOSTICS START ---")
        if tree.node:
            ros_node.get_logger().info(f"Diagnostic: tree.node name after setup: {tree.node.get_name()}")
            ros_node.get_logger().info(f"Diagnostic: tree.node namespace after setup: {tree.node.get_namespace()}")
            if tree.node is ros_node:
                ros_node.get_logger().info("Diagnostic: tree.node IS the same instance as ros_node.")
            else:
                ros_node.get_logger().warn("Diagnostic: tree.node is NOT the same instance as ros_node.")
        else:
            ros_node.get_logger().error("Diagnostic: tree.node is None after setup!")

        # Print tree structure
        print("\n" + "="*80)
        print("BEHAVIOR TREE STRUCTURE:")
        print("="*80)
        print(py_trees.display.ascii_tree(tree.root))
        print("="*80)
        
        # Create ROS executor for processing callbacks
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(ros_node)
        
        # Create a background thread for the executor
        def spin_ros():
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
                
        ros_thread = threading.Thread(target=spin_ros)
        ros_thread.daemon = True
        ros_thread.start()
        
        print("Tree is running with ROS topic publishing...")
        node_name = ros_node.get_name()
        print(f"Expected topics for visualization (node '{node_name}'):")
        print(f"  - /{node_name}/snapshots")
        print(f"  - /{node_name}/status_changes")
        print(f"  - /{node_name}/logs")
        print(f"  - /{node_name}/blackboard/activity_stream")
        print(f"  - /{node_name}/blackboard/access_stream")
        print(f"Expected services (node '{node_name}'):")
        print(f"  - /{node_name}/blackboard/get")
        print(f"  - /{node_name}/blackboard/set")
        print("="*80 + "\\\\n")
        
        # Main loop
        count = 0
        while rclpy.ok() and count < 100:  # Run for ~10 seconds
            tree.tick()
            time.sleep(0.1)
            count += 1
            
        print("\nDemo complete. Tree ran for 10 seconds.")
            
    except KeyboardInterrupt:
        print("\nShutting down behavior tree...")
        if 'tree' in locals():
            tree.shutdown()
    except Exception as e:
        print(f"Error running behavior tree: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass