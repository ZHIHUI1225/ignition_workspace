#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from collections import defaultdict
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Header
import re

class ModelStateSplitter(Node):
    def __init__(self):
        super().__init__('model_state_publisher')
        
        # 使用defaultdict自动处理新模型
        self.publisher_factory = lambda: {
            'pose_pub': None,
            'twist_pub': None
        }
        self.model_publishers = defaultdict(self.publisher_factory)
        
        # 创建Gazebo模型状态订阅
        self.subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10
        )
        self.get_logger().info("Model state splitter initialized")

    def _determine_namespace(self, model_name):
        """Determine namespace based on model name's last number"""
        # Try to extract the last number from the model name
        match = re.search(r'(\d+)$', model_name)
        if match:
            number = match.group(1)
            return f"tb{number}"
        else:
            # Default case if no number found
            return ""

    def _create_publishers(self, model_name):
        """动态创建指定模型的发布者"""
        namespace = self._determine_namespace(model_name)
        if model_name in ['RelayPoint']:
            if not self.model_publishers[model_name]['pose_pub']:
                topic_name = f'/{model_name}/pose'
                self.model_publishers[model_name]['pose_pub'] = \
                    self.create_publisher(PoseStamped, topic_name, 10)
        else:
            if not self.model_publishers[model_name]['pose_pub']:
                # Use namespace in topic path if a namespace exists
                topic_name = f'/{namespace}/{model_name}/pose' if namespace else f'/{model_name}/pose'
                self.model_publishers[model_name]['pose_pub'] = \
                    self.create_publisher(PoseStamped, topic_name, 10)
                
            if not self.model_publishers[model_name]['twist_pub']:
                topic_name = f'/{namespace}/{model_name}/twist' if namespace else f'/{model_name}/twist'
                self.model_publishers[model_name]['twist_pub'] = \
                    self.create_publisher(TwistStamped, topic_name, 10)
            
        self.get_logger().debug(f"Created publishers for: {model_name} in namespace: {namespace}")

    def model_states_callback(self, msg):
        try:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'world'

            for name, pose, twist in zip(msg.name, msg.pose, msg.twist):
                # 跳过不需要处理的模型
                if name in ['ground_plane', 'WALL', 'MAZE']:
                    continue
                
                # 动态初始化发布者
                self._create_publishers(name)
                
                # 构造带时间戳的消息
                pose_msg = PoseStamped()
                pose_msg.header = header
                pose_msg.pose = pose
                
                twist_msg = TwistStamped()
                twist_msg.header = header
                twist_msg.twist = twist
                
                # 发布消息
                self.model_publishers[name]['pose_pub'].publish(pose_msg)
                self.model_publishers[name]['twist_pub'].publish(twist_msg)

        except Exception as e:
            self.get_logger().error(f"Callback error: {str(e)}", throttle_duration_sec=5)

def main(args=None):
    rclpy.init(args=args)
    node = ModelStateSplitter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()