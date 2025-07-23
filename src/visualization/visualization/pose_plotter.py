import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt

TOPICS = [
    '/vrpn_mocap/parcel0/pose',
    '/vrpn_mocap/parcel1/pose',
    '/vrpn_mocap/robot0/pose',
    '/vrpn_mocap/robot1/pose',
    '/vrpn_mocap/robot2/pose',
]
COLORS = ['r', 'g', 'b', 'c', 'm']
LABELS = ['parcel0', 'parcel1', 'robot0', 'robot1', 'robot2']

class PosePlotter(Node):
    def __init__(self):
        super().__init__('pose_plotter')
        self.poses = {label: [] for label in LABELS}
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for topic, label in zip(TOPICS, LABELS):
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, l=label: self.pose_callback(msg, l),
                qos
            )

    def pose_callback(self, msg, label):
        self.poses[label].append((msg.pose.position.x, msg.pose.position.z))
        if len(self.poses[label]) > 1000:
            self.poses[label] = self.poses[label][-1000:]

def main(args=None):
    rclpy.init(args=args)
    node = PosePlotter()
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))  # Make the plot bigger
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            ax.clear()
            for label, color in zip(LABELS, COLORS):
                data = node.poses[label]
                if data:
                    xs, zs = zip(*data)
                    ax.plot(xs, zs, color, label=label)
                    ax.scatter(xs[-1], zs[-1], color=color, s=100, marker='o')  # Highlight latest point, bigger
            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title('VRPN Mocap Trajectories (X vs Z)')
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()