import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt

TOPICS = [
    '/vrpn_mocap/wall0/pose',
    '/vrpn_mocap/wall1/pose',
    '/vrpn_mocap/parcel1/pose',
    '/vrpn_mocap/parcel/pose',
    '/vrpn_mocap/parcel0/pose',
    '/vrpn_mocap/parcel2/pose',
    '/vrpn_mocap/robot1/pose',
    '/vrpn_mocap/robot2/pose',
    '/vrpn_mocap/robot0/pose',
]
COLORS = ['gray', 'dimgray', 'orange', 'gold', 'red', 'lime', 'blue', 'green', 'purple']
LABELS = ['wall0', 'wall1', 'parcel1', 'parcel', 'parcel0', 'parcel2', 'robot1', 'robot2', 'robot0']

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
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            ax.clear()
            # Plot walls as rectangles
            if node.poses['wall0']:
                x, z = node.poses['wall0'][-1]
                rect = plt.Rectangle((x-0.3, z-0.3), 0.6, 0.6, color='gray', alpha=0.5, label='wall0')
                ax.add_patch(rect)
            if node.poses['wall1']:
                x, z = node.poses['wall1'][-1]
                rect = plt.Rectangle((x-0.45, z-0.4), 0.9, 0.8, color='dimgray', alpha=0.5, label='wall1')
                ax.add_patch(rect)
            # Plot parcel1 as a square
            if node.poses['parcel1']:
                x, z = node.poses['parcel1'][-1]
                square = plt.Rectangle((x-0.025, z-0.025), 0.05, 0.05, color='orange', alpha=0.8, label='parcel1')
                ax.add_patch(square)
            # Plot parcel as a square
            if node.poses['parcel']:
                x, z = node.poses['parcel'][-1]
                square = plt.Rectangle((x-0.025, z-0.025), 0.05, 0.05, color='gold', alpha=0.8, label='parcel')
                ax.add_patch(square)
            # Plot parcel0 as a square
            if node.poses['parcel0']:
                x, z = node.poses['parcel0'][-1]
                square = plt.Rectangle((x-0.025, z-0.025), 0.05, 0.05, color='red', alpha=0.8, label='parcel0')
                ax.add_patch(square)
            # Plot parcel2 as a square
            if node.poses['parcel2']:
                x, z = node.poses['parcel2'][-1]
                square = plt.Rectangle((x-0.025, z-0.025), 0.05, 0.05, color='lime', alpha=0.8, label='parcel2')
                ax.add_patch(square)
            # Plot robot1 as a circle
            if node.poses['robot1']:
                x, z = node.poses['robot1'][-1]
                circ = plt.Circle((x, z), 0.04, color='blue', alpha=0.8, label='robot1')
                ax.add_patch(circ)
            # Plot robot2 as a circle
            if node.poses['robot2']:
                x, z = node.poses['robot2'][-1]
                circ = plt.Circle((x, z), 0.04, color='green', alpha=0.8, label='robot2')
                ax.add_patch(circ)
            # Plot robot0 as a circle
            if node.poses['robot0']:
                x, z = node.poses['robot0'][-1]
                circ = plt.Circle((x, z), 0.04, color='purple', alpha=0.8, label='robot0')
                ax.add_patch(circ)
            # Plot trajectories for all
            for label, color in zip(LABELS, COLORS):
                data = node.poses[label]
                if data:
                    xs, zs = zip(*data)
                    ax.plot(xs, zs, color, label=f'{label} traj')
            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title('VRPN Mocap Visualization')
            ax.set_aspect('equal', adjustable='datalim')
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()