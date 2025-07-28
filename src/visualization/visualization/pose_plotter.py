import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import json
import numpy as np
import math

# Camera node topics
TOPICS = [
    '/parcel0/pose',
    '/parcel1/pose',
    '/parcel2/pose',
    '/robot0/pose',
    '/robot1/pose',
    '/robot2/pose',
]
COLORS = ['red', 'orange', 'lime', 'purple', 'blue', 'green']
LABELS = ['parcel0', 'parcel1', 'parcel2', 'robot0', 'robot1', 'robot2']

# Load environment information
try:
    with open('/root/workspace/data/Enviro_experiments.json', 'r') as f:
        ENV_DATA = json.load(f)
except FileNotFoundError:
    ENV_DATA = {
        "polygons": [],
        "coord_bounds": [0, 1102, 0, 590],
        "width": 1102,
        "height": 590
    }
    print("Warning: Enviro_experiments.json not found, using default environment")

# Pixel to meter conversion factors (same as camera_node)
PIXEL_TO_METER_X = 0.00234
PIXEL_TO_METER_Y = 0.00232

class PosePlotter(Node):
    def __init__(self):
        super().__init__('pose_plotter')
        self.poses = {label: [] for label in LABELS}
        self.get_logger().info(f"Subscribing to topics: {TOPICS}")
        self.get_logger().info(f"Environment dimensions: {ENV_DATA['width']}x{ENV_DATA['height']} pixels")
        self.get_logger().info(f"Pixel to meter conversion: X={PIXEL_TO_METER_X}, Y={PIXEL_TO_METER_Y}")
        
        # Calculate environment size in meters
        env_width_m = ENV_DATA['width'] * PIXEL_TO_METER_X
        env_height_m = ENV_DATA['height'] * PIXEL_TO_METER_Y
        self.get_logger().info(f"Environment size in meters: {env_width_m:.3f}m x {env_height_m:.3f}m")
        
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for topic, label in zip(TOPICS, LABELS):
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, l=label: self.pose_callback(msg, l),
                qos
            )
            self.get_logger().info(f"Subscribed to {topic} for {label}")

    def pose_callback(self, msg, label):
        # Store poses as (x, y, orientation) for top-down view
        # Extract yaw from quaternion for direction visualization
        orientation = msg.pose.orientation
        # Convert quaternion to yaw angle (rotation around Z axis for top-down view)
        # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        self.poses[label].append((msg.pose.position.x, msg.pose.position.y, yaw))
        if len(self.poses[label]) > 1000:
            self.poses[label] = self.poses[label][-1000:]
        
    def draw_environment(self, ax):
        """Draw the environment polygons from Enviro_experiments.json converted to meters"""
        for i, polygon in enumerate(ENV_DATA['polygons']):
            vertices = np.array(polygon['vertices'], dtype=float)
            
            # Convert pixels to meters using the same conversion factors as camera_node
            vertices_meters = vertices.copy()
            vertices_meters[:, 0] *= PIXEL_TO_METER_X  # Convert X pixels to meters
            vertices_meters[:, 1] *= PIXEL_TO_METER_Y  # Convert Y pixels to meters
            
            # Center the environment around origin to match pose coordinate system
            center_x = ENV_DATA['width'] * PIXEL_TO_METER_X / 2
            center_y = ENV_DATA['height'] * PIXEL_TO_METER_Y / 2
            vertices_meters[:, 0] -= center_x  # Center X around origin
            vertices_meters[:, 1] -= center_y  # Center Y around origin
            
            # Close the polygon by adding the first vertex at the end
            vertices_meters = np.vstack([vertices_meters, vertices_meters[0]])
            
            # Plot polygon outline
            ax.plot(vertices_meters[:, 0], vertices_meters[:, 1], 'k-', linewidth=2, alpha=0.7)
            
            # Fill polygon with light gray
            polygon_patch = plt.Polygon(vertices_meters[:-1], color='lightgray', alpha=0.3, 
                                      label=f'Environment {i+1}' if i < 2 else "")
            ax.add_patch(polygon_patch)

def main(args=None):
    rclpy.init(args=args)
    node = PosePlotter()
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            ax.clear()
            
            # Draw environment first
            node.draw_environment(ax)
            
            # Plot trajectories for all entities
            for label, color in zip(LABELS, COLORS):
                data = node.poses[label]
                if data and len(data) > 1:
                    # Extract x, y coordinates for trajectory (ignore orientation)
                    if len(data[0]) == 3:  # New format with orientation
                        xs, ys, _ = zip(*data)
                    else:  # Old format without orientation
                        xs, ys = zip(*data)
                    ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1, label=f'{label} trajectory')
            
            # Plot current positions and orientations
            for label, color in zip(LABELS, COLORS):
                data = node.poses[label]
                if data:
                    if len(data[-1]) == 3:  # New format with orientation
                        x, y, yaw = data[-1]  # Latest position and orientation
                    else:  # Old format without orientation
                        x, y = data[-1]
                        yaw = 0  # Default orientation
                    
                    if 'parcel' in label:
                        # Plot parcels as squares (5cm x 5cm markers)
                        square = plt.Rectangle((x-0.025, y-0.025), 0.05, 0.05, 
                                             color=color, alpha=0.8, label=label)
                        ax.add_patch(square)
                    elif 'robot' in label:
                        # Plot robots as circles (3.6cm x 3.6cm markers, so radius ~0.018)
                        circle = plt.Circle((x, y), 0.018, color=color, alpha=0.8, label=label)
                        ax.add_patch(circle)
                        
                        # Draw direction arrow for robots
                        arrow_length = 0.08  # Length of direction arrow
                        dx = arrow_length * math.cos(yaw)
                        dy = arrow_length * math.sin(yaw)
                        
                        # Draw arrow showing robot direction
                        ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.02, 
                                fc=color, ec=color, alpha=0.9, linewidth=2)
                    
                    # Add text label near the object
                    ax.text(x + 0.05, y + 0.05, label, fontsize=8, color=color, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Set plot properties
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title('Camera Node Pose Tracking - Environment View (X-Y Plane)')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set coordinate bounds based on environment size in meters
            env_width_m = ENV_DATA['width'] * PIXEL_TO_METER_X
            env_height_m = ENV_DATA['height'] * PIXEL_TO_METER_Y
            
            # Center the view around the environment
            ax.set_xlim(-env_width_m/2 - 0.5, env_width_m/2 + 0.5)
            ax.set_ylim(-env_height_m/2 - 0.5, env_height_m/2 + 0.5)
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            plt.pause(0.05)
            
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down pose plotter...")
    
    node.destroy_node()
    rclpy.shutdown()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
