#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class PathPlotterNode(Node):
    def __init__(self):
        super().__init__('path_plotter_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/camera/processed_image', self.image_callback, 10)
        self.pose_subs = []
        self.robot_paths = {0: [], 1: [], 2: []}  # Store (x, y) in meters (live)
        self.loaded_paths = {0: [], 1: [], 2: []}  # Store loaded (x, y) in meters
        self.relay_points = []  # Store relay points (nodes) in pixels
        self.waypoint_flags = {}  # Store waypoint flags
        self.first_waypoint = None  # Store first waypoint
        self.last_waypoint = None  # Store last waypoint
        self.last_positions = {0: None, 1: None, 2: None}
        for i in range(3):
            topic = f'/robot{i}/pose'
            self.pose_subs.append(
                self.create_subscription(PoseStamped, topic, lambda msg, idx=i: self.pose_callback(msg, idx), 10)
            )
        self.current_frame = None
        self.window_name = 'Video with Path'
        cv2.namedWindow(self.window_name)
        self.get_logger().info('PathPlotterNode started.')

        # Load trajectory files for each robot
        import os, json
        base_path = '/root/workspace/data/experi'
        traj_files = [
            os.path.join(base_path, 'tb0_Trajectory.json'),
            os.path.join(base_path, 'tb1_Trajectory.json'),
            os.path.join(base_path, 'tb2_Trajectory.json'),
        ]
        for idx, file in enumerate(traj_files):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    # Each entry: [x, y, theta, v, w] in meters and radians
                    self.loaded_paths[idx] = [(float(pt[0]), float(pt[1]), float(pt[2])) for pt in data['Trajectory']]
                self.get_logger().info(f"Loaded {len(self.loaded_paths[idx])} points from {file}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load {file}: {e}")

        # Load waypoint flags from WayPointFlag3experi.json
        try:
            waypoint_file = '/root/workspace/data/WayPointFlag3experi.json'
            with open(waypoint_file, 'r') as f:
                waypoint_data = json.load(f)
                waypoints = waypoint_data['Waypoints']
                flags = waypoint_data['Flags']
                # Store first and last waypoints
                if waypoints:
                    self.first_waypoint = waypoints[0]
                    self.last_waypoint = waypoints[-1]
                # Create mapping of waypoint_id -> flag
                for waypoint_id, flag in zip(waypoints, flags):
                    self.waypoint_flags[waypoint_id] = flag
            self.get_logger().info(f"Loaded waypoint flags: {self.waypoint_flags}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load waypoint flags: {e}")

        # Load relay points (nodes) from Graph_new_experi.json
        try:
            graph_file = '/root/workspace/data/Graph_new_experi.json'
            with open(graph_file, 'r') as f:
                graph_data = json.load(f)
                # Each node: [id, [x_pixel, y_pixel], null, false]
                for node in graph_data['nodes']:
                    node_id = node[0]
                    x_pixel, y_pixel = node[1]
                    # Include nodes that are:
                    # 1. The first waypoint (regardless of flag), OR
                    # 2. The last waypoint (regardless of flag), OR
                    # 3. In waypoint flags and have non-zero flags
                    if (node_id == self.first_waypoint) or (node_id == self.last_waypoint) or (node_id in self.waypoint_flags and self.waypoint_flags[node_id] != 0):
                        self.relay_points.append((node_id, int(x_pixel), int(y_pixel)))
            self.get_logger().info(f"Loaded {len(self.relay_points)} filtered relay points from graph")
        except Exception as e:
            self.get_logger().warn(f"Failed to load graph file: {e}")

    def pose_callback(self, msg, robot_idx):
        # Use x, y from pose (assume z is up, so x/y is ground plane)
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.last_positions[robot_idx] = (x, y)
        # Store path (limit to last 100 points for memory)
        self.robot_paths[robot_idx].append((x, y))
        if len(self.robot_paths[robot_idx]) > 100:
            self.robot_paths[robot_idx] = self.robot_paths[robot_idx][-100:]

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # Create overlay for transparent drawing
        overlay = frame.copy()
        alpha = 0.6  # Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)
        
        # Draw relay points (nodes) from graph on overlay
        # Convert 0.12m radius to pixels using the conversion factor
        radius_meters = 0.12
        meter_to_pixel = 1.0 / 0.0023  # pixels per meter
        radius_pixels = int(radius_meters * meter_to_pixel)
        
        for node_id, px, py in self.relay_points:
            # Use different colors for different waypoint types
            if node_id == self.first_waypoint:
                # Draw first waypoint with green color
                cv2.circle(overlay, (px, py), radius_pixels, (0, 255, 0), 2)  # Green circle
                cv2.circle(overlay, (px, py), radius_pixels-2, (0, 200, 0), -1)  # Green fill
            elif node_id == self.last_waypoint:
                # Draw last waypoint with red color
                cv2.circle(overlay, (px, py), radius_pixels, (0, 0, 255), 2)  # Red circle
                cv2.circle(overlay, (px, py), radius_pixels-2, (0, 0, 200), -1)  # Red fill
            else:
                # Draw flagged waypoints with white/gray color
                cv2.circle(overlay, (px, py), radius_pixels, (255, 255, 255), 2)  # White circle
                cv2.circle(overlay, (px, py), radius_pixels-2, (128, 128, 128), -1)  # Gray fill
            # Draw node ID (opaque)
            cv2.putText(frame, str(node_id), (px-5, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw loaded (planned) paths for each robot on overlay
        colors = [(0,0,255), (0,255,0), (255,0,0)]  # BGR for robots 0,1,2
        for idx, path in self.loaded_paths.items():
            if len(path) > 1:
                pts = [self.world_to_image_coords(x, y, frame.shape) for (x, y, theta) in path]
                for j in range(1, len(pts)):
                    cv2.line(overlay, pts[j-1], pts[j], colors[idx], 1, lineType=cv2.LINE_AA)
                # Draw orientation arrows at some points
                for j in range(0, len(path), 5):  # Every 5th point
                    x, y, theta = path[j]
                    px, py = self.world_to_image_coords(x, y, frame.shape)
                    # Draw small arrow showing orientation
                    arrow_length = 15
                    end_x = int(px + arrow_length * np.cos(theta))
                    end_y = int(py + arrow_length * np.sin(theta))
                    cv2.arrowedLine(overlay, (px, py), (end_x, end_y), colors[idx], 1, tipLength=0.3)
        # Draw live (actual) paths for each robot on overlay
        for idx, path in self.robot_paths.items():
            if len(path) > 1:
                pts = [self.world_to_image_coords(x, y, frame.shape) for (x, y) in path]
                for j in range(1, len(pts)):
                    cv2.line(overlay, pts[j-1], pts[j], colors[idx], 2)
        
        # Blend overlay with original frame for transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Optionally, draw current robot positions
        for idx, pos in self.last_positions.items():
            if pos is not None:
                pt = self.world_to_image_coords(pos[0], pos[1], frame.shape)
                cv2.circle(frame, pt, 6, colors[idx], -1)
                cv2.putText(frame, f'R{idx}', (pt[0]+8, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def world_to_image_coords(self, x, y, img_shape):
        # Convert world coordinates (meters) to image pixel coordinates
        # Origin is at top-left corner of the image
        # Conversion factor: divide by 0.0023 to convert from meters to pixels
        h, w = img_shape[:2]
        
        # Convert meters to pixels using the specified conversion factor
        meter_to_pixel = 1.0 / 0.0023  # pixels per meter
        px = int(x * meter_to_pixel)
        py = int(y * meter_to_pixel)  # y=0 at top
        
        # Clamp to image bounds
        px = max(0, min(w-1, px))
        py = max(0, min(h-1, py))
        
        return (px, py)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PathPlotterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
