#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import sys

# Import coordinate transformation functions
sys.path.append('/root/workspace/src/Replanning/scripts')
from coordinate_transform import (
    convert_world_meter_to_camera_pixel
)

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
        self.robot0_orientation = None  # Store robot0 orientation from odometry
        
        # Subscribe to pose topics for all robots
        for i in range(3):
            topic = f'/robot{i}/pose'
            self.pose_subs.append(
                self.create_subscription(PoseStamped, topic, lambda msg, idx=i: self.pose_callback(msg, idx), 10)
            )
        
        # Subscribe to robot0 odometry for orientation
        self.odom_sub = self.create_subscription(
            Odometry, '/robot0/odom', self.odom_callback, 10)
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
                # Note: These coordinates are in world_pixel frame, need to convert to world_meter
                relay_points_data = []  # Store all relay points data for saving
                for node in graph_data['nodes']:
                    node_id = node[0]
                    world_px, world_py = node[1]
                    # Include nodes that are:
                    # 1. The first waypoint (regardless of flag), OR
                    # 2. The last waypoint (regardless of flag), OR
                    # 3. In waypoint flags and have non-zero flags
                    if (node_id == self.first_waypoint) or (node_id == self.last_waypoint) or (node_id in self.waypoint_flags and self.waypoint_flags[node_id] != 0):
                        # Store relay point in world_pixel coordinates
                        relay_points_data.append({
                            "id": node_id,
                            "world_pixel": [world_px, world_py],
                            "is_first": (node_id == self.first_waypoint),
                            "is_last": (node_id == self.last_waypoint),
                            "flag": self.waypoint_flags.get(node_id, 0)
                        })
                        
                        # Convert world_pixel to world_meter coordinates for display
                        world_x, world_y = self.world_pixel_to_world_meter(world_px, world_py)
                        self.relay_points.append((node_id, world_x, world_y))
                
                # Save relay points to a dedicated file for easy loading by behavior tree
                relay_points_file = '/root/workspace/data/relay_points.json'
                try:
                    with open(relay_points_file, 'w') as f:
                        json.dump({
                            "relay_points": relay_points_data,
                            "pixel_to_meter_scale": 0.0023,  # Include scale factor for convenience
                            "description": "Relay points in world_pixel coordinates for robot coordination"
                        }, f, indent=2)
                    self.get_logger().info(f"Saved {len(relay_points_data)} relay points to {relay_points_file}")
                except Exception as save_err:
                    self.get_logger().error(f"Failed to save relay points file: {save_err}")
                    
            self.get_logger().info(f"Loaded {len(self.relay_points)} filtered relay points from graph")
        except Exception as e:
            self.get_logger().warn(f"Failed to load graph file: {e}")
            
    def world_pixel_to_world_meter(self, world_px, world_py):
        """Convert world pixel coordinates to world meter coordinates"""
        # Convert world_pixel to world_meter using scale factor
        pixel_to_meter = 0.0023  # meters per pixel
        world_x = world_px * pixel_to_meter
        world_y = world_py * pixel_to_meter
        
        return world_x, world_y

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle (rotation around Z-axis)"""
        # Extract quaternion components
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        
        # Convert to yaw angle for pure Z-axis rotation
        # For quaternions created with qz = sin(yaw/2), qw = cos(yaw/2), qx = 0, qy = 0
        # The inverse formula is: yaw = 2 * atan2(qz, qw)
        yaw = 2 * math.atan2(z, w)
        
        # DEBUG: Print quaternion and extracted yaw
        print(f"[PLOT_VIDEO] Received quaternion: qx={x:.4f}, qy={y:.4f}, qz={z:.4f}, qw={w:.4f}")
        print(f"[PLOT_VIDEO] Extracted yaw: {yaw:.4f} rad = {math.degrees(yaw):.2f} deg")
        
        return yaw

    def odom_callback(self, msg):
        """Handle odometry messages for robot0"""
        # Extract position and orientation from odometry (already in world_meter frame)
        world_meter_x = msg.pose.pose.position.x
        world_meter_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw angle
        orientation_quat = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(orientation_quat)
        
        # DEBUG: Print robot0 position in different coordinate frames
        print(f"[ROBOT0_POSITION] === Position in different frames ===")
        print(f"[ROBOT0_POSITION] World_meter: x={world_meter_x:.4f}m, y={world_meter_y:.4f}m")
        
        # Convert to world_pixel coordinates using the scale factor
        meter_to_pixel = 1.0 / 0.0023  # pixels per meter
        world_pixel_x = world_meter_x * meter_to_pixel
        world_pixel_y = world_meter_y * meter_to_pixel
        print(f"[ROBOT0_POSITION] World_pixel: x={world_pixel_x:.2f}px, y={world_pixel_y:.2f}px")
        
        # Convert to camera_pixel coordinates using the coordinate transformation function
        try:
            world_meter_pos = [world_meter_x, world_meter_y]
            camera_pixel_pos = convert_world_meter_to_camera_pixel(world_meter_pos)
            camera_pixel_x = int(camera_pixel_pos[0])
            camera_pixel_y = int(camera_pixel_pos[1])
            
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                h, w = self.current_frame.shape[:2]
                print(f"[ROBOT0_POSITION] Camera_pixel: x={camera_pixel_x}px, y={camera_pixel_y}px (image: {w}x{h})")
            else:
                print(f"[ROBOT0_POSITION] Camera_pixel: x={camera_pixel_x}px, y={camera_pixel_y}px")
        except Exception as e:
            print(f"[ROBOT0_POSITION] Error in coordinate transformation: {e}")
            # Fallback to simple conversion
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                h, w = self.current_frame.shape[:2]
                camera_pixel_x = int(world_pixel_x)
                camera_pixel_y = int(h - world_pixel_y)  # Flip Y-axis
                print(f"[ROBOT0_POSITION] Camera_pixel (fallback): x={camera_pixel_x}px, y={camera_pixel_y}px (image: {w}x{h})")
        
        print(f"[ROBOT0_POSITION] Orientation: {yaw:.4f} rad = {math.degrees(yaw):.2f} deg")
        print(f"[ROBOT0_POSITION] =====================================")
        
        # Store position and orientation
        self.last_positions[0] = (world_meter_x, world_meter_y)
        self.robot0_orientation = yaw

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
            self.current_frame = frame  # Store frame for coordinate calculations
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
        
        for node_id, world_x, world_y in self.relay_points:
            # Convert world_meter coordinates to camera_pixel coordinates
            px, py = self.world_to_image_coords(world_x, world_y, frame.shape)
            
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
                    # Note: Since Y-axis is flipped in camera coordinates, we need to negate theta
                    camera_theta = -theta  # Flip angle for camera coordinate system
                    arrow_length = 15
                    end_x = int(px + arrow_length * np.cos(camera_theta))
                    end_y = int(py + arrow_length * np.sin(camera_theta))
                    cv2.arrowedLine(overlay, (px, py), (end_x, end_y), colors[idx], 1, tipLength=0.3)
        # Draw live (actual) paths for each robot on overlay
        for idx, path in self.robot_paths.items():
            if len(path) > 1:
                pts = [self.world_to_image_coords(x, y, frame.shape) for (x, y) in path]
                for j in range(1, len(pts)):
                    cv2.line(overlay, pts[j-1], pts[j], colors[idx], 2)
        
        # Blend overlay with original frame for transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw current robot positions
        for idx, pos in self.last_positions.items():
            if pos is not None:
                pt = self.world_to_image_coords(pos[0], pos[1], frame.shape)
                cv2.circle(frame, pt, 6, colors[idx], -1)
                cv2.putText(frame, f'R{idx}', (pt[0]+8, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)
                
                # Draw direction arrow for robot0 if orientation is available
                if idx == 0 and self.robot0_orientation is not None:
                    # Convert world orientation to camera coordinate system
                    camera_theta = -self.robot0_orientation  # Flip angle for camera Y-axis
                    arrow_length = 25
                    end_x = int(pt[0] + arrow_length * np.cos(camera_theta))
                    end_y = int(pt[1] + arrow_length * np.sin(camera_theta))
                    cv2.arrowedLine(frame, pt, (end_x, end_y), colors[idx], 3, tipLength=0.4)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def world_to_image_coords(self, x, y, img_shape):
        # Convert world coordinates (meters) to camera pixel coordinates using the coordinate transformation function
        try:
            world_meter_pos = [x, y]
            camera_pixel_pos = convert_world_meter_to_camera_pixel(world_meter_pos)
            camera_px = int(camera_pixel_pos[0])
            camera_py = int(camera_pixel_pos[1])
            
            # Clamp to image bounds
            h, w = img_shape[:2]
            camera_px = max(0, min(w-1, camera_px))
            camera_py = max(0, min(h-1, camera_py))
            
            return (camera_px, camera_py)
            
        except Exception as e:
            # Fallback to the previous method if coordinate transformation fails
            h, w = img_shape[:2]
            
            # Step 1: Convert from world_meter to world_pixel using scale factor
            meter_to_pixel = 1.0 / 0.0023  # pixels per meter
            world_px = x * meter_to_pixel
            world_py = y * meter_to_pixel
            
            # Step 2: Convert from world_pixel (bottom-left origin, Y-up) to camera_pixel (top-left origin, Y-down)
            # In world_pixel: (0,0) is bottom-left, Y increases upward
            # In camera_pixel: (0,0) is top-left, Y increases downward
            camera_px = int(world_px)
            camera_py = int(h - world_py)  # Flip Y-axis: camera_py = height - world_py
            
            # Clamp to image bounds
            camera_px = max(0, min(w-1, camera_px))
            camera_py = max(0, min(h-1, camera_py))
            
            return (camera_px, camera_py)

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
