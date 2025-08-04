#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
import sys
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# Import coordinate transformation functions
sys.path.append('/root/workspace/src/Replanning/scripts')
from coordinate_transform import (
    convert_camera_pixel_to_world_meter,
    convert_camera_angle_to_world_angle,
    convert_world_meter_to_camera_pixel,
    get_frame_info
)

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Create publisher for processed video
        self.image_publisher = self.create_publisher(Image, '/camera/processed_image', 10)
        self.bridge = CvBridge()
        
        # Create publishers for ArUco marker poses
        # Robot publishers (IDs 0, 1, 2)
        self.robot0_publisher = self.create_publisher(Odometry, '/robot0/odom', 10)
        self.robot1_publisher = self.create_publisher(Odometry, '/robot1/odom', 10)
        self.robot2_publisher = self.create_publisher(Odometry, '/robot2/odom', 10)
        
        # Parcel publishers (IDs 5, 6, 7)
        self.parcel0_publisher = self.create_publisher(Odometry, '/parcel0/odom', 10)
        self.parcel1_publisher = self.create_publisher(Odometry, '/parcel1/odom', 10)
        self.parcel2_publisher = self.create_publisher(Odometry, '/parcel2/odom', 10)
        
        # Camera setup
        self.cap = cv2.VideoCapture('/dev/video0')
        
        # Optimize camera settings for consistent performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set camera FPS to ensure consistent capture
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, -30)  # 降低曝光度
        self.cap.set(cv2.CAP_PROP_CONTRAST, 5)  # 增加对比度
        
        # Additional optimization settings
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for consistent performance

        # Camera calibration parameters
        fx_r = 582.59118861
        fy_r = 582.65884802
        cx_r = 629.53535406
        cy_r = 348.71988126
        k1_r = 0.00239457
        k2_r = -0.03004914
        p1_r = -0.00062043
        p2_r = -0.00057221
        k3_r = 0.01083464
        self.cameraMatrix_r = np.array([[fx_r,0.0,cx_r], [0.0,fy_r,cy_r], [0.0,0.0,1]], dtype=np.float32)
        self.distCoeffs_r = np.array([k1_r, k2_r, p1_r, p2_r, k3_r], dtype=np.float32)
        
        # Undistortion setup
        dz2 = (1280,720)
        newCameraMatrix_r, _ = cv2.getOptimalNewCameraMatrix(self.cameraMatrix_r, self.distCoeffs_r, dz2, 0, dz2)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(self.cameraMatrix_r, self.distCoeffs_r, None, newCameraMatrix_r, dz2, cv2.CV_16SC2)

        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera")
            return

        self.get_logger().info("Camera Node Started - Publishing processed video on /camera/processed_image")
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera resolution: {actual_width}x{actual_height}")
        
        # ArUco setup for markers with ID < 15
        # DICT_6X6_50 contains markers 0-49, which covers all needed IDs (0,1,2,5,6,7)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        aruco_params = cv2.aruco.DetectorParameters()
        
        # 优化检测参数
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 23
        aruco_params.adaptiveThreshWinSizeStep = 10
        aruco_params.adaptiveThreshConstant = 7
        aruco_params.minMarkerPerimeterRate = 0.03
        aruco_params.maxMarkerPerimeterRate = 4.0
        aruco_params.polygonalApproxAccuracyRate = 0.03
        aruco_params.minCornerDistanceRate = 0.05
        aruco_params.minDistanceToBorder = 3
        aruco_params.minMarkerDistanceRate = 0.05
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 5
        aruco_params.cornerRefinementMaxIterations = 30
        aruco_params.cornerRefinementMinAccuracy = 0.1
        
        # Create single detector for markers with ID < 15
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # 像素到米的转换因子
        self.pixel_to_meter_x = 0.0023
        self.pixel_to_meter_y = 0.0023

        # Create timer for camera capture
        self.timer = self.create_timer(0.1, self.capture_and_publish)  # 10 Hz
        
        # Create timer for periodic health checks
        self.health_timer = self.create_timer(5.0, self.check_system_health)  # Every 5 seconds
        
        # Setup display window (no mouse tracking needed)
        cv2.namedWindow('Camera Feed')
        
        # ROI tracking for consecutive frames
        self.previous_markers = {}  # Store previous marker positions {marker_id: (center_x, center_y)}
        self.roi_expansion = 100  # Pixels to expand around previous marker position
        self.full_detection_interval = 10  # Perform full detection every N frames
        self.frame_count = 0
        self.show_roi_debug = False  # Set to True to visualize ROI regions
        self.required_marker_ids = {0, 1, 2, 5, 6, 7}  # All required marker IDs
        
        # Log coordinate frame information
        try:
            frame_info = get_frame_info()
            self.get_logger().info("=== Coordinate Frame Configuration ===")
            for frame_name, frame_data in frame_info.items():
                self.get_logger().info(f"{frame_data['name']}: Units={frame_data['units']}")
            self.get_logger().info("Publishing robot poses in world frame (origin: left-lower corner, Y-up, meters)")
        except Exception as e:
            self.get_logger().warn(f"Could not load coordinate frame info: {e}")

    def create_roi_mask(self, img_shape):
        """Create ROI mask based on previous marker detections"""
        if not self.previous_markers:
            # No previous markers, return full image mask
            return np.ones((img_shape[0], img_shape[1]), dtype=np.uint8) * 255
        
        # Create black mask initially
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        
        # Add ROI regions around previous marker positions
        for marker_id, (prev_x, prev_y) in self.previous_markers.items():
            # Calculate ROI bounds with expansion
            x1 = max(0, int(prev_x - self.roi_expansion))
            y1 = max(0, int(prev_y - self.roi_expansion))
            x2 = min(img_shape[1], int(prev_x + self.roi_expansion))
            y2 = min(img_shape[0], int(prev_y + self.roi_expansion))
            
            # Set ROI region to white (255) in mask
            mask[y1:y2, x1:x2] = 255
            
            # Optional: Draw ROI rectangle for visualization
            # cv2.rectangle(mask, (x1, y1), (x2, y2), 255, 2)
        
        return mask

    def detect_markers_with_roi(self, img):
        """Detect ArUco markers with ROI optimization"""
        self.frame_count += 1
        
        # Check if all required markers are currently detected
        detected_marker_ids = set(self.previous_markers.keys()) if self.previous_markers else set()
        all_markers_detected = self.required_marker_ids.issubset(detected_marker_ids)
        
        # Perform full detection if:
        # 1. Periodic interval reached
        # 2. No previous markers exist
        # 3. Not all required markers (0,1,2,5,6,7) are detected
        if (self.frame_count % self.full_detection_interval == 0) or not self.previous_markers or not all_markers_detected:
            # Full image detection
            corners, ids, _ = self.detector.detectMarkers(img)
            if not all_markers_detected and self.previous_markers:
                self.get_logger().debug(f"Frame {self.frame_count}: Full detection (missing markers: {self.required_marker_ids - detected_marker_ids})")
            else:
                self.get_logger().debug(f"Frame {self.frame_count}: Full detection (periodic/initial)")
        else:
            # ROI-based detection (only when all required markers are detected)
            roi_mask = self.create_roi_mask(img.shape)
            
            # Apply mask to image for detection
            masked_img = cv2.bitwise_and(img, img, mask=roi_mask)
            corners, ids, _ = self.detector.detectMarkers(masked_img)
            self.get_logger().debug(f"Frame {self.frame_count}: ROI detection (all markers present)")
        
        return corners, ids

    def update_marker_positions(self, corners, ids):
        """Update stored marker positions for next frame ROI"""
        if ids is not None:
            current_markers = {}
            for i, marker_id in enumerate(ids):
                marker_id_val = int(marker_id[0])
                
                # Calculate marker center from corners
                corner_points = corners[i][0]  # Get the 4 corner points
                center_x = np.mean(corner_points[:, 0])
                center_y = np.mean(corner_points[:, 1])
                current_markers[marker_id_val] = (center_x, center_y)
            
            # Update previous markers for next frame
            self.previous_markers = current_markers

    def publish_aruco_pose_simple(self, marker_id, center_x, center_y, camera_angle_rad):
        """Publish ArUco marker pose using simple center position and X-axis orientation"""
        try:
            # Convert camera pixel position to world coordinates
            camera_pixel_pos = [center_x, center_y]
            world_meter_pos = convert_camera_pixel_to_world_meter(camera_pixel_pos)
            
            world_x = float(world_meter_pos[0])
            world_y = float(world_meter_pos[1])
            world_z = 0.0  # Robots are on the ground plane
            
            # Convert camera angle to world angle
            world_angle_rad = convert_camera_angle_to_world_angle(camera_angle_rad)
            
            # Create Odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "world"
            odom_msg.child_frame_id = "base_link"
            
            # Set position
            odom_msg.pose.pose.position.x = world_x
            odom_msg.pose.pose.position.y = world_y
            odom_msg.pose.pose.position.z = world_z
            
            # Create quaternion for pure Z-axis rotation (yaw only)
            qx = 0.0
            qy = 0.0
            qz = np.sin(world_angle_rad / 2.0)
            qw = np.cos(world_angle_rad / 2.0)
            
            # Set orientation
            odom_msg.pose.pose.orientation.x = float(qx)
            odom_msg.pose.pose.orientation.y = float(qy)
            odom_msg.pose.pose.orientation.z = float(qz)
            odom_msg.pose.pose.orientation.w = float(qw)
            
            # Set twist to zero
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0
            
            # Publish based on marker ID
            success = False
            try:
                if marker_id == 0:
                    self.robot0_publisher.publish(odom_msg)
                    # Special logging for robot 0
                    camera_pixel_x = int(round(center_x))
                    camera_pixel_y = int(round(center_y))
                    yaw_deg = np.degrees(world_angle_rad)
                    print(f"[ROBOT0_POSITION] === Position in different frames ===")
                    print(f"[ROBOT0_POSITION] World_meter: x={world_x:.4f}m, y={world_y:.4f}m")
                    print(f"[ROBOT0_POSITION] Camera_pixel: x={camera_pixel_x}px, y={camera_pixel_y}px (image: 1100x600)")
                    print(f"[ROBOT0_POSITION] Orientation: {world_angle_rad:.4f} rad = {yaw_deg:.2f} deg")
                    print(f"[ROBOT0_POSITION] =====================================")
                    success = True
                elif marker_id == 1:
                    self.robot1_publisher.publish(odom_msg)
                    success = True
                elif marker_id == 2:
                    self.robot2_publisher.publish(odom_msg)
                    success = True
                elif marker_id == 5:
                    self.parcel0_publisher.publish(odom_msg)
                    success = True
                elif marker_id == 6:
                    self.parcel1_publisher.publish(odom_msg)
                    success = True
                elif marker_id == 7:
                    self.parcel2_publisher.publish(odom_msg)
                    success = True
                else:
                    self.get_logger().warn(f"Unknown marker ID for pose publishing: {marker_id}")
                    return
                    
            except Exception as publish_error:
                self.get_logger().error(f"Failed to publish pose for marker {marker_id}: {publish_error}")
                
        except Exception as e:
            self.get_logger().error(f"Error in publish_aruco_pose_simple for marker {marker_id}: {e}")

    def capture_and_publish(self):
        # Check if camera is available before trying to read
        if not self.cap.isOpened():
            self.get_logger().error("Camera not opened")
            return
            
        ret, imgrgb = self.cap.read()
        if not ret:
            self.get_logger().warn("Can't receive frame from camera")
            return
            
        # ==================================================================================
        # CRITICAL: ALL DETECTION AND POSITION OUTPUT OPERATIONS BELOW USE CROPPED IMAGE
        # ==================================================================================
        # Undistort and crop image - ALL SUBSEQUENT PROCESSING USES THIS CROPPED IMAGE
        img = cv2.remap(imgrgb, self.map1_r, self.map2_r, interpolation=cv2.INTER_LINEAR)
        img = img[65:665, 50:1150]  # Crop to region of interest
        
        # Create a clean copy of the image for publishing (this is the priority)
        clean_img = img.copy()
        
        # Publish the clean processed image via ROS2 topic (before ArUco detection)
        try:
            ros_image = self.bridge.cv2_to_imgmsg(clean_img, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_frame"
            self.image_publisher.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")
            return

        # ==================================================================================
        # ARUCO DETECTION AND POSE ESTIMATION - ALL USING CROPPED IMAGE COORDINATES
        # ==================================================================================

        # ArUco detection with ROI tracking optimization
        corners, ids = self.detect_markers_with_roi(img)
        
        # Filter markers to only process those with ID < 15
        if ids is not None:
            filtered_corners = []
            filtered_ids = []
            
            for i, marker_id in enumerate(ids):
                marker_id_val = int(marker_id[0])
                if marker_id_val < 15:  # Only process markers with ID < 15
                    filtered_corners.append(corners[i])
                    filtered_ids.append(marker_id)
            
            # Update corners and ids with filtered results
            if filtered_corners:
                corners = filtered_corners
                ids = np.array(filtered_ids).reshape(-1, 1)
                
                # Update marker positions for next frame ROI tracking
                self.update_marker_positions(corners, ids)
            else:
                corners = None
                ids = None
        
        if ids is not None:
            # Draw all detected markers with ID < 15
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            
            # Simple pose estimation using marker center and X-axis direction
            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])
                
                # Get the 4 corner points of the marker
                corner_points = corner[0]  # Shape: (4, 2) - four corners with (x, y)
                
                # Calculate marker center as the average of all corners
                center_x = np.mean(corner_points[:, 0])
                center_y = np.mean(corner_points[:, 1])
                
                # Calculate orientation using Y-axis of ArUco marker
                # Y-axis points from top edge to bottom edge (corner 0->3 and corner 1->2)
                # Use the left edge (corner 3 to corner 0) as Y-axis direction (opposite direction)
                y_axis_vector = corner_points[0] - corner_points[3]  # Top - Bottom (left edge, reversed)
                
                # Calculate angle of Y-axis in camera pixel coordinates
                camera_angle_rad = np.arctan2(y_axis_vector[1], y_axis_vector[0])
                
                try:
                    # Publish pose for specific marker IDs (robots: 0,1,2 and parcels: 5,6,7)
                    if marker_id in [0, 1, 2, 5, 6, 7]:
                        self.publish_aruco_pose_simple(marker_id, center_x, center_y, camera_angle_rad)
                        
                    # Draw marker center and orientation arrow
                    center_int = (int(center_x), int(center_y))
                    cv2.circle(img, center_int, 5, (0, 255, 0), -1)  # Green center dot
                    
                    # Draw orientation arrow (Y-axis direction)
                    arrow_length = 30
                    end_x = int(center_x + arrow_length * np.cos(camera_angle_rad))
                    end_y = int(center_y + arrow_length * np.sin(camera_angle_rad))
                    cv2.arrowedLine(img, center_int, (end_x, end_y), (255, 0, 0), 2)  # Blue arrow
                    
                    # Draw marker ID
                    cv2.putText(img, f"ID{marker_id}", 
                               (int(center_x-10), int(center_y-20)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                except Exception as e:
                    self.get_logger().error(f"Error processing marker {marker_id}: {e}")
        
        # Optional: Draw ROI regions for debugging
        if self.show_roi_debug and self.previous_markers:
            for marker_id, (prev_x, prev_y) in self.previous_markers.items():
                x1 = max(0, int(prev_x - self.roi_expansion))
                y1 = max(0, int(prev_y - self.roi_expansion))
                x2 = min(img.shape[1], int(prev_x + self.roi_expansion))
                y2 = min(img.shape[0], int(prev_y + self.roi_expansion))
                
                # Draw ROI rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow rectangle
                cv2.putText(img, f"ROI_{marker_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show frame locally (optional - can be disabled for headless operation)
        cv2.imshow('Camera Feed', img)
        cv2.waitKey(1)

    def check_system_health(self):
        """Periodic health check for pose publishing system"""
        try:
            # Report active tracking status
            active_markers = list(self.previous_markers.keys())
            if active_markers:
                self.get_logger().info(f"Actively tracking markers: {sorted(active_markers)}")
            else:
                self.get_logger().info("No markers currently detected")
                
        except Exception as e:
            self.get_logger().error(f"Error in system health check: {e}")
    
    def destroy_node(self):
        """Enhanced cleanup with robust error handling"""
        try:
            self.get_logger().info("Shutting down camera node...")
            
            # Cancel timers
            if hasattr(self, 'timer'):
                self.timer.cancel()
            if hasattr(self, 'health_timer'):
                self.health_timer.cancel()
            
            # Release camera
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
                self.get_logger().info("Camera released")
            
            # Destroy OpenCV windows
            cv2.destroyAllWindows()
            
            # Log final statistics
            if hasattr(self, 'previous_markers') and self.previous_markers:
                self.get_logger().info(f"Final tracked markers: {list(self.previous_markers.keys())}")
            
        except Exception as e:
            self.get_logger().error(f"Error during node destruction: {e}")
        finally:
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        camera_node = CameraNode()
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'camera_node' in locals():
            camera_node.destroy_node()
        rclpy.shutdown()
        print("Camera node shutdown complete")

if __name__ == "__main__":
    main()