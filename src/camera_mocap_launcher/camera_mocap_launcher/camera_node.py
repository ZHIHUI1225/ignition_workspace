#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

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
        
        # 定义不同大小的ArUco标记的3D坐标（单位：米）
        # Robot markers (IDs 0, 1, 2): 5cm × 5cm
        robot_marker_size = 0.05 # 5cm
        robot_half_size = robot_marker_size / 2
        self.robot_marker_points_3d = np.array([
            [-robot_half_size, robot_half_size, 0],   # 左上角
            [robot_half_size, robot_half_size, 0],    # 右上角
            [robot_half_size, -robot_half_size, 0],   # 右下角
            [-robot_half_size, -robot_half_size, 0]   # 左下角
        ], dtype=np.float32)
        
        # Parcel markers (IDs 5, 6, 7): 5cm × 5cm
        parcel_marker_size = 0.05  # 5cm
        parcel_half_size = parcel_marker_size / 2
        self.parcel_marker_points_3d = np.array([
            [-parcel_half_size, parcel_half_size, 0],   # 左上角
            [parcel_half_size, parcel_half_size, 0],    # 右上角
            [parcel_half_size, -parcel_half_size, 0],   # 右下角
            [-parcel_half_size, -parcel_half_size, 0]   # 左下角
        ], dtype=np.float32)
        
        # 像素到米的转换因子
        self.pixel_to_meter_x = 0.00234
        self.pixel_to_meter_y = 0.00232

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
        
        # Robust pose publishing enhancements
        self.previous_poses = {}  # Store previous valid poses for smoothing {marker_id: (rvec, tvec, timestamp)}
        self.pose_smoothing_factor = 0.3  # Alpha for exponential smoothing (0=no smoothing, 1=no history)
        self.max_pose_distance = 0.5  # Maximum allowed pose jump in meters
        self.max_pose_age = 1.0  # Maximum age of previous pose for fallback (seconds)
        self.pose_publish_failures = {}  # Track publishing failures {marker_id: failure_count}
        self.max_publish_failures = 5  # Maximum consecutive failures before logging error
        
        # Pose validation thresholds
        self.min_reprojection_error = 0.0  # Minimum acceptable reprojection error
        self.max_reprojection_error = 10.0  # Maximum acceptable reprojection error (pixels)
        self.min_marker_area = 100  # Minimum marker area in pixels
        self.max_z_distance = 2.0  # Maximum allowed Z distance in meters
        
        # Log robust pose publishing configuration after variables are initialized
        self.get_logger().info("=== Robust Pose Publishing Configuration ===")
        self.get_logger().info(f"Pose smoothing factor: {self.pose_smoothing_factor}")
        self.get_logger().info(f"Max pose distance jump: {self.max_pose_distance}m")
        self.get_logger().info(f"Max pose age for fallback: {self.max_pose_age}s")
        self.get_logger().info(f"Max reprojection error: {self.max_reprojection_error}px")
        self.get_logger().info(f"Min marker area: {self.min_marker_area}px")
        self.get_logger().info(f"Max Z distance: {self.max_z_distance}m")
        self.get_logger().info(f"Target markers: {sorted(self.required_marker_ids)}")
        self.get_logger().info("=== End Configuration ===")

    def create_pose_message(self, rvec, tvec, frame_id="camera_frame"):
        """Convert rotation and translation vectors to Odometry message with validation"""
        try:
            # Input validation
            if rvec is None or tvec is None:
                self.get_logger().warn("Invalid pose vectors: rvec or tvec is None")
                return None
                
            if not isinstance(rvec, np.ndarray) or not isinstance(tvec, np.ndarray):
                self.get_logger().warn("Invalid pose vectors: not numpy arrays")
                return None
                
            if rvec.shape != (3, 1) or tvec.shape != (3, 1):
                self.get_logger().warn(f"Invalid pose vector shapes: rvec={rvec.shape}, tvec={tvec.shape}")
                return None
            
            # Check for NaN or infinite values
            if np.any(np.isnan(rvec)) or np.any(np.isinf(rvec)) or np.any(np.isnan(tvec)) or np.any(np.isinf(tvec)):
                self.get_logger().warn("Invalid pose vectors: contains NaN or infinite values")
                return None
            
            # Validate translation vector magnitude (reasonable bounds)
            translation_magnitude = np.linalg.norm(tvec)
            if translation_magnitude > self.max_z_distance or translation_magnitude < 0.01:
                self.get_logger().warn(f"Invalid translation magnitude: {translation_magnitude:.3f}m")
                return None
                
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "world"  # Use world frame instead of camera frame
            odom_msg.child_frame_id = "base_link"
            
            # Transform from camera coordinates to world coordinates
            # Camera coordinate system: Origin at left-upper corner, X-right, Y-down (pixels)
            # World coordinate system: Origin at left-lower corner, X-right, Y-up (meters)
            # Conversion factor: pixels to meters = multiply by 0.0023
            
            # Get camera pose in meters (solvePnP returns in meters based on marker size)
            camera_x_m = float(tvec[0][0])  # Camera X in meters (right from camera center)
            camera_y_m = float(tvec[1][0])  # Camera Y in meters (down from camera center) 
            camera_z_m = float(tvec[2][0])  # Camera Z in meters (distance from camera)
            
            # Convert camera pose to pixel coordinates first
            # Since camera is overhead, we need to project the 3D position to 2D pixel coordinates
            # This is a simplified projection assuming camera is directly above
            camera_x_px = camera_x_m / 0.0023  # Convert meters to pixels
            camera_y_px = camera_y_m / 0.0023  # Convert meters to pixels
            
            # Get image dimensions for coordinate transformation
            img_height = 600  # Cropped image height (665-65=600)
            img_width = 1100   # Cropped image width (1150-50=1100)
            
            # Transform from camera pixel coordinates to world coordinates
            # Camera origin: left-upper corner, World origin: left-lower corner
            # Camera X-right → World X-right (same direction)
            # Camera Y-down → World Y-up (flip and offset)
            
            # Add image center offset since camera pose is relative to camera center
            center_x_px = img_width / 2
            center_y_px = img_height / 2
            
            # World coordinates in pixels (relative to left-lower corner)
            world_x_px = center_x_px + camera_x_px  # Add camera center offset
            world_y_px = img_height - (center_y_px + camera_y_px)  # Flip Y axis and add offset
            
            # Convert to world coordinates in meters
            world_x = world_x_px * 0.0023  # Convert pixels to meters
            world_y = world_y_px * 0.0023  # Convert pixels to meters
            world_z = 0.0  # Robots are on the ground plane
            
            # Set transformed position
            odom_msg.pose.pose.position.x = world_x
            odom_msg.pose.pose.position.y = world_y
            odom_msg.pose.pose.position.z = world_z
            
            # Convert rotation vector to rotation matrix, then to quaternion
            try:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
            except cv2.error as e:
                self.get_logger().warn(f"Failed to convert rotation vector: {e}")
                return None
            
            # Validate rotation matrix (should be orthogonal)
            det = np.linalg.det(rotation_matrix)
            if abs(det - 1.0) > 0.1:  # Allow some tolerance
                self.get_logger().warn(f"Invalid rotation matrix determinant: {det}")
                return None
            
            # Transform rotation from camera frame to world frame
            # Camera frame: X-right, Y-down, Z-forward
            # World frame: X-right, Y-forward, Z-up
            # For overhead camera looking down: need to transform the rotation
            
            # Create transformation matrix from camera frame to world frame
            # Rotation around X-axis by 180 degrees to flip Y and Z axes
            camera_to_world_rot = np.array([
                [1,  0,  0],
                [0, -1,  0],  # Flip Y axis
                [0,  0, -1]   # Flip Z axis
            ], dtype=np.float32)
            
            # Transform the marker's rotation matrix to world coordinates
            world_rotation_matrix = camera_to_world_rot @ rotation_matrix
            
            # Extract only the Z-axis rotation (yaw) for ground robots
            # This assumes robots move in 2D plane with only yaw rotation
            # Extract yaw angle from rotation matrix
            yaw = np.arctan2(world_rotation_matrix[1, 0], world_rotation_matrix[0, 0])
            
            # Create quaternion for pure Z-axis rotation (yaw only)
            qx = 0.0
            qy = 0.0
            qz = np.sin(yaw / 2.0)
            qw = np.cos(yaw / 2.0)
            
            # Set orientation (quaternion)
            odom_msg.pose.pose.orientation.x = float(qx)
            odom_msg.pose.pose.orientation.y = float(qy)
            odom_msg.pose.pose.orientation.z = float(qz)
            odom_msg.pose.pose.orientation.w = float(qw)
            
            # Set twist (velocities) to zero since we don't have velocity information from ArUco
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0
            
            return odom_msg
            
        except Exception as e:
            self.get_logger().error(f"Unexpected error in create_pose_message: {e}")
            return None

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

    def validate_pose_quality(self, corners, marker_points_3d, rvec, tvec, camera_matrix=None):
        """Validate pose quality using reprojection error and other metrics"""
        try:
            # Use provided camera matrix or fall back to default
            if camera_matrix is None:
                camera_matrix = self.cameraMatrix_r
                
            # Calculate marker area to check if marker is too small/far
            corner_points = corners[0]  # Get the 4 corner points
            area = cv2.contourArea(corner_points)
            if area < self.min_marker_area:
                return False, f"Marker area too small: {area:.1f} pixels"
            
            # Calculate reprojection error using appropriate camera matrix
            projected_points, _ = cv2.projectPoints(
                marker_points_3d, 
                rvec, 
                tvec, 
                camera_matrix,  # Use the provided camera matrix
                self.distCoeffs_r
            )
            
            # Calculate RMS reprojection error
            reprojection_error = cv2.norm(corner_points, projected_points.reshape(-1, 2), cv2.NORM_L2) / len(projected_points)
            
            if reprojection_error > self.max_reprojection_error:
                return False, f"Reprojection error too high: {reprojection_error:.2f} pixels"
            
            # Check Z distance (too close or too far)
            z_distance = abs(float(tvec[2][0]))
            if z_distance > self.max_z_distance:
                return False, f"Z distance too far: {z_distance:.3f}m"
            
            return True, f"Valid pose (error: {reprojection_error:.2f}px, area: {area:.0f}px, z: {z_distance:.3f}m)"
            
        except Exception as e:
            return False, f"Pose validation error: {e}"

    def smooth_pose(self, marker_id, rvec, tvec):
        """Apply temporal smoothing to reduce pose jitter"""
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
            
            # Check if we have previous pose for this marker
            if marker_id in self.previous_poses:
                prev_rvec, prev_tvec, prev_time = self.previous_poses[marker_id]
                
                # Check if previous pose is not too old
                if (current_time - prev_time) < self.max_pose_age:
                    # Calculate pose distance to detect outliers
                    translation_diff = np.linalg.norm(tvec - prev_tvec)
                    
                    if translation_diff > self.max_pose_distance:
                        self.get_logger().warn(f"Marker {marker_id}: Large pose jump detected ({translation_diff:.3f}m), using previous pose")
                        # Use previous pose if jump is too large (likely outlier)
                        return prev_rvec, prev_tvec
                    
                    # Apply exponential smoothing
                    alpha = self.pose_smoothing_factor
                    smoothed_tvec = alpha * tvec + (1 - alpha) * prev_tvec
                    
                    # For rotation vectors, we need to be more careful due to angle wrapping
                    # Simple linear interpolation for small changes
                    rotation_diff = np.linalg.norm(rvec - prev_rvec)
                    if rotation_diff < np.pi:  # Small rotation change
                        smoothed_rvec = alpha * rvec + (1 - alpha) * prev_rvec
                    else:
                        # Large rotation change, don't smooth to avoid interpolation artifacts
                        smoothed_rvec = rvec
                    
                    # Store smoothed pose
                    self.previous_poses[marker_id] = (smoothed_rvec, smoothed_tvec, current_time)
                    return smoothed_rvec, smoothed_tvec
            
            # No previous pose or too old, use current pose
            self.previous_poses[marker_id] = (rvec, tvec, current_time)
            return rvec, tvec
            
        except Exception as e:
            self.get_logger().error(f"Pose smoothing error for marker {marker_id}: {e}")
            return rvec, tvec  # Return original pose on error

    def publish_aruco_pose(self, marker_id, rvec, tvec):
        """Publish ArUco marker pose based on its ID with robust error handling"""
        try:
            # Apply pose smoothing
            smoothed_rvec, smoothed_tvec = self.smooth_pose(marker_id, rvec, tvec)
            
            # Create pose message with validation
            odom_msg = self.create_pose_message(smoothed_rvec, smoothed_tvec)
            
            if odom_msg is None:
                # Failed to create valid pose message
                self.pose_publish_failures[marker_id] = self.pose_publish_failures.get(marker_id, 0) + 1
                if self.pose_publish_failures[marker_id] <= self.max_publish_failures:
                    self.get_logger().warn(f"Failed to create pose message for marker {marker_id}")
                return
            
            # Reset failure count on successful pose creation
            self.pose_publish_failures[marker_id] = 0
            
            # Publish based on marker ID with error handling
            success = False
            try:
                if marker_id == 0:
                    self.robot0_publisher.publish(odom_msg)
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
                
                if success:
                    # Log successful publication (optional, can be disabled for performance)
                    # position = pose_msg.pose.position
                    # self.get_logger().debug(f"Published pose for marker {marker_id}: ({position.x:.3f}, {position.y:.3f}, {position.z:.3f})")
                    pass
                    
            except Exception as publish_error:
                self.get_logger().error(f"Failed to publish pose for marker {marker_id}: {publish_error}")
                self.pose_publish_failures[marker_id] = self.pose_publish_failures.get(marker_id, 0) + 1
                
        except Exception as e:
            self.get_logger().error(f"Unexpected error in publish_aruco_pose for marker {marker_id}: {e}")
            self.pose_publish_failures[marker_id] = self.pose_publish_failures.get(marker_id, 0) + 1

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
        
        # Adjust camera matrix for cropped image coordinate system
        # The crop removes 50 pixels from left and 65 pixels from top
        adjusted_camera_matrix = self.cameraMatrix_r.copy()
        adjusted_camera_matrix[0, 2] -= 50  # Adjust cx for left crop
        adjusted_camera_matrix[1, 2] -= 65  # Adjust cy for top crop
        
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
            
            # 使用solvePnP进行姿态估计（替代已移除的estimatePoseSingleMarkers）
            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])
                
                # 根据marker ID选择正确的3D点坐标
                if marker_id in [0, 1, 2]:  # Robot markers (3.6cm)
                    marker_points_3d = self.robot_marker_points_3d
                    axis_length = 0.018  # robot_half_size for axis display
                elif marker_id in [5, 6, 7]:  # Parcel markers (5cm)
                    marker_points_3d = self.parcel_marker_points_3d
                    axis_length = 0.025  # parcel_half_size for axis display
                else:
                    # Default to parcel size for unknown markers
                    marker_points_3d = self.parcel_marker_points_3d
                    axis_length = 0.025
                
                # 使用solvePnP估计每个标记的姿态 - 使用调整后的相机矩阵
                flags = cv2.SOLVEPNP_ITERATIVE  # 默认迭代法，稳定推荐
                # 其他选项：SOLVEPNP_IPPE（平面物体）
                try:
                    ret, rvec, tvec = cv2.solvePnP(
                        marker_points_3d, 
                        corner.reshape(-1, 2),  # 确保角点是正确的2D格式
                        adjusted_camera_matrix,  # 使用调整后的相机矩阵
                        self.distCoeffs_r,
                        flags=flags
                    )
                    
                    if ret:
                        # Validate pose quality before publishing (using adjusted matrix)
                        is_valid, validation_msg = self.validate_pose_quality(corner, marker_points_3d, rvec, tvec, adjusted_camera_matrix)
                        
                        if is_valid:
                            # Draw axis for each marker with appropriate length (using adjusted matrix)
                            cv2.drawFrameAxes(img, adjusted_camera_matrix, self.distCoeffs_r, 
                                             rvec, tvec, axis_length)
                            
                            # Publish pose for specific marker IDs (robots: 0,1,2 and parcels: 5,6,7)
                            if marker_id in [0, 1, 2, 5, 6, 7]:
                                self.publish_aruco_pose(marker_id, rvec, tvec)
                                
                                # Log detected markers with validation info (optional)
                                # self.get_logger().debug(f"Marker {marker_id}: {validation_msg}")
                        else:
                            # Log validation failure
                            self.get_logger().debug(f"Marker {marker_id} pose rejected: {validation_msg}")
                            
                            # Still draw the marker detection but without axis
                            cv2.putText(img, f"ID{marker_id}:INVALID", 
                                       (int(corner[0][0][0]), int(corner[0][0][1])-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        self.get_logger().debug(f"solvePnP failed for marker {marker_id}")
                        
                except cv2.error as cv_error:
                    self.get_logger().warn(f"OpenCV error in pose estimation for marker {marker_id}: {cv_error}")
                except Exception as e:
                    self.get_logger().error(f"Unexpected error in pose estimation for marker {marker_id}: {e}")
        
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
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # Check for persistent publishing failures
            for marker_id, failure_count in self.pose_publish_failures.items():
                if failure_count >= self.max_publish_failures:
                    self.get_logger().error(f"Marker {marker_id}: {failure_count} consecutive pose publishing failures")
            
            # Check for stale poses (markers not updated recently)
            stale_markers = []
            for marker_id, (_, _, timestamp) in self.previous_poses.items():
                if (current_time - timestamp) > (self.max_pose_age * 2):  # 2x the max age threshold
                    stale_markers.append(marker_id)
            
            if stale_markers:
                self.get_logger().warn(f"Stale pose data for markers: {stale_markers}")
                # Clean up stale poses
                for marker_id in stale_markers:
                    del self.previous_poses[marker_id]
            
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
            if hasattr(self, 'previous_poses') and self.previous_poses:
                self.get_logger().info(f"Final tracked markers: {list(self.previous_poses.keys())}")
            
            if hasattr(self, 'pose_publish_failures') and any(self.pose_publish_failures.values()):
                self.get_logger().info(f"Total pose publish failures: {dict(self.pose_publish_failures)}")
            
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