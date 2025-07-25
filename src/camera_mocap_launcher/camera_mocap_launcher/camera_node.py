#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # Create publisher for processed video
        self.image_publisher = self.create_publisher(Image, '/camera/processed_image', 10)
        self.bridge = CvBridge()
        
        # Camera setup
        self.cap = cv2.VideoCapture('/dev/video0')
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -20)  # 降低曝光度
        self.cap.set(cv2.CAP_PROP_CONTRAST, 28)  # 增加对比度

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
        
        # ArUco setup
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # Create timer for camera capture
        self.timer = self.create_timer(0.033, self.capture_and_publish)  # ~30 FPS
        
        # Setup display window (no mouse tracking needed)
        cv2.namedWindow('Camera Feed')

    def capture_and_publish(self):
        ret, imgrgb = self.cap.read()
        if not ret:
            self.get_logger().error("Can't receive frame from camera")
            return
            
        # Undistort and crop image
        img = cv2.remap(imgrgb, self.map1_r, self.map2_r, interpolation=cv2.INTER_LINEAR)
        img = img[70:660, 95:1197]  # Crop to region of interest
        
        # # 增强图像对比度和亮度
        # alpha = 1.3  # 对比度增强因子 (1.0-3.0)
        # beta = -10   # 亮度调整 (-100-100)
        # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Create a clean copy of the image for publishing
        clean_img = img.copy()
        
        # Publish the clean processed image via ROS2 topic (before ArUco detection)
        try:
            ros_image = self.bridge.cv2_to_imgmsg(clean_img, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_frame"
            self.image_publisher.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")

        # ArUco detection for display (only affects the displayed image, not the published one)
        corners, ids, _ = self.detector.detectMarkers(img)
        
        if ids is not None:
            # Draw all detected markers (not just ID > 10)
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            
            # Estimate pose for all markers
            marker_length = 0.05  # 5cm, adjust to your actual marker size
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, self.cameraMatrix_r, self.distCoeffs_r)
            
            # Draw axis for each marker
            for i in range(len(ids)):
                cv2.drawFrameAxes(img, self.cameraMatrix_r, self.distCoeffs_r, 
                                 rvecs[i], tvecs[i], marker_length/2)
                
                # No text labels will be displayed
                # The marker is already visualized by drawDetectedMarkers and drawFrameAxes
        
        # Show frame locally (optional - can be disabled for headless operation)
        cv2.imshow('Camera Feed', img)
        cv2.waitKey(1)
    
    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
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