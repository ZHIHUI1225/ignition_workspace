#!/usr/bin/env python3
import cv2
import numpy as np
import math
import rclpy
import json
import time
import os
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque

class RectangleDetector(Node):
    def __init__(self):
        super().__init__('rectangle_detector')
        
        # ROS2 setup
        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/processed_image',
            self.image_callback,
            10
        )
        
        # Rectangle tracking
        self.rectangles = {}  # Store current rectangle information
        self.tracked_rectangles = {}  # Store tracked rectangle history
        self.tracking_duration = 5.0  # Track for 5 seconds before saving
        self.tracking_start_time = None
        self.is_tracking = False
        self.history_length = 30  # Number of frames to keep in history
        self.save_path = os.path.join(os.getcwd(), 'rectangle_data.json')
        
        # 存储图像尺寸信息
        self.image_width = 1280  # 默认值
        self.image_height = 720  # 默认值
        
        # Create a timer for saving data
        self.create_timer(0.5, self.check_tracking_status)
        
        self.get_logger().info("Rectangle Detector Started - Subscribing to /camera/processed_image")
        self.get_logger().info("Detecting light blue rectangles in the video stream (minimum size: 100x100)")
        self.get_logger().info(f"Will track rectangles for {self.tracking_duration} seconds before saving to {self.save_path}")
    def image_callback(self, msg):
        """Callback function for receiving images from camera node"""
        try:
            # Convert ROS2 image message to OpenCV format
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 更新图像尺寸
            self.image_height, self.image_width = img.shape[:2]
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
            
        # 创建原始图像的副本，用于显示结果
        display_img = img.copy()
        
        # 检测通用矩形
        rectangles = self.detect_general_rectangles(img, display_img)
        
        # 更新当前检测到的矩形
        self.rectangles = rectangles
        
        # 开始跟踪，如果检测到至少一个矩形并且还没开始跟踪
        if rectangles and not self.is_tracking:
            self.is_tracking = True
            self.tracking_start_time = time.time()
            self.get_logger().info(f"Started tracking rectangles for {self.tracking_duration} seconds")
            # 初始化跟踪历史
            for rect_name, rect_info in rectangles.items():
                self.tracked_rectangles[rect_name] = {
                    'centers': deque(maxlen=self.history_length),
                    'widths': deque(maxlen=self.history_length),
                    'heights': deque(maxlen=self.history_length),
                    'areas': deque(maxlen=self.history_length),
                    'corners': deque(maxlen=self.history_length)
                }
        
        # 更新跟踪历史
        if self.is_tracking and rectangles:
            for rect_name, rect_info in rectangles.items():
                # 如果是新的矩形，添加到跟踪列表
                if rect_name not in self.tracked_rectangles:
                    self.tracked_rectangles[rect_name] = {
                        'centers': deque(maxlen=self.history_length),
                        'widths': deque(maxlen=self.history_length),
                        'heights': deque(maxlen=self.history_length),
                        'areas': deque(maxlen=self.history_length),
                        'corners': deque(maxlen=self.history_length)
                    }
                
                # 更新历史数据
                self.tracked_rectangles[rect_name]['centers'].append(rect_info['center'])
                self.tracked_rectangles[rect_name]['widths'].append(rect_info['width'])
                self.tracked_rectangles[rect_name]['heights'].append(rect_info['height'])
                self.tracked_rectangles[rect_name]['areas'].append(rect_info['area'])
                self.tracked_rectangles[rect_name]['corners'].append(rect_info['corners'])
        
        # Print rectangle information
        if rectangles:
            tracking_status = ""
            if self.is_tracking:
                elapsed = time.time() - self.tracking_start_time
                remaining = max(0, self.tracking_duration - elapsed)
                tracking_status = f" (Tracking: {remaining:.1f}s remaining)"
                
            self.get_logger().info(f"Detected {len(rectangles)} rectangles{tracking_status}:")
            for rect_name, rect_info in rectangles.items():
                self.get_logger().info(f"{rect_name}: Center={rect_info['center']}, "
                                     f"Size={rect_info['width']}x{rect_info['height']}, "
                                     f"Area={rect_info['area']}")
        
        # Add status text
        status_text = f"Rectangles detected: {len(rectangles) if rectangles else 0}"
        if self.is_tracking:
            elapsed = time.time() - self.tracking_start_time
            remaining = max(0, self.tracking_duration - elapsed)
            status_text += f" | Tracking: {remaining:.1f}s remaining"
        
        cv2.putText(display_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the frame with rectangle detection results
        cv2.imshow('Rectangle Detector', display_img)
        cv2.waitKey(1)
    
    def detect_general_rectangles(self, img, display_img):
        """Detect general rectangles in the image without using ArUco markers"""
        # 创建结果字典
        rectangles = {}
        
        # 转换为HSV颜色空间，增强浅蓝色检测
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 定义浅蓝色的HSV范围
        # 浅蓝色HSV范围大约是: H: 90-110, S: 50-255, V: 50-255
        lower_light_blue = np.array([90, 50, 50])
        upper_light_blue = np.array([110, 255, 255])
        
        # 创建掩膜
        mask = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 执行膨胀和腐蚀操作以消除小噪点
        kernel = np.ones((5, 5), np.uint8)
        mask_processed = cv2.dilate(blurred, kernel, iterations=1)
        mask_processed = cv2.erode(mask_processed, kernel, iterations=1)
        
        # 将处理后的掩膜应用到原始图像
        filtered = cv2.bitwise_and(img, img, mask=mask_processed)
        
        # 转换为灰度图像用于轮廓检测
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 处理每个轮廓
        rect_count = 0
        min_area = 500  # 最小矩形面积，过滤掉太小的噪点
        
        # 按照轮廓面积排序，先处理大面积的轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 忽略太小的区域
            if area < min_area:
                continue
                
            # 使用最小外接矩形来提高矩形检测的准确性
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 计算矩形的宽和高
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # 确保宽和高的顺序一致（宽始终大于高）
            if width < height:
                width, height = height, width
                
            # 忽略小于100x100的矩形
            if width < 100 or height < 100:
                continue
                
            # 使用多边形近似来检查是否接近矩形形状
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 检查是否是近似四边形（4个顶点或接近4个顶点）
            if len(approx) >= 4 and len(approx) <= 6:
                
                # 绘制轮廓
                cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                
                # 计算中心点（使用最小外接矩形的中心）
                center = (int(rect[0][0]), int(rect[0][1]))
                cx, cy = center
                
                # 绘制中心点
                cv2.circle(display_img, center, 5, (0, 0, 255), -1)
                
                # 添加到结果字典
                rect_name = f"rect_{rect_count}"
                rectangles[rect_name] = {
                    'center': center,
                    'width': width,
                    'height': height,
                    'area': area,
                    'corners': box.tolist()
                }
                
                # 绘制矩形信息
                cv2.putText(display_img, rect_name, (cx - 20, cy - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                rect_count += 1
        
        return rectangles
        
    # ArUco检测相关的方法已移除
    
    def check_tracking_status(self):
        """检查跟踪状态并在需要时保存数据"""
        if not self.is_tracking:
            return
            
        elapsed = time.time() - self.tracking_start_time
        if elapsed >= self.tracking_duration:
            # 防止重复保存
            self.is_tracking = False
            self.image_subscriber.destroy()  # 取消订阅，不再接收图像
            self.save_rectangle_data()
            # 停止检测并关闭节点
            self.get_logger().info("Data saved. Shutting down rectangle detector...")
            # 使用一个短暂的定时器来安全关闭节点
            self.create_timer(1.0, self.shutdown_node)
    
    def save_rectangle_data(self):
        """计算平均值并保存矩形数据为JSON文件，然后停止检测"""
        if not self.tracked_rectangles:
            self.get_logger().warn("No rectangle data to save")
            return
            
        # 默认保存路径和环境文件路径
        rect_save_path = self.save_path
        enviro_save_path = os.path.join(os.getcwd(), 'data', 'Enviro_experiments.json')
        
        # 确保data文件夹存在
        os.makedirs(os.path.dirname(enviro_save_path), exist_ok=True)
        
        # 计算平均值
        average_data = {}
        polygons = []
        
        for rect_name, rect_history in self.tracked_rectangles.items():
            if not rect_history['centers']:
                continue
                
            # 计算中心点的平均值
            centers = rect_history['centers']
            avg_center_x = sum(c[0] for c in centers) / len(centers)
            avg_center_y = sum(c[1] for c in centers) / len(centers)
            
            # 计算宽度和高度的平均值
            avg_width = sum(rect_history['widths']) / len(rect_history['widths'])
            avg_height = sum(rect_history['heights']) / len(rect_history['heights'])
            avg_area = sum(rect_history['areas']) / len(rect_history['areas'])
            
            # 计算角点的平均值
            corners_list = rect_history['corners']
            if corners_list and len(corners_list) > 0:
                # 每个角点都是一个列表，需要分别计算平均值
                avg_corners = []
                num_corners = len(corners_list[0])
                for i in range(num_corners):
                    avg_x = sum(corners[i][0] for corners in corners_list) / len(corners_list)
                    avg_y = sum(corners[i][1] for corners in corners_list) / len(corners_list)
                    avg_corners.append([float(avg_x), float(avg_y)])
            else:
                avg_corners = []
            
            average_data[rect_name] = {
                'center': [float(avg_center_x), float(avg_center_y)],
                'width': float(avg_width),
                'height': float(avg_height),
                'area': float(avg_area),
                'corners': avg_corners,
                'sample_count': len(centers)
            }
            
            # 为环境文件添加多边形
            if avg_corners:
                polygons.append({
                    'vertices': [[int(point[0]), int(point[1])] for point in avg_corners]
                })
        
        # 添加时间戳
        data_to_save = {
            'timestamp': time.time(),
            'date_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tracking_duration_seconds': self.tracking_duration,
            'rectangles': average_data
        }
        
        # 使用捕获的图像尺寸
        img_width = self.image_width
        img_height = self.image_height
        
        # 创建环境数据结构
        environment_data = {
            'polygons': polygons,
            'coord_bounds': [0, img_width, 0, img_height],
            'width': img_width,
            'height': img_height
        }
        
        # 保存矩形数据为JSON文件
        try:
            with open(rect_save_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            self.get_logger().info(f"Rectangle data saved to {rect_save_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save rectangle data: {e}")
            
        # 保存环境数据为JSON文件
        try:
            with open(enviro_save_path, 'w') as f:
                json.dump(environment_data, f, indent=2)
            self.get_logger().info(f"Environment data saved to {enviro_save_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save environment data: {e}")
    
    def shutdown_node(self):
        """安全关闭节点的方法"""
        self.get_logger().info("Shutting down rectangle detector node...")
        # 关闭所有窗口
        cv2.destroyAllWindows()
        # 使用系统退出命令，这样可以在ROS2环境中安全地退出
        import sys
        sys.exit(0)
    
    def destroy_node(self):
        # 如果正在跟踪，保存当前数据
        if self.is_tracking and self.tracked_rectangles:
            self.save_rectangle_data()
            
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        rectangle_detector = RectangleDetector()
        rclpy.spin(rectangle_detector)
    except KeyboardInterrupt:
        pass
    finally:
        if 'rectangle_detector' in locals():
            rectangle_detector.destroy_node()
        rclpy.shutdown()
        print("Rectangle detector shutdown complete")

if __name__ == "__main__":
    main()
