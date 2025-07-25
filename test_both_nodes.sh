#!/bin/bash

# Source the workspace
cd /root/workspace
source install/setup.bash

echo "Testing Camera Node and Rectangle Detector using direct Python execution..."
echo ""

# Start camera node in background
echo "Starting Camera Node..."
python3 build/camera_mocap_launcher/camera_mocap_launcher/camera_node.py &
CAMERA_PID=$!

# Wait a moment for camera to start
sleep 3

# Start rectangle detector
echo "Starting Rectangle Detector..."
python3 build/camera_mocap_launcher/camera_mocap_launcher/rectangle_detector.py &
DETECTOR_PID=$!

echo ""
echo "Both nodes are running!"
echo "Camera Node PID: $CAMERA_PID"
echo "Rectangle Detector PID: $DETECTOR_PID"
echo ""
echo "The camera node publishes on /camera/processed_image"
echo "The rectangle detector subscribes to /camera/processed_image"
echo ""
echo "Press Ctrl+C to stop both nodes"

# Wait for interrupt
trap "echo 'Stopping nodes...'; kill $CAMERA_PID $DETECTOR_PID; exit" INT
wait
