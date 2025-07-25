#!/bin/bash

cd /root/workspace
source install/setup.bash

echo "=== Testing ROS2 Package Detection ==="
echo "1. Package list (should show camera_mocap_launcher):"
ros2 pkg list | grep camera_mocap_launcher

echo -e "\n2. Package executables (should show camera_node and rectangle_detector):"
ros2 pkg executables camera_mocap_launcher

echo -e "\n3. Package prefix (should show the path):"
ros2 pkg prefix camera_mocap_launcher

echo -e "\n4. Direct executable test (camera_node):"
which camera_node

echo -e "\n5. Manual execution test:"
echo "Testing camera_node..."
timeout 5s ros2 run camera_mocap_launcher camera_node 2>&1 | head -5 || echo "Camera node test failed or timed out"

echo -e "\n6. Python import test:"
python3 -c "
try:
    from camera_mocap_launcher.camera_node import main
    print('✓ camera_node import successful')
except Exception as e:
    print(f'✗ camera_node import failed: {e}')

try:
    from camera_mocap_launcher.rectangle_detector import main
    print('✓ rectangle_detector import successful')
except Exception as e:
    print(f'✗ rectangle_detector import failed: {e}')
"

echo -e "\n7. Files in bin directory:"
ls -la install/camera_mocap_launcher/bin/

echo -e "\n8. Content of camera_node executable:"
head -10 install/camera_mocap_launcher/bin/camera_node
