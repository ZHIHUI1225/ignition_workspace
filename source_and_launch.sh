#!/bin/bash

# Source ROS2 and workspace
source /opt/ros/humble/setup.bash
source /root/workspace/install/setup.bash

# Export TURTLEBOT3 model (required for turtlebot3 packages)
export TURTLEBOT3_MODEL=burger

# Run the launch command
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
