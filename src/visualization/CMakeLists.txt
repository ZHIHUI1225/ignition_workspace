cmake_minimum_required(VERSION 3.5)
project(visualization)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)

install(
  PROGRAMS
    visualization/pose_plotter.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
