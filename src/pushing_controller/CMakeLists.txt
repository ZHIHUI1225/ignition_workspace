cmake_minimum_required(VERSION 3.8)
project(pushing_controller)

set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_STANDARD 20)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# ament dependencies
find_package(ament_cmake REQUIRED)

# general dependencies
find_package(Eigen3 REQUIRED)  # Removed version requirement to use available version 3.3.7
find_package(yaml-cpp REQUIRED)  # Fixed package name from Yaml-cpp to yaml-cpp
# ROS dependencies
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
set(CMAKE_PREFIX_PATH "/root/workspace/install/pushing_interfaces" ${CMAKE_PREFIX_PATH})
find_package(pushing_interfaces REQUIRED)
find_package(tf2 REQUIRED)


include_directories(PUBLIC
    include
)

add_library(estimator
    src/estimator/kalman_lin.cpp
    src/estimator/kalman_ang.cpp
)
target_link_libraries(estimator PUBLIC Eigen3::Eigen)
ament_target_dependencies(estimator PUBLIC
    geometry_msgs
    nav_msgs
    tf2
)
add_library(mpc
    src/codegen/PushingMPClib.cpp
    src/codegen/rt_nonfinite.cpp
    src/codegen/rtGetInf.cpp
    src/codegen/rtGetNaN.cpp
)
target_include_directories(mpc PUBLIC
    include/codegen
)

add_executable(pushing_controller_node
    src/ros_main.cpp
    src/pushing_controller_withoutObst.cpp
)
ament_target_dependencies(pushing_controller_node PUBLIC
    rclcpp
    rclcpp_action
    geometry_msgs
    nav_msgs
    visualization_msgs
    std_srvs
    pushing_interfaces
    tf2
    tf2_ros
    tf2_geometry_msgs
)
target_link_libraries(pushing_controller_node PUBLIC
    yaml-cpp
    mpc
    Eigen3::Eigen
)

install(TARGETS
    pushing_controller_node
    DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
    scripts/Pushing_trajectory_define.py
    DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/Robot_trajectory_follow.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/MPCcontroller.py
  DESTINATION lib/${PROJECT_NAME}
)
# Install the rviz config file to "share/pushing_controller/rviz"
install(
  DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

# Ensure Python package installation
# ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    find_package(ament_cmake_gtest REQUIRED)
    list(APPEND AMENT_LINT_AUTO_EXCLUDE
      ament_cmake_cpplint
      ament_cmake_flake8
    )
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
