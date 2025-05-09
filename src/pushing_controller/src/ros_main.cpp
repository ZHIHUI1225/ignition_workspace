/**
 * @file ros_main.cpp
 * @author Damien Six (damien@robotsix.net)
 * @brief Main for the ROS2 executable
 * @date 2023-03-30
 * @copyright Copyright (c) 2023 Technology Innovation Institute
 */
#include "pushing_controller/pushing_controller.hpp"
#include <memory>

/**
 * Initializes the ROS 2 node and executes the main loop.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 *
 * @returns 0 on successful execution.
 */
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PushingController>();
  rclcpp::spin(node);
  return 0;
}
