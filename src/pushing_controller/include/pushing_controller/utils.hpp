#pragma once

#include <array>
#include <cmath>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/utils.h>
namespace pushing_utils
{
  inline double wrapAngle(const double& angle)
  {
    double wrapped = angle;
    while ((wrapped >= M_PI) || (wrapped <= -M_PI)) {
      if (angle < 0.0) {
        wrapped = fmod(angle - M_PI, 2.0 * M_PI) + M_PI;
      } else {
        wrapped = fmod(angle + M_PI, 2.0 * M_PI) - M_PI;
      }
    }
    return wrapped;
  }

  inline double computeDistance(geometry_msgs::msg::Pose a, geometry_msgs::msg::Pose b, std::array<double, 3> weights)
  {
    double x_diff = a.position.x - b.position.x;
    double y_diff = a.position.y - b.position.y;
    double yaw_diff = wrapAngle(tf2::getYaw(a.orientation) - tf2::getYaw(b.orientation));
    return weights.at(0) * x_diff * x_diff + weights.at(1) * y_diff * y_diff + weights.at(2) * yaw_diff * yaw_diff;
  }
} // namespace pushing_utils
