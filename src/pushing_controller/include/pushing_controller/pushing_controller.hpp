#pragma once
// general deps
#include "codegen/mpc.h"

#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <geometry_msgs/msg/detail/polygon__struct.hpp>
#include <geometry_msgs/msg/detail/pose__struct.hpp>
#include <memory>
#include <optional>
#include <pushing_interfaces/msg/pushing_action.hpp>
#include <pushing_interfaces/msg/pushing_config.hpp>
#include <variant>
// ros deps
#include "rclcpp_action/rclcpp_action.hpp"

#include <geometry_msgs/msg/detail/twist__struct.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pushing_interfaces/action/apply_push.hpp>
#include <pushing_interfaces/action/detail/move_robot_to__struct.hpp>
#include <pushing_interfaces/action/move_robot_to.hpp>
#include <pushing_interfaces/srv/set_pushing_env.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/server_goal_handle.hpp>
#include <rclcpp_action/types.hpp>
#include <std_srvs/srv/detail/trigger__struct.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class PushingController : public rclcpp::Node
{
public:
  struct TrackedPush {
    pushing_interfaces::msg::PushingAction action;
    size_t trajectory_counter;
  };
  PushingController();
  using SetPushingEnv = pushing_interfaces::srv::SetPushingEnv;
  using SetPushingEnvReq = pushing_interfaces::srv::SetPushingEnv_Request;
  using SetPushingEnvRes = pushing_interfaces::srv::SetPushingEnv_Response;
  using MoveRobotTo = pushing_interfaces::action::MoveRobotTo;
  using GoalHandleMoveRobotTo = rclcpp_action::ServerGoalHandle<MoveRobotTo>;
  using GoalHandleMoveRobotToSharedPtr = std::shared_ptr<GoalHandleMoveRobotTo>;
  using ApplyPush = pushing_interfaces::action::ApplyPush;
  using GoalHandleApplyPush = rclcpp_action::ServerGoalHandle<ApplyPush>;
  using GoalHandleApplyPushSharedPtr = std::shared_ptr<GoalHandleApplyPush>;

private:
  void declareRosParameters();
  rcl_interfaces::msg::SetParametersResult parametersCallback([[maybe_unused]] const std::vector<rclcpp::Parameter>& parameters);
  void setWeights();
  void initVariables();
  void setObstacles();
  void mainLoop();
  void onRobotPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void onObjectPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void onPause(const std_srvs::srv::SetBool_Request::SharedPtr req, const std_srvs::srv::SetBool_Response::SharedPtr res);
  void onSetEnv(const SetPushingEnvReq::SharedPtr req, const SetPushingEnvRes::SharedPtr res);
  rclcpp_action::GoalResponse handleMoveRobotToGoal(const rclcpp_action::GoalUUID& uuid, std::shared_ptr<const MoveRobotTo::Goal> goal);
  rclcpp_action::CancelResponse handleMoveRobotToCancel(const GoalHandleMoveRobotToSharedPtr goal);
  void handleMoveRobotToAccepted(const GoalHandleMoveRobotToSharedPtr goal);
  rclcpp_action::GoalResponse handleApplyPushGoal(const rclcpp_action::GoalUUID& uuid, std::shared_ptr<const ApplyPush::Goal> goal);
  rclcpp_action::CancelResponse handleApplyPushCancel(const GoalHandleApplyPushSharedPtr goal);
  void handleApplyPushAccepted(const GoalHandleApplyPushSharedPtr goal);
  geometry_msgs::msg::Twist robotControl(const GoalHandleMoveRobotToSharedPtr goal_handle);
  geometry_msgs::msg::Twist pushingControl(const GoalHandleApplyPushSharedPtr goal_handle);
  void publishPrediction(real_T* xopt);
  void publishReference();
  void publishObstacles();
  void setPushingMeasuredDisturbances();
  void updateObstaclesMPC();
  // ROS
  OnSetParametersCallbackHandle::SharedPtr parameters_ch_;
  rclcpp::TimerBase::SharedPtr main_timer_;                                                /**< Timer for the main loop */
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr robot_pose_sub_;        /**< subscription to robot pose */
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr object_pose_sub_;       /**< subscription to object pose */
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;              /**< publisher for velocity commands */
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr reference_publisher_;                  /**< publisher for reference path */
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr prediction_publisher_;                 /**< publisher for predicted path */
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_publisher_; /**< publisher for predicted path */
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr pause_service_;                       /**< service to pause the execution */
  rclcpp::Service<SetPushingEnv>::SharedPtr env_service_;                                  /**< service to pause the execution */
  rclcpp_action::Server<MoveRobotTo>::SharedPtr move_robot_action_server_;                 /**< action server for robot only movement */
  rclcpp_action::Server<ApplyPush>::SharedPtr apply_push_action_server_; /**< action server for pushing along a trajectory */
  // environment
  std::vector<geometry_msgs::msg::Pose> obstacles_;
  std::vector<pushing_interfaces::msg::PushingConfig> pushing_configurations_;
  geometry_msgs::msg::Polygon robot_footprint_;
  geometry_msgs::msg::Polygon object_footprint_;
  // execution
  std::optional<std::variant<GoalHandleMoveRobotToSharedPtr, GoalHandleApplyPushSharedPtr>> goal_handle_;
  bool paused_;
  TrackedPush pushing_trajectory_;
  // state
  geometry_msgs::msg::Pose object_pose;
  geometry_msgs::msg::Pose robot_pose;
  // MPC
  real_T robot_state[5];
  real_T object_state[3];
  real_T robot_meas_dist[7 * P_HOR];
  real_T robot_old_input[2 * (P_HOR + 1)];
  struct4_T robot_state_data;
  struct5_T robot_online_data;
  real_T robot_vel_cmd[2];
  real_T robot_manipulated_variable[2];
  real_T robot_input_sequence[2 * (P_HOR + 1)];
  struct11_T robot_param;
  struct12_T robot_mpc_info;
  real_T robot_iter;
  real_T robot_state_prediction[5 * (P_HOR + 1)];
  // pushing MPC
  real_T pushing_state[3];
  real_T pushing_meas_dist[5 * (P_HOR + 1)];
  real_T pushing_old_input[2 * (P_HOR + 1)];
  struct14_T pushing_state_data;
  struct15_T pushing_online_data;
  real_T pushing_vel_cmd[2];
  real_T pushing_manupilated_variable[2];
  real_T pushing_input_sequence[2 * (P_HOR + 1)];
  struct20_T pushing_param;
  struct21_T pushing_mpc_info;
  real_T pushing_iter;
  real_T pushing_predicted_state[3 * (P_HOR + 1)];
};
