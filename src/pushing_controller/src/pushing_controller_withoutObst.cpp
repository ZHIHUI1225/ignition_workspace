#include "codegen/mpc.h"

#include <Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <memory>
#include <pushing_controller/pushing_controller.hpp>
#include <pushing_controller/utils.hpp>
#include <pushing_interfaces/action/apply_push.hpp>
#include <pushing_interfaces/srv/set_pushing_env.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/create_server.hpp>
#include <rclcpp_action/server.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <type_traits>
#include <unistd.h>

PushingController::PushingController() : rclcpp::Node("pushing_controller")
{
  using namespace std::placeholders;

  // parameters
  declareRosParameters();
  // initialize variables
  initVariables();
  parameters_ch_ =
      add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& parameters) { return parametersCallback(parameters); });
  // subscriptions
  robot_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "pushing/robot_pose", 1, [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { onRobotPose(msg); });
  object_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "pushing/object_pose", 1, [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { onObjectPose(msg); });
  // publishers
  cmd_vel_publisher_ = create_publisher<geometry_msgs::msg::Twist>("pushing/cmd_vel", 1);
  reference_publisher_ = create_publisher<nav_msgs::msg::Path>("pushing/ref_path", 1);
  prediction_publisher_ = create_publisher<nav_msgs::msg::Path>("pushing/pred_path", 1);
  // services
  pause_service_ = create_service<std_srvs::srv::SetBool>(
      "pushing/pause", [this](const std_srvs::srv::SetBool_Request::SharedPtr req, const std_srvs::srv::SetBool_Response::SharedPtr res) {
        onPause(req, res);
      });
  env_service_ = create_service<pushing_interfaces::srv::SetPushingEnv>(
      "pushing/controller_set_env", [this](const pushing_interfaces::srv::SetPushingEnv_Request::SharedPtr req,
                                            const pushing_interfaces::srv::SetPushingEnv_Response::SharedPtr res) { onSetEnv(req, res); });
  // action servers
  move_robot_action_server_ = rclcpp_action::create_server<MoveRobotTo>(
      this, "pushing/move_robot_to", std::bind(&PushingController::handleMoveRobotToGoal, this, _1, _2),
      std::bind(&PushingController::handleMoveRobotToCancel, this, _1), std::bind(&PushingController::handleMoveRobotToAccepted, this, _1));
  apply_push_action_server_ = rclcpp_action::create_server<ApplyPush>(
      this, "pushing/apply_push", std::bind(&PushingController::handleApplyPushGoal, this, _1, _2),
      std::bind(&PushingController::handleApplyPushCancel, this, _1), std::bind(&PushingController::handleApplyPushAccepted, this, _1));
  // timer
  main_timer_ = rclcpp::create_timer(get_node_base_interface(), get_node_timers_interface(), get_clock(),
                                     rclcpp::Duration::from_seconds(0.1), [this]() { mainLoop(); });
}

void PushingController::initVariables()
{
  // Initialize variables
  argInit_5x1_real_T(robot_state);
  argInit_3x1_real_T(object_state);
  argInit_26x7_real_T(robot_meas_dist);
  argInit_26x2_real_T(robot_old_input);
  argInit_struct4_T(&robot_state_data);
  argInit_struct5_T(&robot_online_data);
  argInit_2x1_real_T(robot_vel_cmd);
  argInit_2x1_real_T(robot_manipulated_variable);
  argInit_26x2_real_T(robot_input_sequence);
  argInit_struct11_T(&robot_param);
  // default robot param values
  robot_param.x_or = argInit_real_T();
  robot_param.y_or = argInit_real_T();
  robot_param.L = 0.053 / 2;;
  robot_param.R =  0.041 / 2;
  robot_param.constraints = 0;
  robot_param.J_o = 0.1;
  robot_param.J_r = 0.033;
  robot_param.M_r = 0.216;
  robot_param.M_o = 0.152;
  robot_param.mu_o = 0.3;
  robot_param.mu_i = 0.7;
  robot_param.s_o = 0.1;
  robot_param.gamma_u = 0.7;
  robot_param.gamma_l = -0.7;
  robot_param.safe_d = 0.15;
  robot_vel_cmd[0] = 0.0;
  robot_vel_cmd[1] = 0.0;
  // Object MPC param
  argInit_3x1_real_T(pushing_state);
  argInit_26x5_real_T(pushing_meas_dist);
  argInit_26x2_real_T(pushing_old_input);
  argInit_struct14_T(&pushing_state_data);
  argInit_struct15_T(&pushing_online_data);
  argInit_2x1_real_T(pushing_vel_cmd);
  argInit_2x1_real_T(pushing_manupilated_variable);
  argInit_26x2_real_T(pushing_input_sequence);
  argInit_struct20_T(&pushing_param);
  pushing_param.x_or = argInit_real_T();
  pushing_param.y_or = argInit_real_T();
  pushing_param.L = 0.053 / 2;
  pushing_param.R =  0.041 / 2;
  pushing_param.constraints = 0;
  pushing_param.Rlim_p = 0.7;
  pushing_param.Rlim_m = -0.7;
  pushing_param.safe_d = 0.15;
  pushing_param.W_lim = 2 * M_PI;
  pushing_vel_cmd[0] = 0.0;
  pushing_vel_cmd[1] = 0.0;
  setWeights();
}

void PushingController::declareRosParameters()
{
  // general parameters
  declare_parameter<double>("min_turning_radius", 0.47);
  declare_parameter<double>("safe_distance", 0.15);
  // Robot MPC Parameters
  declare_parameter<double>("robot.distance_success_threshold", 0.02);
  declare_parameter<double>("robot.x_gain", 1.0);
  declare_parameter<double>("robot.y_gain", 1.0);
  declare_parameter<double>("robot.theta_gain", 0.1);
  declare_parameter<double>("robot.v_gain", 0.01);
  declare_parameter<double>("robot.omega_gain", 0.01);
  declare_parameter<double>("robot.P_mult", 10.0);
  declare_parameter<double>("robot.lin_acc_gain", 0.05);
  declare_parameter<double>("robot.ang_acc_gain", 0.05);
  declare_parameter<double>("robot.lin_acc_rate_gain", 0.01);
  declare_parameter<double>("robot.ang_acc_rate_gain", 0.01);
  // pushing mpc
  declare_parameter<double>("pushing.x_gain", 1.0);
  declare_parameter<double>("pushing.y_gain", 1.0);
  declare_parameter<double>("pushing.theta_gain", 0.1);
  declare_parameter<double>("pushing.lin_vel_gain", 0.1);
  declare_parameter<double>("pushing.ang_vel_gain", 0.1);
  declare_parameter<double>("pushing.P_mult", 50.0);
  declare_parameter<double>("pushing.lin_vel_rate_gain", 1.0);
  declare_parameter<double>("pushing.ang_vel_rate_gain", 1.0);
}

void PushingController::setWeights()
{
  double min_turning_radius = get_parameter("min_turning_radius").as_double();
  double safe_distance = get_parameter("safe_distance").as_double();
  double robot_x_gain = get_parameter("robot.x_gain").as_double();
  double robot_y_gain = get_parameter("robot.y_gain").as_double();
  double robot_theta_gain = get_parameter("robot.theta_gain").as_double();
  double robot_v_gain = get_parameter("robot.v_gain").as_double();
  double robot_omega_gain = get_parameter("robot.omega_gain").as_double();
  double robot_P_mult = get_parameter("robot.P_mult").as_double();
  double robot_lin_acc_gain = get_parameter("robot.lin_acc_gain").as_double();
  double robot_ang_acc_gain = get_parameter("robot.ang_acc_gain").as_double();
  double robot_lin_acc_rate_gain = get_parameter("robot.lin_acc_rate_gain").as_double();
  double robot_ang_acc_rate_gain = get_parameter("robot.ang_acc_rate_gain").as_double();
  double pushing_x_gain = get_parameter("pushing.x_gain").as_double();
  double pushing_y_gain = get_parameter("pushing.y_gain").as_double();
  double pushing_theta_gain = get_parameter("pushing.theta_gain").as_double();
  double pushing_lin_vel_gain = get_parameter("pushing.lin_vel_gain").as_double();
  double pushing_ang_vel_gain = get_parameter("pushing.ang_vel_gain").as_double();
  double pushing_P_mult = get_parameter("pushing.P_mult").as_double();
  double pushing_lin_vel_rate_gain = get_parameter("pushing.lin_vel_rate_gain").as_double();
  double pushing_ang_vel_rate_gain = get_parameter("pushing.ang_vel_rate_gain").as_double();

  for (int i = 0; i < P_HOR; i++) {
    // output weights
    robot_online_data.weights.y[i] = robot_x_gain;
    robot_online_data.weights.y[i + P_HOR] = robot_y_gain;
    robot_online_data.weights.y[i + 2 * P_HOR] = robot_theta_gain;
    robot_online_data.weights.y[i + 3 * P_HOR] = robot_v_gain;
    robot_online_data.weights.y[i + 4 * P_HOR] = robot_omega_gain;
    pushing_online_data.weights.y[i] = pushing_x_gain;
    pushing_online_data.weights.y[i + P_HOR] = pushing_y_gain;
    pushing_online_data.weights.y[i + 2 * P_HOR] = pushing_theta_gain;
    // input weights
    robot_online_data.weights.u[i] = robot_lin_acc_gain;
    robot_online_data.weights.u[i + P_HOR] = robot_ang_acc_gain;
    pushing_online_data.weights.u[i] = pushing_lin_vel_gain;
    pushing_online_data.weights.u[i + P_HOR] = pushing_ang_vel_gain;
    // input rate weights
    robot_online_data.weights.du[i] = robot_lin_acc_rate_gain;
    robot_online_data.weights.du[i + P_HOR] = robot_ang_acc_rate_gain;
    pushing_online_data.weights.du[i] = pushing_lin_vel_rate_gain;
    pushing_online_data.weights.du[i + P_HOR] = pushing_ang_vel_rate_gain;
  }

  robot_online_data.weights.y[P_HOR - 1] = robot_P_mult * robot_x_gain;
  robot_online_data.weights.y[2 * P_HOR - 1] = robot_P_mult * robot_y_gain;
  robot_online_data.weights.y[3 * P_HOR - 1] = robot_P_mult * robot_theta_gain;
  robot_online_data.weights.y[4 * P_HOR - 1] = robot_P_mult * robot_v_gain;
  robot_online_data.weights.y[5 * P_HOR] = robot_P_mult * robot_omega_gain;

  pushing_online_data.weights.y[P_HOR - 1] = pushing_P_mult * pushing_x_gain;
  pushing_online_data.weights.y[2 * P_HOR - 1] = pushing_P_mult * pushing_y_gain;
  pushing_online_data.weights.y[3 * P_HOR - 1] = pushing_P_mult * pushing_theta_gain;

  robot_param.safe_d = safe_distance;
  robot_param.gamma_u = min_turning_radius;
  robot_param.gamma_l = -min_turning_radius;
  pushing_param.safe_d = safe_distance;
  pushing_param.Rlim_m = -min_turning_radius;
  pushing_param.Rlim_p = min_turning_radius;
}

rcl_interfaces::msg::SetParametersResult
    PushingController::parametersCallback([[maybe_unused]] const std::vector<rclcpp::Parameter>& parameters)
{
  setWeights();
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  return result;
}

void PushingController::mainLoop()
{
  geometry_msgs::msg::Twist cmd_msg;
  if (paused_ || !goal_handle_.has_value()) {
    cmd_msg.linear.x = 0.0;
    cmd_msg.angular.z = 0.0;
  } else {
    // set state variable
    robot_state[0] = robot_pose.position.x;
    robot_state[1] = robot_pose.position.y;
    robot_state[2] = tf2::getYaw(robot_pose.orientation);
    robot_state[3] = robot_vel_cmd[0];
    robot_state[4] = robot_vel_cmd[1];
    object_state[0] = object_pose.position.x;
    object_state[1] = object_pose.position.y;
    object_state[2] = tf2::getYaw(object_pose.orientation);
    pushing_state[0] = object_pose.position.x;
    pushing_state[1] = object_pose.position.y;
    pushing_state[2] = robot_state[2];
    cmd_msg = std::visit(
        [this](auto&& gh) -> geometry_msgs::msg::Twist {
          using T = std::decay_t<decltype(gh)>;
          if constexpr (std::is_same_v<T, GoalHandleMoveRobotToSharedPtr>) {
            return robotControl(gh);
          } else if constexpr (std::is_same_v<T, GoalHandleApplyPushSharedPtr>) {
            return pushingControl(gh);
          } else
            static_assert(always_false_v<T>, "non-exhaustive visitor!");
        },
        goal_handle_.value());
  }
  cmd_vel_publisher_->publish(cmd_msg);
  return;
}

geometry_msgs::msg::Twist PushingController::robotControl(const GoalHandleMoveRobotToSharedPtr goal_handle)
{
  const auto goal = goal_handle->get_goal();
  auto feedback = std::make_shared<MoveRobotTo::Feedback>();
  auto result = std::make_shared<MoveRobotTo::Result>();
  result->distance = pushing_utils::computeDistance(goal->target, robot_pose,
                                                    { get_parameter("robot.x_gain").as_double(), get_parameter("robot.y_gain").as_double(),
                                                      get_parameter("robot.theta_gain").as_double() });
  result->success = result->distance < get_parameter("robot.distance_success_threshold").as_double();
  // Check if there is a cancel request
  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal canceled");
    goal_handle_ = std::nullopt;
    return geometry_msgs::msg::Twist();
  }
  // check if we reached goal
  if (result->success && goal_handle->is_active()) {
    goal_handle->succeed(result);
    RCLCPP_INFO(this->get_logger(), "Target Reached");
  }
  geometry_msgs::msg::Twist cmd_msg;
  cmd_msg.linear.x = robot_vel_cmd[0];
  cmd_msg.angular.z = robot_vel_cmd[1];

  robot_param.x_or = cos(robot_state[2]) * (object_state[0] - robot_state[0]) + sin(robot_state[2]) * (object_state[1] - robot_state[1]);
  robot_param.y_or = -sin(robot_state[2]) * (object_state[0] - robot_state[0]) + cos(robot_state[2]) * (object_state[1] - robot_state[1]);
  pushMove(robot_state, robot_meas_dist, robot_input_sequence, (&robot_state_data), (&robot_online_data), (&robot_param), robot_vel_cmd,
           robot_manipulated_variable, robot_input_sequence, (&robot_state_data), (&robot_mpc_info), (&robot_iter), robot_state_prediction);
  publishPrediction(robot_state_prediction);
  return cmd_msg;
}

geometry_msgs::msg::Twist PushingController::pushingControl(const GoalHandleApplyPushSharedPtr goal_handle)
{
  const auto goal = goal_handle->get_goal();
  auto feedback = std::make_shared<ApplyPush::Feedback>();
  auto result = std::make_shared<ApplyPush::Result>();
  double distance =
      pushing_utils::computeDistance(goal->action.trajectory.back().pose, object_pose,
                                     { get_parameter("pushing.x_gain").as_double(), get_parameter("pushing.y_gain").as_double(),
                                       get_parameter("pushing.theta_gain").as_double() });
  result->success = distance < get_parameter("robot.distance_success_threshold").as_double() ? true : false;
  // Check if there is a cancel request
  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal canceled");
    goal_handle_ = std::nullopt;
    return geometry_msgs::msg::Twist();
  }
  // check if we reached goal
  if (result->success && goal_handle->is_active()) {
    goal_handle->succeed(result);
    RCLCPP_INFO(this->get_logger(), "Target Reached");
  }
  setPushingMeasuredDisturbances();
  pushing_param.x_or = cos(robot_state[2]) * (object_state[0] - robot_state[0]) + sin(robot_state[2]) * (object_state[1] - robot_state[1]);
  pushing_param.y_or = -sin(robot_state[2]) * (object_state[0] - robot_state[0]) + cos(robot_state[2]) * (object_state[1] - robot_state[1]);
  // pushMoveObj(pushing_state, pushing_meas_dist, pushing_input_sequence, (&pushing_state_data), (&pushing_online_data), (&pushing_param),
  //             pushing_vel_cmd, pushing_manupilated_variable, pushing_input_sequence, (&pushing_state_data), (&pushing_mpc_info),
  //             (&pushing_iter), pushing_predicted_state);
  // publishPrediction(pushing_predicted_state);
  pushMove(robot_state, pushing_meas_dist,  pushing_input_sequence, (&robot_state_data), (&robot_online_data),(&robot_param), pushing_vel_cmd,
  pushing_manupilated_variable, pushing_input_sequence, (&robot_state_data), (&robot_mpc_info),(&pushing_iter), pushing_predicted_state);
  publishPrediction(pushing_predicted_state);
  publishReference();

  geometry_msgs::msg::Twist cmd_msg;
  cmd_msg.linear.x = pushing_vel_cmd[0];
  cmd_msg.angular.z = pushing_vel_cmd[1];
  pushing_trajectory_.trajectory_counter++;
  return cmd_msg;
}

void PushingController::setPushingMeasuredDisturbances()
{
  size_t trajectory_length = pushing_trajectory_.action.trajectory.size();

  for (size_t j = 0; j < P_HOR + 1; ++j) {
    size_t index = pushing_trajectory_.trajectory_counter + j;
    geometry_msgs::msg::Pose pose;
    geometry_msgs::msg::Twist velocity;
    if (index < trajectory_length - 1) {
      pose = pushing_trajectory_.action.trajectory.at(index).pose;
      velocity = pushing_trajectory_.action.trajectory.at(index).twist;
    } else {
      pose = pushing_trajectory_.action.trajectory.at(trajectory_length - 1).pose;
      velocity.linear.x = 0.0;
      velocity.angular.z = 0.0;
    }

    pushing_meas_dist[j] = pose.position.x;
    pushing_meas_dist[j + P_HOR + 1] = pose.position.y;
    // pushing_meas_dist[j + 2 * (P_HOR + 1)] = tf2::getYaw(pose.orientation); // TODO consider this is object yaw not robot
    pushing_meas_dist[j + 2 * (P_HOR + 1)] = pose.orientation.z;
    pushing_meas_dist[j + 3 * (P_HOR + 1)] = velocity.linear.x;
    pushing_meas_dist[j + 4 * (P_HOR + 1)] = velocity.angular.z;
  }
}

void PushingController::publishReference()
{
  nav_msgs::msg::Path reference_path;
  reference_path.header.frame_id = "world";
  reference_path.header.stamp = this->now();
  
  size_t trajectory_length = pushing_trajectory_.action.trajectory.size();
  // size_t end_index = std::min(pushing_trajectory_.trajectory_counter + P_HOR + 1, trajectory_length);
  size_t end_index = trajectory_length;
  for (size_t i = pushing_trajectory_.trajectory_counter; i < end_index; ++i) {
    geometry_msgs::msg::PoseStamped p;
    p.header = pushing_trajectory_.action.trajectory[i].header;
    p.pose = pushing_trajectory_.action.trajectory[i].pose;
    reference_path.poses.push_back(p);
  }
  
  reference_publisher_->publish(reference_path);
}

void PushingController::onRobotPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  // TODO implement kalman filter
  robot_pose = msg->pose;
}

void PushingController::onObjectPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  // TODO implement kalman filter
  object_pose = msg->pose;
}

void PushingController::onPause(const std_srvs::srv::SetBool_Request::SharedPtr req, const std_srvs::srv::SetBool_Response::SharedPtr res)
{
  if (paused_ == req->data) {
    res->message = paused_ ? "Already paused" : "Already executing";
    res->success = false;
  } else {
    paused_ = req->data;
    res->message = paused_ ? "Execution paused" : "Execution resumed";
    res->success = false;
  }
}

void PushingController::onSetEnv(const SetPushingEnvReq::SharedPtr req, const SetPushingEnvRes::SharedPtr res)
{
  // save pushing configuration
  pushing_configurations_ = req->contact_configurations;

  // save footprints
  robot_footprint_ = req->robot_footprint;
  object_footprint_ = req->object_footprint;
  // return success
  res->success = true;
}

rclcpp_action::GoalResponse PushingController::handleMoveRobotToGoal([[maybe_unused]] const rclcpp_action::GoalUUID& uuid,
                                                                     std::shared_ptr<const MoveRobotTo::Goal> goal)
{
  RCLCPP_INFO(this->get_logger(), "Received request to move robot to x:%f y:%f yaw:%f", goal->target.position.x, goal->target.position.y,
              tf2::getYaw(goal->target.orientation));
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse PushingController::handleMoveRobotToCancel(const std::shared_ptr<GoalHandleMoveRobotTo> goal_handle)
{
  RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
  if (!goal_handle_.has_value()) {
    RCLCPP_WARN(this->get_logger(), "No existing goal: rejecting cancel request");
    return rclcpp_action::CancelResponse::REJECT;
  } else {
    auto curr_goal_uuid = std::visit(
        [](auto&& gh) -> auto {
          using T = std::decay_t<decltype(gh)>;
          if constexpr (std::is_same_v<T, GoalHandleMoveRobotToSharedPtr>) {
            return gh->get_goal_id();
          } else if constexpr (std::is_same_v<T, GoalHandleApplyPushSharedPtr>) {
            return gh->get_goal_id();
          } else
            static_assert(always_false_v<T>, "non-exhaustive visitor!");
        },
        goal_handle_.value());
    if (goal_handle->get_goal_id() != curr_goal_uuid) {
      RCLCPP_WARN(this->get_logger(), "Requesting cancelation of different goal");
      return rclcpp_action::CancelResponse::REJECT;
    } else {
      RCLCPP_INFO(this->get_logger(), "Canceling current goal");
      goal_handle_ = std::nullopt;
      return rclcpp_action::CancelResponse::ACCEPT;
    }
  }
}

void PushingController::handleMoveRobotToAccepted(const std::shared_ptr<GoalHandleMoveRobotTo> goal_handle)
{
  argInit_26x2_real_T(robot_old_input);
  argInit_26x2_real_T(robot_input_sequence);
  argInit_struct4_T(&robot_state_data);
  auto target = goal_handle->get_goal()->target;
  double theta = tf2::getYaw(target.orientation);
  RCLCPP_INFO(get_logger(), "Moving to x: %f y: %f theta: %f", target.position.x, target.position.y, theta);
  argInit_26x7_real_T(robot_meas_dist);
  for (int j = 0; j <= P_HOR; ++j) {
    robot_meas_dist[j] = target.position.x;
    robot_meas_dist[j + P_HOR + 1] = target.position.y;
    robot_meas_dist[j + 2 * (P_HOR + 1)] = theta;
  }
  if (goal_handle_.has_value()) {
    RCLCPP_WARN(get_logger(), "Aborting current goal");
    std::visit(
        [](auto&& gh) {
          using T = std::decay_t<decltype(gh)>;
          if constexpr (std::is_same_v<T, GoalHandleMoveRobotToSharedPtr>) {
            auto res = std::make_shared<MoveRobotTo::Result>();
            res->success = false;
            gh->abort(res);
          } else if constexpr (std::is_same_v<T, GoalHandleApplyPushSharedPtr>) {
            auto res = std::make_shared<ApplyPush::Result>();
            res->success = false;
            gh->abort(res);
          } else
            static_assert(always_false_v<T>, "non-exhaustive visitor!");
        },
        goal_handle_.value());
  }
  goal_handle_ = goal_handle;
}

rclcpp_action::GoalResponse PushingController::handleApplyPushGoal([[maybe_unused]] const rclcpp_action::GoalUUID& uuid,
                                                                   [[maybe_unused]] std::shared_ptr<const ApplyPush::Goal> goal)
{
  RCLCPP_INFO(this->get_logger(), "Received request to push along trajectory");
  if (goal->action.trajectory.empty()) {
    RCLCPP_WARN(this->get_logger(), "Invalid empty trajectory received");
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse PushingController::handleApplyPushCancel(const std::shared_ptr<GoalHandleApplyPush> goal_handle)
{
  RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
  if (!goal_handle_.has_value()) {
    RCLCPP_WARN(this->get_logger(), "No existing goal: rejecting cancel request");
    return rclcpp_action::CancelResponse::REJECT;
  } else {
    auto curr_goal_uuid = std::visit(
        [](auto&& gh) -> auto {
          using T = std::decay_t<decltype(gh)>;
          if constexpr (std::is_same_v<T, GoalHandleMoveRobotToSharedPtr>) {
            return gh->get_goal_id();
          } else if constexpr (std::is_same_v<T, GoalHandleApplyPushSharedPtr>) {
            return gh->get_goal_id();
          } else
            static_assert(always_false_v<T>, "non-exhaustive visitor!");
        },
        goal_handle_.value());
    if (goal_handle->get_goal_id() != curr_goal_uuid) {
      RCLCPP_WARN(this->get_logger(), "Requesting cancelation of different goal");
      return rclcpp_action::CancelResponse::REJECT;
    } else {
      RCLCPP_INFO(this->get_logger(), "Canceling current goal");
      goal_handle_ = std::nullopt;
      return rclcpp_action::CancelResponse::ACCEPT;
    }
  }
}

void PushingController::handleApplyPushAccepted(const std::shared_ptr<GoalHandleApplyPush> goal_handle)
{
  if (goal_handle_.has_value()) {
    RCLCPP_WARN(get_logger(), "Aborting current goal");
    std::visit(
        [](auto&& gh) {
          using T = std::decay_t<decltype(gh)>;
          if constexpr (std::is_same_v<T, GoalHandleMoveRobotToSharedPtr>) {
            auto res = std::make_shared<MoveRobotTo::Result>();
            res->success = false;
            gh->abort(res);
          } else if constexpr (std::is_same_v<T, GoalHandleApplyPushSharedPtr>) {
            auto res = std::make_shared<ApplyPush::Result>();
            res->success = false;
            gh->abort(res);
          } else
            static_assert(always_false_v<T>, "non-exhaustive visitor!");
        },
        goal_handle_.value());
  }
  pushing_trajectory_.trajectory_counter = 0;
  pushing_trajectory_.action = goal_handle->get_goal()->action;
  goal_handle_ = goal_handle;
  pushing_param.Rlim_p = pushing_trajectory_.action.config.turning_radius;
  pushing_param.Rlim_m = -pushing_trajectory_.action.config.turning_radius;
  RCLCPP_INFO(this->get_logger(), "starting");
}

void PushingController::publishPrediction(real_T* xopt)
{
  nav_msgs::msg::Path path_msg;
  path_msg.header.frame_id = "world";
  geometry_msgs::msg::PoseStamped pose_msg;
  tf2::Quaternion quat_tf;
  for (int i = 0; i < P_HOR + 1; i++) {
    pose_msg.header.frame_id = "world";
    pose_msg.pose.position.x = xopt[i];
    pose_msg.pose.position.y = xopt[i + P_HOR + 1];
    pose_msg.pose.position.z = 0.0;
    quat_tf.setRPY(0.0, 0.0, xopt[i + 2 * (P_HOR + 1)]);
    pose_msg.pose.orientation.x = quat_tf.getX();
    pose_msg.pose.orientation.y = quat_tf.getY();
    pose_msg.pose.orientation.z = quat_tf.getZ();
    pose_msg.pose.orientation.w = quat_tf.getW();
    path_msg.poses.push_back(pose_msg);
  }
  prediction_publisher_->publish(path_msg);
}