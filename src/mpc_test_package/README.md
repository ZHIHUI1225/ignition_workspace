# MPC Test Package

This ROS2 package tests the MobileRobotMPC class by subscribing to robot pose data and publishing velocity commands.

## Features

- Subscribes to `/robot0/pose` or `/turtlebot0/odom_map` for robot state
- Publishes velocity commands to `/robot0/cmd_vel`
- Loads trajectory data from `tb0_Trajectory.json`
- Publishes reference and predicted trajectories for visualization
- Uses Model Predictive Control (MPC) for trajectory following

## File Structure

```
mpc_test_package/
├── mpc_test_package/
│   ├── __init__.py
│   ├── mobile_robot_mpc.py      # MPC controller class
│   └── mpc_test_node.py         # Main test node
├── launch/
│   └── mpc_test.launch.py       # Launch file
├── package.xml
├── setup.py
└── README.md
```

## Build and Run

1. **Build the package:**
   ```bash
   cd /root/workspace
   colcon build --packages-select mpc_test_package
   source install/setup.bash
   ```

2. **Run the test node:**
   ```bash
   # Option 1: Direct execution
   ros2 run mpc_test_package mpc_test_node
   
   # Option 2: Using launch file
   ros2 launch mpc_test_package mpc_test.launch.py
   ```

3. **Monitor the topics:**
   ```bash
   # Check velocity commands being published
   ros2 topic echo /robot0/cmd_vel
   
   # Check reference trajectory
   ros2 topic echo /robot0/reference_path
   
   # Check predicted trajectory
   ros2 topic echo /robot0/predicted_path
   ```

## Configuration

The MPC controller uses the following parameters (configurable in `mobile_robot_mpc.py`):

- **Prediction horizon (N):** 8 steps
- **Control horizon (N_c):** 3 steps
- **Time step (dt):** 0.5 seconds
- **Max linear velocity:** 0.2 m/s
- **Max angular velocity:** π/3 rad/s

## Trajectory Loading

The node attempts to load trajectory data from these locations (in order):
1. `/root/workspace/data/experi/tb0_Trajectory.json`
2. `/root/workspace/data/simple_maze/tb0_Trajectory.json`
3. `/root/workspace/data/simulation/tb0_Trajectory.json`
4. `/root/workspace/data/tb0_Trajectory.json`

If no trajectory file is found, it creates a simple circular test trajectory.

## Visualization

The node publishes the following topics for visualization in RViz:
- `/robot0/reference_path` - Reference trajectory path
- `/robot0/predicted_path` - MPC predicted trajectory

## Dependencies

- rclpy
- geometry_msgs
- nav_msgs
- tf_transformations
- numpy
- casadi (for MPC optimization)

## Troubleshooting

1. **No pose data:** Ensure the robot simulation is running and publishing to `/robot0/pose` or `/turtlebot0/odom_map`
2. **MPC solve failures:** Check casadi installation and solver configuration
3. **No trajectory file:** Verify trajectory files exist in the expected locations
