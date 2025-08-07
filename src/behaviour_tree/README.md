# Multi-Robot Behavior Tree System

This package provides behavior tree functionality for multi-robot coordination with snapshot stream support for PyTrees Viewer integration.

## Overview

DO NOT USE LAUNCH ! WILL CAUSE THREAD 
Launch 文件默认的多线程执行器与行为树节点的并发设计叠加，导致线程数膨胀:
Root Causes of Thread Proliferation:
Excessive Callback Groups: Each behavior is creating its own callback groups instead of using shared ones
Multiple Timers: Each behavior creates its own ROS timers for control loops
Subscription Duplication: Similar subscriptions are created by multiple behaviors
Control Threads: Some behaviors still use dedicated control threads instead of ROS timers
File Watchers: ThreadedNotifier for file watching adds additional threads

The behavior tree system supports two deployment modes:
1. **Multi-Robot Mode**: Individual behavior tree instances for each robot
2. **Centralized Mode**: Single behavior tree coordinating all robots

## Features

- **PyTrees Viewer Integration**: Real-time visualization of behavior tree execution
- **Multi-Robot Support**: Separate behavior tree instances with proper namespacing
- **Snapshot Streams**: Robot-specific snapshot streams for debugging
- **ROS Parameter Support**: Robot ID and namespace configuration

## Launch Files

### 1. Single Robot Behavior Tree
```bash
ros2 launch behaviour_tree single_robot_behavior_tree.launch.py robot_id:=0 robot_namespace:=turtlebot0
```

### 2. Multi-Robot Demo (2 robots)
```bash
ros2 launch behaviour_tree demo_multi_robot.launch.py
```

### 3. Complete Multi-Robot System (5 robots)
```bash
ros2 launch behaviour_tree multi_robot_complete.launch.py
```

### 4. Multi-Robot with Controllers
```bash
ros2 launch behaviour_tree multi_robot_complete.launch.py include_controllers:=true
```

### 5. Centralized Behavior Tree
```bash
ros2 launch behaviour_tree multi_robot_complete.launch.py include_single_behavior_tree:=true include_controllers:=false
```

## PyTrees Viewer Integration

Each robot publishes its behavior tree state to robot-specific topics:

### For Robot 0 (turtlebot0):
- Tree log: `/turtlebot0/tree_log`
- Tree snapshot: `/turtlebot0/tree_snapshot`
- Tree updates: `/turtlebot0/tree_updates`
- Snapshot streams: `/turtlebot0/tree/snapshot_streams`

### Connecting PyTrees Viewer:
```bash
# For robot 0
py-trees-tree-watcher --namespace=/turtlebot0/tree/snapshot_streams

# For robot 1
py-trees-tree-watcher --namespace=/turtlebot1/tree/snapshot_streams

# For centralized tree
py-trees-tree-watcher --namespace=/tree/snapshot_streams
```

## ROS Topics Structure

### Multi-Robot Mode
Each robot has its own namespace:
```
/turtlebot0/
├── tree_log
├── tree_snapshot
├── tree_updates
├── Ready_flag
├── Pushing_flag
├── Pickup_flag
├── PickUpDone
├── cmd_vel
└── odom

/turtlebot1/
├── tree_log
├── tree_snapshot
├── tree_updates
├── Ready_flag
├── Pushing_flag
├── Pickup_flag
├── PickUpDone
├── cmd_vel
└── odom
```

### Centralized Mode
All robots controlled by single tree:
```
/tree_log
/tree_snapshot
/tree_updates
/turtlebot0/Ready_flag
/turtlebot1/Ready_flag
...
```

## Manual Execution

### Run individual behavior tree:
```bash
# Source the workspace
source /root/workspace/install/setup.bash

# Run for specific robot
ros2 run behaviour_tree my_behaviour_tree --ros-args -p robot_id:=0 -p robot_namespace:=turtlebot0
```

### Run demo executables:
```bash
# Snapshot streams demo
ros2 run behaviour_tree snapshot_streams_demo

# Tutorial 4 reproduction
ros2 run behaviour_tree tutorial_four_reproduction
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `robot_id` | int | 0 | Unique robot identifier (0-4) |
| `robot_namespace` | string | "turtlebot0" | Robot namespace for topic isolation |
| `use_sim_time` | bool | true | Use simulation time |

## Behavior Tree Structure

The behavior tree implements a multi-step coordination sequence:
1. **Reset Flags**: Initialize robot state
2. **Wait**: Coordination delay
3. **Approach Object**: Navigate to target
4. **Push Object**: Execute pushing behavior
5. **Return Home**: Navigate back to start position

## Integration with Existing System

The multi-robot behavior trees integrate with:
- **Follow Controllers**: Robot movement control
- **Pickup Controllers**: Object manipulation
- **State Supervision**: Robot state monitoring
- **Parcel Management**: Object tracking

## Troubleshooting

### Common Issues:

1. **No PyTrees Viewer Connection**:
   - Check the snapshot stream namespace
   - Ensure behavior tree is running
   - Verify PyTrees Viewer installation

2. **Robot Parameter Not Found**:
   - Check launch file parameter passing
   - Verify robot_id and robot_namespace values

3. **Topic Remapping Issues**:
   - Verify namespace configuration
   - Check topic names with `ros2 topic list`

4. **Build Errors**:
   ```bash
   cd /root/workspace
   colcon build --packages-select behaviour_tree
   ```

## Examples

### Launch 2 robots with PyTrees Viewer:
```bash
# Terminal 1: Launch behavior trees
ros2 launch behaviour_tree demo_multi_robot.launch.py

# Terminal 2: Watch robot 0
py-trees-tree-watcher --namespace=/turtlebot0/tree/snapshot_streams

# Terminal 3: Watch robot 1  
py-trees-tree-watcher --namespace=/turtlebot1/tree/snapshot_streams
```

### Launch all robots with full system:
```bash
ros2 launch behaviour_tree multi_robot_complete.launch.py include_controllers:=true
```
