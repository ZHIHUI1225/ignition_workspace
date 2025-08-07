# How to Run the Behavior Tree with Different Parameters

## Basic Usage

The `simple_tree.py` script now accepts command-line arguments for robot ID and case.

### Default (same as before)
```bash
ros2 run behaviour_tree simple_tree
```
This runs with:
- Robot ID: 0 (robot0)
- Case: 'experi'
- Control DT: 0.5s

### With Custom Robot ID
```bash
ros2 run behaviour_tree simple_tree --robot-id 1
```
This runs robot1 with case 'experi'

### With Custom Case
```bash
ros2 run behaviour_tree simple_tree --case simulation
```
This runs robot0 with case 'simulation'

### With Both Robot ID and Case
```bash
ros2 run behaviour_tree simple_tree --robot-id 2 --case test
```
This runs robot2 with case 'test'

### With All Parameters
```bash
ros2 run behaviour_tree simple_tree --robot-id 3 --case experi --control-dt 0.2
```
This runs robot3 with case 'experi' and control delta time of 0.2 seconds

### Help
```bash
ros2 run behaviour_tree simple_tree --help
```
Shows all available options

## ROS2 Run Commands

The preferred way to run ROS2 nodes is using `ros2 run`:

### Default
```bash
ros2 run behaviour_tree simple_tree
```

### With Custom Robot ID
```bash
ros2 run behaviour_tree simple_tree --robot-id 1
```

### With Custom Case
```bash
ros2 run behaviour_tree simple_tree --case simulation
```

### With Both Robot ID and Case
```bash
ros2 run behaviour_tree simple_tree --robot-id 2 --case test
```

### With All Parameters
```bash
ros2 run behaviour_tree simple_tree --robot-id 3 --case experi --control-dt 0.2
```

## Direct Python Alternative

You can also run directly with Python (useful for development/debugging):
```bash
cd /root/workspace/src/behaviour_tree/behaviour_tree
python3 simple_tree.py --robot-id 1 --case simulation
```

## Multiple Robots

To run multiple robots simultaneously, open different terminals:

Terminal 1:
```bash
ros2 run behaviour_tree simple_tree --robot-id 0 --case experi
```

Terminal 2:
```bash
ros2 run behaviour_tree simple_tree --robot-id 1 --case experi
```

Terminal 3:
```bash
ros2 run behaviour_tree simple_tree --robot-id 2 --case simulation
```

## Launch Files (Advanced)

For running multiple robots with launch files, create a launch file like this:
```bash
ros2 launch behaviour_tree multi_robot.launch.py
```

This would typically handle multiple robot instances automatically.
