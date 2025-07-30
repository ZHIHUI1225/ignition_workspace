# E-Puck Driver for ROS2 Humble

This package provides a ROS2 driver for the e-puck robot with **WiFi communication** using binary protocol.

## Features

- ✅ **WiFi Communication**: Uses e-puck2's WiFi interface with binary protocol  
- ✅ **ROS2 Humble**: Native ROS2 support with modern Python 3
- ✅ **Sensor Integration**: Proximity sensors, accelerometer, gyroscope
- ✅ **Motor Control**: Differential drive motor control
- ✅ **LED Control**: Individual LED control support
- ✅ **Camera Support**: RGB image capture (when available)

## Dependencies

This driver requires the following dependencies:

### System Dependencies
```bash
sudo apt-get update
sudo apt-get install python3-setuptools python3-pil python3-dev
```

## Installation

1. Clone this package into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
git clone <repository_url>
```

2. Build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select epuck_driver_ros2
source install/setup.bash
```

## WiFi Configuration

### Setting up e-puck2 WiFi

1. **Connect e-puck2 to WiFi**: Use the e-puck2's configuration interface to connect to your WiFi network
2. **Find IP Address**: Determine your e-puck's IP address using:
   - Router admin interface
   - Network scanner: `nmap -sn 192.168.1.0/24`
   - e-puck display or configuration interface
3. **Default Port**: e-puck2 WiFi uses port **1000**

### Configuration Files

The driver supports multiple robots with individual config files:

- `config/robot0.yaml` - First robot configuration
- `config/robot1.yaml` - Second robot configuration  
- `config/robot2.yaml` - Third robot configuration

**Example robot0.yaml**:
```yaml
epuck_driver_ros2:
  ros__parameters:
    epuck_address: "192.168.0.164:1000"  # WiFi IP:PORT format
    epuck_name: "epuck_0"
    init_motors: true
    init_camera: false
    init_leds: true
```

## Usage

### Basic Launch

To launch the driver with default configuration:
```bash
ros2 launch epuck_driver_ros2 epuck_driver.launch.py
```

This uses the default configuration file located at `config/epuck_driver.yaml`.

### Launch with Specific Robot Configuration

To launch with a specific robot configuration:
```bash
# Robot 0 (IP: 192.168.0.164:1000)
ros2 launch epuck_driver_ros2 epuck_driver.launch.py config_file:=robot0.yaml

# Robot 1 (IP: 192.168.0.165:1000)  
ros2 launch epuck_driver_ros2 epuck_driver.launch.py config_file:=robot1.yaml

# Robot 2 (IP: 192.168.0.166:1000)
ros2 launch epuck_driver_ros2 epuck_driver.launch.py config_file:=robot2.yaml
```

### Customizing Configuration

To modify robot settings, edit the appropriate YAML file:

```bash
# Edit robot configuration
nano $(ros2 pkg prefix epuck_driver_ros2)/share/epuck_driver_ros2/config/robot0.yaml
```

**Example configuration with camera enabled**:
```yaml
epuck_driver_ros2:
  ros__parameters:
    epuck_address: "192.168.0.164:1000"  # WiFi IP:PORT
    epuck_name: "epuck_0"
    init_motors: true
    init_camera: true     # Enable camera
    init_leds: true
```

## Protocol Details

This driver uses the **e-puck2 WiFi binary protocol** as documented in the [official e-puck documentation](https://www.gctronic.com/doc/index.php?title=e-puck2_PC_side_development#WiFi_2).

### Key Features:
- **Binary Protocol**: 20-byte command packets for efficient communication
- **TCP Communication**: Reliable connection over WiFi
- **Sensor Data**: 104-byte sensor response packets
- **Real-time Control**: Motor commands and LED control

## Topics

### Published Topics

- `/epuck_X/proximity0` to `/epuck_X/proximity7` (sensor_msgs/Range) - Proximity sensor readings
- `/epuck_X/accel` (sensor_msgs/Imu) - Accelerometer data  
- `/epuck_X/odom` (nav_msgs/Odometry) - Odometry based on wheel encoders
- `/epuck_X/camera` (sensor_msgs/Image) - Camera image (if enabled)

### Subscribed Topics

- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands to control the robot

## Parameters

All parameters are defined in the YAML configuration files:

### Parameters

- `epuck_address` (string, required) - **WiFi IP:PORT** address of the e-puck (e.g., "192.168.0.164:1000")  
- `epuck_name` (string, default: "epuck") - Name of the robot
- `xpos`, `ypos`, `theta` (double, default: 0.0) - Initial position and orientation
- `init_motors` (bool, default: true) - Enable motor control
- `init_camera` (bool, default: false) - Enable camera capture
- `init_leds` (bool, default: true) - Enable LED control
- `init_proximity` (bool, default: true) - Enable proximity sensors
- `init_accelerometer` (bool, default: true) - Enable accelerometer

## Robot Configurations

This package includes configuration files for three robots:

| Robot  | IP Address:Port     | Config File   |
|--------|-------------------|---------------|
| robot0 | 192.168.0.164:1000 | robot0.yaml   |
| robot1 | 192.168.0.165:1000 | robot1.yaml   |  
| robot2 | 192.168.0.166:1000 | robot2.yaml   |

## Example: Controlling the Robot

To move the robot forward:
```bash
ros2 topic pub /epuck_0/cmd_vel geometry_msgs/Twist "linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}" --once
```

To rotate the robot:
```bash
ros2 topic pub /epuck_0/cmd_vel geometry_msgs/Twist "linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}" --once
```

To stop the robot:
```bash
ros2 topic pub /epuck_0/cmd_vel geometry_msgs/Twist "linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}" --once
```

## Troubleshooting

### WiFi Connection Issues

1. **Check Network**: Ensure e-puck2 and computer are on the same WiFi network
2. **Verify IP Address**: Confirm the e-puck's IP address is correct
   ```bash
   ping 192.168.0.164  # Test connectivity
   ```
3. **Port Availability**: Check if port 1000 is accessible
   ```bash
   telnet 192.168.0.164 1000
   ```
4. **Robot State**: Ensure e-puck2 WiFi is enabled and not already connected to another client

### Configuration Issues

If the robot doesn't respond:
- Check the `epuck_address` parameter format: `"IP:PORT"`
- Verify the robot's actual IP address on your network
- Ensure the robot is powered on and WiFi is active
- Try different configuration files for different robots

### Debug Mode

Enable debug output by setting debug=True in the configuration or modifying the launch file.

### Import Errors

If you get import errors for the epuck module, make sure you've built the package and sourced the setup file:
```bash
colcon build --packages-select epuck_driver_ros2
source install/setup.bash
```

## License

GPLv3
