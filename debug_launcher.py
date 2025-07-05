#!/usr/bin/env python3
"""
ROS2 Launch File Debug Wrapper
This script helps debug ROS2 launch files by setting up the proper environment
and handling the launch system correctly for debugging.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_ros_environment():
    """Set up ROS2 environment variables"""
    # Source ROS2 setup
    ros_setup = "/opt/ros/humble/setup.bash"
    workspace_setup = "/root/workspace/install/setup.bash"
    
    # Set essential ROS2 environment variables
    os.environ["ROS_DISTRO"] = "humble"
    os.environ["ROS_VERSION"] = "2"
    os.environ["ROS_DOMAIN_ID"] = "0"
    
    # Set paths
    os.environ["AMENT_PREFIX_PATH"] = "/root/workspace/install:/opt/ros/humble"
    os.environ["CMAKE_PREFIX_PATH"] = "/root/workspace/install:/opt/ros/humble"
    
    # Update PYTHONPATH
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    new_paths = [
        "/root/workspace/install/lib/python3.10/site-packages",
        "/opt/ros/humble/lib/python3.10/site-packages"
    ]
    
    all_paths = new_paths + ([current_pythonpath] if current_pythonpath else [])
    os.environ["PYTHONPATH"] = ":".join(all_paths)
    
    # Unbuffered Python output for better debugging
    os.environ["PYTHONUNBUFFERED"] = "1"

def run_launch_file(launch_file_path):
    """Run a ROS2 launch file"""
    print(f"Debug: Checking launch file: {launch_file_path}")
    
    if not Path(launch_file_path).exists():
        print(f"Error: Launch file not found: {launch_file_path}")
        print(f"Debug: Current working directory: {os.getcwd()}")
        print(f"Debug: Absolute path check: {Path(launch_file_path).absolute()}")
        return 1
    
    # Setup environment
    setup_ros_environment()
    
    # Change to workspace directory
    os.chdir("/root/workspace")
    
    # Check if ros2 command is available
    try:
        ros2_check = subprocess.run(["which", "ros2"], capture_output=True, text=True)
        if ros2_check.returncode != 0:
            print("Error: ros2 command not found in PATH")
            print("Debug: Trying to source ROS2 environment...")
            # Try to source ROS2 environment
            source_cmd = "source /opt/ros/humble/setup.bash && which ros2"
            result = subprocess.run(source_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Debug: ros2 found at: {result.stdout.strip()}")
            else:
                print("Error: Could not find ros2 even after sourcing")
                return 1
        else:
            print(f"Debug: ros2 found at: {ros2_check.stdout.strip()}")
    except Exception as e:
        print(f"Debug: Error checking ros2 command: {e}")
    
    # Run the launch file using ros2 launch
    cmd = ["bash", "-c", f"source /opt/ros/humble/setup.bash && source /root/workspace/install/setup.bash 2>/dev/null || true && ros2 launch {launch_file_path}"]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print(f"ROS_DISTRO: {os.environ.get('ROS_DISTRO')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=False)
        print(f"Debug: Process completed with return code: {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running launch file: {e}")
        print(f"Debug: Return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nLaunch interrupted by user")
        return 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: debug_launcher.py <launch_file_path>")
        print("Example: debug_launcher.py /root/workspace/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/launch/launch_simple_test.launch.py")
        print(f"Debug: Received {len(sys.argv)} arguments: {sys.argv}")
        sys.exit(1)
    
    launch_file = sys.argv[1]
    print(f"Debug: Starting debug launcher with file: {launch_file}")
    
    try:
        exit_code = run_launch_file(launch_file)
        print(f"Debug: Launch completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        print(f"Debug: Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
