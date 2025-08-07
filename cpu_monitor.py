#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import psutil
import csv
import os
import time
import re  # Added for path sanitization
from datetime import datetime
from rcl_interfaces.msg import ParameterDescriptor

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')
        
        # Declare parameters
        self.declare_parameter('target_node', '/robot0/tree_0', 
                              ParameterDescriptor(description='Name of node to monitor'))
        self.declare_parameter('output_dir', '~/ros2_monitor_logs', 
                              ParameterDescriptor(description='Output directory for logs'))
        self.declare_parameter('interval', 1.0, 
                              ParameterDescriptor(description='Monitoring interval in seconds'))

 # Get parameters
        self.target_node = self.get_parameter('target_node').value
        output_dir = os.path.expanduser(self.get_parameter('output_dir').value)
        self.interval = self.get_parameter('interval').value
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize node name for filesystem compatibility
        sanitized_node_name = re.sub(r'[/\\:*?"<>|]', '_', self.target_node.lstrip('/'))
        self.filename = os.path.join(output_dir, f"{sanitized_node_name}_monitor_{timestamp}.csv")
        
        # Initialize CSV file
        with open(self.filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cpu_percent', 'num_threads', 'memory_mb'])
        
        self.get_logger().info(f"Monitoring started for node: {self.target_node}")
        self.get_logger().info(f"Logging to: {self.filename}")
        self.timer = self.create_timer(self.interval, self.monitor_callback)
        self.target_pid = None

    def find_node_process(self):
        """Find PID of target ROS 2 node"""
        # Parse the target node to extract namespace and node name
        if self.target_node.startswith('/'):
            parts = self.target_node[1:].split('/')  # Remove leading slash and split
            if len(parts) >= 2:
                namespace = parts[0]
                node_name = parts[1]
            else:
                namespace = None
                node_name = parts[0] if parts else self.target_node
        else:
            namespace = None
            node_name = self.target_node
            
        ros_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                # Collect all ROS-related processes for debugging
                if ('ros' in proc.info['name'].lower() or 
                    'python' in proc.info['name'].lower() or
                    any('ros' in str(arg).lower() for arg in proc.info['cmdline']) or
                    self.target_node in cmdline):
                    ros_processes.append((proc.info['pid'], proc.info['name'], cmdline))
                
                # Look for ROS 2 node patterns
                is_match = False
                
                # Method 1: Direct node name match in command line
                if self.target_node in cmdline:
                    is_match = True
                
                # Method 2: Check for ROS 2 node arguments pattern
                elif ('__node:=' + node_name in cmdline and 
                      (namespace is None or '__ns:=/' + namespace in cmdline)):
                    is_match = True
                
                # Method 3: Check for simple_tree or ultra_simple_tree executables
                elif (node_name == 'tree_0' and namespace == 'robot0' and 
                      ('simple_tree' in cmdline or 'ultra_simple_tree' in cmdline)):
                    # Prefer the actual executable over the launcher
                    if '/lib/behaviour_tree/simple_tree' in cmdline or '/lib/behaviour_tree/ultra_simple_tree' in cmdline:
                        is_match = True
                        self.get_logger().info(f"Found actual behavior tree executable: PID {proc.info['pid']}")
                    elif 'ros2 run' not in cmdline:  # Only match launcher if no better option
                        is_match = True
                
                if is_match:
                    self.get_logger().info(f"Found target node process: PID {proc.info['pid']}, Name: {proc.info['name']}")
                    return proc.info['pid']
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        
        # If not found, log some ROS processes for debugging
        self.get_logger().info("ROS-related processes found:")
        for pid, name, cmd in ros_processes[:10]:  # Limit to first 10
            self.get_logger().info(f"  PID: {pid}, Name: {name}, CMD: {cmd[:100]}...")
            
        return None

    def monitor_callback(self):
        if not self.target_pid:
            self.target_pid = self.find_node_process()
            if not self.target_pid:
                self.get_logger().warning(f"Node '{self.target_node}' not found. Retrying...")
                return
        
        try:
            process = psutil.Process(self.target_pid)
            cpu_percent = process.cpu_percent()
            num_threads = process.num_threads()
            memory_mb = process.memory_info().rss / (1024 ** 2)  # RSS in MB
            timestamp = time.time()
            
            # Log to CSV
            with open(self.filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, cpu_percent, num_threads, memory_mb])
                
            self.get_logger().info(
                f"CPU: {cpu_percent:.1f}% | Threads: {num_threads} | Mem: {memory_mb:.1f} MB",
                throttle_duration_sec=5  # Throttle to avoid spamming console
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.get_logger().error(f"Process {self.target_pid} disappeared")
            self.target_pid = None

def main(args=None):
    rclpy.init(args=args)
    monitor = SystemMonitor()
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()