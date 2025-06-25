#!/usr/bin/env python3
"""
Debug script to monitor robot topic subscriptions and callback activity.
This helps diagnose why the third robot misses topic data when switching to PushObject.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import time
import threading
from collections import defaultdict
import json


class DebugSubscriptionMonitor(Node):
    """Monitor robot topic subscriptions and callback activity"""
    
    def __init__(self):
        super().__init__('debug_subscription_monitor')
        
        # Track callback activity for each robot
        self.callback_stats = defaultdict(lambda: {
            'robot_odom': {'count': 0, 'last_time': 0},
            'parcel_pose': {'count': 0, 'last_time': 0}
        })
        
        # Create separate callback groups for each robot to simulate BT behavior
        self.robot_callback_groups = {}
        for i in range(3):
            robot_ns = f'turtlebot{i}'
            self.robot_callback_groups[robot_ns] = {
                'control': MutuallyExclusiveCallbackGroup(),
                'pose': ReentrantCallbackGroup(),
                'sensor': ReentrantCallbackGroup(),
                'misc': MutuallyExclusiveCallbackGroup()
            }
        
        # Create subscriptions for each robot
        self.robot_subs = {}
        self.parcel_subs = {}
        
        for i in range(3):
            robot_ns = f'turtlebot{i}'
            
            # Robot odometry subscription
            robot_topic = f'/{robot_ns}/odom_map'
            self.robot_subs[robot_ns] = self.create_subscription(
                Odometry,
                robot_topic,
                lambda msg, ns=robot_ns: self.robot_callback(msg, ns),
                10,
                callback_group=self.robot_callback_groups[robot_ns]['pose']
            )
            
            # Parcel pose subscription (assume robot starts with parcel 0)
            parcel_topic = f'/parcel{i}/pose'
            self.parcel_subs[robot_ns] = self.create_subscription(
                PoseStamped,
                parcel_topic,
                lambda msg, ns=robot_ns: self.parcel_callback(msg, ns),
                10,
                callback_group=self.robot_callback_groups[robot_ns]['sensor']
            )
            
            print(f"[DEBUG] Created subscriptions for {robot_ns}:")
            print(f"  - Robot odom: {robot_topic}")
            print(f"  - Parcel pose: {parcel_topic}")
            print(f"  - Callback groups: {[id(cg) for cg in self.robot_callback_groups[robot_ns].values()]}")
        
        # Start monitoring timer
        self.monitor_timer = self.create_timer(2.0, self.print_stats)
        
        # Start callback group stress test
        self.stress_test_timer = self.create_timer(0.1, self.stress_test_callbacks)
        self.stress_counter = 0
        
    def robot_callback(self, msg, robot_ns):
        """Monitor robot odometry callbacks"""
        current_time = time.time()
        self.callback_stats[robot_ns]['robot_odom']['count'] += 1
        self.callback_stats[robot_ns]['robot_odom']['last_time'] = current_time
        
        # Simulate processing delay occasionally
        if self.callback_stats[robot_ns]['robot_odom']['count'] % 50 == 0:
            time.sleep(0.01)  # 10ms delay to simulate processing
        
    def parcel_callback(self, msg, robot_ns):
        """Monitor parcel pose callbacks"""
        current_time = time.time()
        self.callback_stats[robot_ns]['parcel_pose']['count'] += 1
        self.callback_stats[robot_ns]['parcel_pose']['last_time'] = current_time
        
    def stress_test_callbacks(self):
        """Stress test callback groups by creating temporary subscriptions"""
        self.stress_counter += 1
        
        # Every 50 iterations (5 seconds), simulate a robot switching to PushObject
        if self.stress_counter % 50 == 0:
            robot_id = (self.stress_counter // 50) % 3
            robot_ns = f'turtlebot{robot_id}'
            
            print(f"[STRESS TEST] Simulating {robot_ns} switching to PushObject...")
            
            # Create temporary subscription similar to PushObject initialization
            temp_topic = f'/{robot_ns}/temp_test'
            try:
                temp_sub = self.create_subscription(
                    Odometry,
                    f'/{robot_ns}/odom_map',
                    lambda msg: None,  # Empty callback
                    10,
                    callback_group=self.robot_callback_groups[robot_ns]['control']
                )
                
                # Destroy after short delay
                def cleanup():
                    time.sleep(0.1)
                    self.destroy_subscription(temp_sub)
                
                cleanup_thread = threading.Thread(target=cleanup)
                cleanup_thread.start()
                
            except Exception as e:
                print(f"[STRESS TEST] ERROR creating temp subscription for {robot_ns}: {e}")
    
    def print_stats(self):
        """Print callback statistics"""
        current_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"CALLBACK ACTIVITY MONITOR - {time.strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        
        for robot_ns in ['turtlebot0', 'turtlebot1', 'turtlebot2']:
            robot_stats = self.callback_stats[robot_ns]
            
            # Calculate time since last callback
            robot_last = robot_stats['robot_odom']['last_time']
            parcel_last = robot_stats['parcel_pose']['last_time']
            
            robot_age = current_time - robot_last if robot_last > 0 else 999
            parcel_age = current_time - parcel_last if parcel_last > 0 else 999
            
            # Status indicators
            robot_status = "🟢" if robot_age < 2.0 else "🔴"
            parcel_status = "🟢" if parcel_age < 2.0 else "🔴"
            
            print(f"{robot_ns}:")
            print(f"  Robot odom: {robot_status} {robot_stats['robot_odom']['count']} calls (last: {robot_age:.1f}s ago)")
            print(f"  Parcel pose: {parcel_status} {robot_stats['parcel_pose']['count']} calls (last: {parcel_age:.1f}s ago)")
            
            # Detect potential issues
            if robot_age > 5.0 and robot_stats['robot_odom']['count'] > 0:
                print(f"  ⚠️  WARNING: {robot_ns} robot odom callbacks stopped!")
            
            if parcel_age > 5.0 and robot_stats['parcel_pose']['count'] > 0:
                print(f"  ⚠️  WARNING: {robot_ns} parcel pose callbacks stopped!")
        
        print(f"\nCallback Groups Status:")
        for robot_ns in ['turtlebot0', 'turtlebot1', 'turtlebot2']:
            groups = self.robot_callback_groups[robot_ns]
            print(f"  {robot_ns}: Control={id(groups['control'])}, Pose={id(groups['pose'])}, Sensor={id(groups['sensor'])}")


def main():
    """Main function"""
    rclpy.init()
    
    try:
        # Create multi-threaded executor to simulate BT behavior
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=12)
        
        # Create monitor node
        monitor = DebugSubscriptionMonitor()
        executor.add_node(monitor)
        
        print("Starting subscription monitoring...")
        print("Watch for callback activity and potential blocking issues")
        print("Press Ctrl+C to stop")
        
        # Run executor
        executor.spin()
        
    except KeyboardInterrupt:
        print("\nShutting down monitor...")
        
    finally:
        # Cleanup
        try:
            if 'monitor' in locals():
                monitor.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
