#!/usr/bin/env python3
"""
System Resource and ROS Topic Monitor
Monitors CPU, memory, threads, and ROS topic data flow to diagnose missing topic data issues.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import psutil
import threading
import time
import json
import os
from datetime import datetime
from collections import defaultdict, deque
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64
import numpy as np


class SystemResourceMonitor:
    """Monitor system resources (CPU, memory, threads, etc.)"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.system_stats = deque(maxlen=100)  # Keep last 100 measurements
        self.monitoring = True
        
    def get_current_stats(self):
        """Get current system resource statistics"""
        try:
            # System-wide stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Current process stats
            process_memory = self.process.memory_info()
            process_cpu = self.process.cpu_percent()
            process_threads = self.process.num_threads()
            process_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            
            # ROS-specific stats
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'num_threads']):
                try:
                    if proc.info['name'] == 'python3' and 'behaviour_tree' in ' '.join(proc.info['cmdline'] or []):
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'threads': proc.info['num_threads'],
                            'cmdline': ' '.join(proc.info['cmdline'][-2:])  # Last 2 args
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            stats = {
                'timestamp': time.time(),
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory.percent,
                'system_memory_available_mb': memory.available / 1024 / 1024,
                'process_cpu_percent': process_cpu,
                'process_memory_mb': process_memory.rss / 1024 / 1024,
                'process_threads': process_threads,
                'process_file_descriptors': process_fds,
                'behavior_tree_processes': python_processes
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting system stats: {e}")
            return None
    
    def start_monitoring(self, interval=1.0):
        """Start continuous monitoring in background thread"""
        def monitor_loop():
            while self.monitoring:
                stats = self.get_current_stats()
                if stats:
                    self.system_stats.append(stats)
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 System resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
    
    def get_recent_stats(self, seconds=10):
        """Get stats from the last N seconds"""
        current_time = time.time()
        recent = [s for s in self.system_stats if current_time - s['timestamp'] <= seconds]
        return recent
    
    def print_summary(self):
        """Print summary of recent resource usage"""
        if not self.system_stats:
            print("No system stats available")
            return
        
        recent = self.get_recent_stats(30)  # Last 30 seconds
        if not recent:
            print("No recent stats available")
            return
        
        # Calculate averages
        avg_cpu = np.mean([s['system_cpu_percent'] for s in recent])
        avg_memory = np.mean([s['system_memory_percent'] for s in recent])
        avg_process_cpu = np.mean([s['process_cpu_percent'] for s in recent])
        avg_process_memory = np.mean([s['process_memory_mb'] for s in recent])
        avg_threads = np.mean([s['process_threads'] for s in recent])
        
        print(f"\n📊 SYSTEM RESOURCE SUMMARY (last 30s, {len(recent)} samples):")
        print(f"   System CPU: {avg_cpu:.1f}% (max: {max(s['system_cpu_percent'] for s in recent):.1f}%)")
        print(f"   System Memory: {avg_memory:.1f}% (available: {recent[-1]['system_memory_available_mb']:.0f}MB)")
        print(f"   Process CPU: {avg_process_cpu:.1f}% (max: {max(s['process_cpu_percent'] for s in recent):.1f}%)")
        print(f"   Process Memory: {avg_process_memory:.1f}MB (max: {max(s['process_memory_mb'] for s in recent):.1f}MB)")
        print(f"   Process Threads: {avg_threads:.1f} (max: {max(s['process_threads'] for s in recent)})")
        print(f"   File Descriptors: {recent[-1]['process_file_descriptors']}")
        
        # Show behavior tree processes
        bt_processes = recent[-1]['behavior_tree_processes']
        if bt_processes:
            print(f"\n🤖 BEHAVIOR TREE PROCESSES ({len(bt_processes)}):")
            for proc in bt_processes:
                print(f"   PID {proc['pid']}: CPU={proc['cpu_percent']:.1f}% MEM={proc['memory_mb']:.1f}MB THREADS={proc['threads']} CMD={proc['cmdline']}")


class RosTopicMonitor(Node):
    """Monitor ROS topic data flow and callback performance"""
    
    def __init__(self):
        super().__init__('system_monitor')
        
        # QoS profiles for different reliability needs
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Topic monitoring data
        self.topic_stats = defaultdict(lambda: {
            'message_count': 0,
            'last_message_time': 0,
            'message_times': deque(maxlen=100),
            'callback_durations': deque(maxlen=50),
            'missed_messages': 0,
            'expected_rate': 0,
            'robot_namespace': ''
        })
        
        # Robot-specific subscriptions
        self.subscriptions = []
        self.monitoring_active = True
        
        # Setup subscriptions for all 3 robots
        self.setup_monitoring_subscriptions()
        
        # Start periodic analysis
        self.analysis_timer = self.create_timer(5.0, self.analyze_topic_performance)
        
        print("🔍 ROS topic monitoring initialized")
    
    def setup_monitoring_subscriptions(self):
        """Setup subscriptions to monitor all robot topics"""
        robots = ['turtlebot0', 'turtlebot1', 'turtlebot2']
        
        for i, robot_ns in enumerate(robots):
            # Robot odometry
            odom_topic = f'/{robot_ns}/odom_map'
            odom_sub = self.create_subscription(
                Odometry, odom_topic, 
                lambda msg, topic=odom_topic, robot=robot_ns: self.odom_callback(msg, topic, robot),
                self.best_effort_qos
            )
            self.subscriptions.append(odom_sub)
            self.topic_stats[odom_topic]['expected_rate'] = 10.0  # Expected ~10Hz
            self.topic_stats[odom_topic]['robot_namespace'] = robot_ns
            
            # Parcel poses (assuming 3 parcels for 3 robots)
            parcel_topic = f'/parcel{i}/pose'
            parcel_sub = self.create_subscription(
                PoseStamped, parcel_topic,
                lambda msg, topic=parcel_topic, robot=robot_ns: self.pose_callback(msg, topic, robot),
                self.reliable_qos
            )
            self.subscriptions.append(parcel_sub)
            self.topic_stats[parcel_topic]['expected_rate'] = 1.0  # Expected ~1Hz
            self.topic_stats[parcel_topic]['robot_namespace'] = robot_ns
            
            # Relay points
            relay_topic = f'/Relaypoint{i+1}/pose'
            relay_sub = self.create_subscription(
                PoseStamped, relay_topic,
                lambda msg, topic=relay_topic, robot=robot_ns: self.pose_callback(msg, topic, robot),
                self.reliable_qos
            )
            self.subscriptions.append(relay_sub)
            self.topic_stats[relay_topic]['expected_rate'] = 0.1  # Expected very low rate (static)
            self.topic_stats[relay_topic]['robot_namespace'] = robot_ns
            
            # Behavior tree status topics
            for status_topic in ['pushing_finished', 'pushing_estimated_time']:
                full_topic = f'/{robot_ns}/{status_topic}'
                if status_topic == 'pushing_finished':
                    status_sub = self.create_subscription(
                        Bool, full_topic,
                        lambda msg, topic=full_topic, robot=robot_ns: self.status_callback(msg, topic, robot, 'bool'),
                        self.reliable_qos
                    )
                else:
                    status_sub = self.create_subscription(
                        Float64, full_topic,
                        lambda msg, topic=full_topic, robot=robot_ns: self.status_callback(msg, topic, robot, 'float64'),
                        self.reliable_qos
                    )
                self.subscriptions.append(status_sub)
                self.topic_stats[full_topic]['expected_rate'] = 1.0  # Expected ~1Hz
                self.topic_stats[full_topic]['robot_namespace'] = robot_ns
        
        print(f"📡 Monitoring {len(self.subscriptions)} ROS topics across {len(robots)} robots")
    
    def odom_callback(self, msg, topic, robot_namespace):
        """Callback for odometry messages"""
        self.record_message(topic, robot_namespace, 'odometry')
    
    def pose_callback(self, msg, topic, robot_namespace):
        """Callback for pose messages"""
        self.record_message(topic, robot_namespace, 'pose')
    
    def status_callback(self, msg, topic, robot_namespace, msg_type):
        """Callback for status messages"""
        self.record_message(topic, robot_namespace, f'status_{msg_type}')
    
    def record_message(self, topic, robot_namespace, msg_type):
        """Record message reception for analysis"""
        current_time = time.time()
        
        stats = self.topic_stats[topic]
        stats['message_count'] += 1
        stats['last_message_time'] = current_time
        stats['message_times'].append(current_time)
        stats['robot_namespace'] = robot_namespace
        
        # Calculate callback duration (simplified)
        callback_start = current_time
        # ... callback processing would happen here ...
        callback_duration = time.time() - callback_start
        stats['callback_durations'].append(callback_duration)
    
    def analyze_topic_performance(self):
        """Analyze topic performance and detect issues"""
        current_time = time.time()
        issues_found = []
        
        print(f"\n📈 ROS TOPIC ANALYSIS ({datetime.now().strftime('%H:%M:%S')})")
        print("=" * 80)
        
        for topic, stats in self.topic_stats.items():
            if stats['message_count'] == 0:
                continue
            
            # Calculate message rate
            recent_messages = [t for t in stats['message_times'] if current_time - t <= 10.0]
            actual_rate = len(recent_messages) / 10.0 if recent_messages else 0.0
            
            # Time since last message
            time_since_last = current_time - stats['last_message_time']
            
            # Expected vs actual rate comparison
            expected_rate = stats['expected_rate']
            rate_ratio = actual_rate / expected_rate if expected_rate > 0 else 1.0
            
            # Detect issues
            status = "✅"
            if time_since_last > 5.0:  # No message in 5 seconds
                status = "❌ STALE"
                issues_found.append(f"{topic}: No messages for {time_since_last:.1f}s")
            elif rate_ratio < 0.5:  # Less than 50% of expected rate
                status = "⚠️ LOW RATE"
                issues_found.append(f"{topic}: Low rate {actual_rate:.1f}Hz (expected {expected_rate:.1f}Hz)")
            elif rate_ratio > 2.0 and expected_rate > 1.0:  # More than 200% of expected rate (for high-rate topics)
                status = "⚠️ HIGH RATE"
            
            # Calculate average callback duration
            avg_callback_duration = np.mean(stats['callback_durations']) * 1000 if stats['callback_durations'] else 0
            
            # Print topic status
            robot_info = f"[{stats['robot_namespace']}]" if stats['robot_namespace'] else ""
            print(f"{status} {robot_info:12} {topic:35} | "
                  f"Rate: {actual_rate:5.1f}Hz (exp: {expected_rate:4.1f}Hz) | "
                  f"Total: {stats['message_count']:4d} | "
                  f"Last: {time_since_last:5.1f}s ago | "
                  f"Callback: {avg_callback_duration:.1f}ms")
        
        # Highlight critical issues
        if issues_found:
            print(f"\n🚨 CRITICAL ISSUES DETECTED ({len(issues_found)}):")
            for issue in issues_found:
                print(f"   • {issue}")
        else:
            print(f"\n✅ All monitored topics are healthy")
        
        # Robot-specific analysis
        self.analyze_robot_specific_issues()
    
    def analyze_robot_specific_issues(self):
        """Analyze issues specific to each robot"""
        print(f"\n🤖 ROBOT-SPECIFIC ANALYSIS:")
        
        robots = ['turtlebot0', 'turtlebot1', 'turtlebot2']
        for robot in robots:
            robot_topics = {topic: stats for topic, stats in self.topic_stats.items() 
                           if stats['robot_namespace'] == robot}
            
            if not robot_topics:
                print(f"   {robot}: No topics monitored")
                continue
            
            # Check for robot-specific issues
            current_time = time.time()
            stale_topics = []
            low_rate_topics = []
            
            for topic, stats in robot_topics.items():
                time_since_last = current_time - stats['last_message_time'] if stats['last_message_time'] > 0 else float('inf')
                
                if time_since_last > 5.0:
                    stale_topics.append(topic.split('/')[-1])  # Just the topic name
                
                recent_messages = [t for t in stats['message_times'] if current_time - t <= 10.0]
                actual_rate = len(recent_messages) / 10.0 if recent_messages else 0.0
                if actual_rate < stats['expected_rate'] * 0.5 and stats['expected_rate'] > 0:
                    low_rate_topics.append(f"{topic.split('/')[-1]}({actual_rate:.1f}Hz)")
            
            # Robot status summary
            status = "✅ HEALTHY"
            if stale_topics:
                status = "❌ DATA LOSS"
            elif low_rate_topics:
                status = "⚠️ DEGRADED"
            
            issues = []
            if stale_topics:
                issues.append(f"stale: {', '.join(stale_topics)}")
            if low_rate_topics:
                issues.append(f"low_rate: {', '.join(low_rate_topics)}")
            
            issue_text = f" ({'; '.join(issues)})" if issues else ""
            print(f"   {robot:12}: {status}{issue_text}")
    
    def get_topic_summary(self):
        """Get summary of all topic statistics"""
        summary = {
            'total_topics': len(self.topic_stats),
            'total_messages': sum(stats['message_count'] for stats in self.topic_stats.values()),
            'topics_by_robot': defaultdict(list)
        }
        
        for topic, stats in self.topic_stats.items():
            robot = stats['robot_namespace']
            if robot:
                summary['topics_by_robot'][robot].append({
                    'topic': topic,
                    'message_count': stats['message_count'],
                    'last_message_time': stats['last_message_time']
                })
        
        return summary


class SystemMonitorNode:
    """Main system monitoring coordinator"""
    
    def __init__(self):
        self.resource_monitor = SystemResourceMonitor()
        self.ros_monitor = None  # Will be initialized after rclpy.init()
        self.monitoring_active = True
        
    def start_monitoring(self):
        """Start all monitoring components"""
        print("🚀 Starting comprehensive system monitoring...")
        
        # Start system resource monitoring
        self.resource_monitor.start_monitoring(interval=1.0)
        
        # Initialize ROS
        rclpy.init()
        
        # Start ROS topic monitoring
        self.ros_monitor = RosTopicMonitor()
        
        print("✅ System monitoring active")
        
        try:
            # Main monitoring loop
            while self.monitoring_active and rclpy.ok():
                # Spin ROS node for topic monitoring
                rclpy.spin_once(self.ros_monitor, timeout_sec=1.0)
                
                # Print periodic summary
                if hasattr(self, '_last_summary_time'):
                    if time.time() - self._last_summary_time > 30.0:  # Every 30 seconds
                        self.print_comprehensive_summary()
                        self._last_summary_time = time.time()
                else:
                    self._last_summary_time = time.time()
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop all monitoring"""
        print("🛑 Stopping system monitoring...")
        
        self.monitoring_active = False
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Stop ROS monitoring
        if self.ros_monitor:
            self.ros_monitor.destroy_node()
        
        try:
            rclpy.shutdown()
        except:
            pass
        
        print("✅ System monitoring stopped")
    
    def print_comprehensive_summary(self):
        """Print comprehensive summary of all monitoring data"""
        print("\n" + "="*100)
        print(f"📊 COMPREHENSIVE SYSTEM ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        # System resource summary
        self.resource_monitor.print_summary()
        
        # ROS topic summary
        if self.ros_monitor:
            topic_summary = self.ros_monitor.get_topic_summary()
            print(f"\n📡 ROS TOPIC SUMMARY:")
            print(f"   Total topics monitored: {topic_summary['total_topics']}")
            print(f"   Total messages received: {topic_summary['total_messages']}")
            
            for robot, topics in topic_summary['topics_by_robot'].items():
                recent_activity = sum(1 for t in topics if time.time() - t['last_message_time'] < 10.0)
                print(f"   {robot}: {len(topics)} topics, {recent_activity} active in last 10s")
        
        print("="*100)


def main():
    """Main function"""
    monitor = SystemMonitorNode()
    
    try:
        monitor.start_monitoring()
    except Exception as e:
        print(f"Error in monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.stop_monitoring()


if __name__ == '__main__':
    main()
