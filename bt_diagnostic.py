#!/usr/bin/env python3
"""
Behavior Tree Node Diagnostic Tool
Specifically monitors behavior tree nodes, callback groups, and subscription health.
"""

import rclpy
from rclpy.node import Node
import psutil
import threading
import time
import re
import os
from collections import defaultdict
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import subprocess  # Ensure subprocess is imported at the top level


class BehaviorTreeDiagnostic:
    """Diagnostic tool for behavior tree nodes and callback issues"""
    
    def __init__(self):
        self.node_info = {}
        self.callback_group_stats = defaultdict(dict)
        self.subscription_health = defaultdict(dict)
        self.monitoring = True
        self.previous_cpu_samples = {}  # Store previous CPU readings for trend analysis
        self.cpu_history = defaultdict(list)  # Store CPU history per process
        self.sample_count = 0  # Count of samples for averaging
        
    def scan_behavior_tree_processes(self):
        """Scan for behavior tree processes and extract information"""
        bt_processes = []
        
        try:
            # Use subprocess to get process information as a fallback
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if 'my_behaviour_tree_modular' in line and 'tree_' in line:
                        parts = line.split()
                        pid = int(parts[1])
                        cpu_percent = float(parts[2])
                        memory_percent = float(parts[3])
                        
                        # Extract robot ID from command line
                        robot_match = re.search(r'tree_(\d+)', line)
                        robot_id = robot_match.group(1) if robot_match else 'unknown'
                        
                        # Try to get more detailed info from psutil if possible
                        try:
                            proc = psutil.Process(pid)
                            memory_mb = proc.memory_info().rss / 1024 / 1024
                            threads = proc.num_threads()
                            uptime_seconds = time.time() - proc.create_time()
                        except:
                            # Fallback to estimated values
                            memory_mb = memory_percent * 16  # Rough estimate assuming 16GB RAM
                            threads = 10  # Rough estimate
                            uptime_seconds = 0
                        
                        bt_processes.append({
                            'pid': pid,
                            'robot_id': robot_id,
                            'robot_namespace': f'turtlebot{robot_id}',
                            'cpu_percent': cpu_percent,
                            'memory_mb': memory_mb,
                            'threads': threads,
                            'uptime_seconds': uptime_seconds,
                            'cmdline': line
                        })
                        print(f"DEBUG: Found BT process via ps - PID: {pid}, Robot: {robot_id}")
            
            # If ps method found processes, return them
            if bt_processes:
                print(f"DEBUG: Found {len(bt_processes)} behavior tree processes via ps command")
                return bt_processes
        
        except Exception as e:
            print(f"DEBUG: Error using ps command: {e}")
        
        # Fallback to psutil method
        total_python_procs = 0
        debug_matches = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'num_threads', 'create_time']):
            try:
                if proc.info['name'] == 'python3':
                    total_python_procs += 1
                    if proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        
                        # Debug: Check for any behavior tree related processes
                        if 'behaviour_tree' in cmdline.lower() or 'my_behaviour_tree' in cmdline:
                            debug_matches.append(f"PID {proc.info['pid']}: {cmdline[:100]}...")
                        
                        # Look for the behavior tree executable in the command line
                        if 'my_behaviour_tree_modular' in cmdline:
                            # Extract robot information from command line
                            robot_match = re.search(r'tree_(\d+)', cmdline)
                            robot_id = robot_match.group(1) if robot_match else 'unknown'
                            
                            bt_processes.append({
                                'pid': proc.info['pid'],
                                'robot_id': robot_id,
                                'robot_namespace': f'turtlebot{robot_id}',
                                'cpu_percent': proc.info['cpu_percent'],
                                'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                                'threads': proc.info['num_threads'],
                                'uptime_seconds': time.time() - proc.info['create_time'],
                                'cmdline': cmdline
                            })
                            print(f"DEBUG: Found BT process via psutil - PID: {proc.info['pid']}, Robot: {robot_id}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                continue
            except Exception as e:
                print(f"DEBUG: Error processing PID {proc.info.get('pid', 'unknown')}: {e}")
                continue
        
        # Debug output
        print(f"DEBUG: Found {total_python_procs} python3 processes via psutil, {len(bt_processes)} behavior tree processes")
        if debug_matches:
            print("DEBUG: Behavior tree related processes found:")
            for match in debug_matches:
                print(f"  {match}")
        
        return bt_processes
    
    def get_system_thread_info(self):
        """Get detailed thread information"""
        try:
            # Get overall system thread count
            total_threads = 0
            python_threads = 0
            
            for proc in psutil.process_iter(['name', 'num_threads']):
                try:
                    total_threads += proc.info['num_threads']
                    if proc.info['name'] == 'python3':
                        python_threads += proc.info['num_threads']
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'total_system_threads': total_threads,
                'python_threads': python_threads,
                'thread_limit': os.sysconf('SC_THREAD_THREADS_MAX') if hasattr(os, 'sysconf') else 'unknown'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_ros_node_health(self):
        """Check ROS node health by attempting to query topics"""
        try:
            # Check if ROS is already initialized
            if not rclpy.ok():
                rclpy.init()
            
            temp_node = rclpy.create_node('diagnostic_node')
            
            # Get list of all nodes
            node_names = temp_node.get_node_names()
            
            # Get topic information
            topic_info = temp_node.get_topic_names_and_types()
            topics_dict = {name: types for name, types in topic_info}
            
            # Analyze behavior tree related topics
            bt_topics = {}
            robot_topics = defaultdict(list)
            
            for topic_name in topics_dict:
                if any(robot in topic_name for robot in ['turtlebot0', 'turtlebot1', 'turtlebot2']):
                    # Extract robot namespace
                    robot_match = re.search(r'/(turtlebot\d+)/', topic_name)
                    if robot_match:
                        robot_ns = robot_match.group(1)
                        robot_topics[robot_ns].append(topic_name)
                
                if any(keyword in topic_name for keyword in ['parcel', 'Relaypoint', 'odom_map', 'pushing']):
                    bt_topics[topic_name] = topics_dict[topic_name]
            
            result = {
                'total_nodes': len(node_names),
                'tree_nodes': [name for name in node_names if 'tree' in name],
                'total_topics': len(topics_dict),
                'bt_related_topics': len(bt_topics),
                'robot_topics': dict(robot_topics),
                'bt_topics': bt_topics
            }
            
            temp_node.destroy_node()
            # Don't shutdown ROS as other parts might need it
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def monitor_topic_publishers_subscribers(self):
        """Monitor topic publishers and subscribers for each robot"""
        try:
            if not rclpy.ok():
                rclpy.init()
            
            temp_node = rclpy.create_node('topic_monitor')
            
            # Topics of interest for each robot
            robot_topics = {
                'turtlebot0': ['/turtlebot0/odom_map', '/parcel0/pose', '/Relaypoint1/pose'],
                'turtlebot1': ['/turtlebot1/odom_map', '/parcel1/pose', '/Relaypoint2/pose'],
                'turtlebot2': ['/turtlebot2/odom_map', '/parcel2/pose', '/Relaypoint3/pose']
            }
            
            topic_health = {}
            
            for robot, topics in robot_topics.items():
                robot_health = {}
                for topic in topics:
                    publishers = temp_node.count_publishers(topic)
                    subscribers = temp_node.count_subscribers(topic)
                    
                    # Determine health status
                    if publishers == 0:
                        status = 'NO_PUBLISHER'
                    elif subscribers == 0:
                        status = 'NO_SUBSCRIBERS'
                    else:
                        status = 'HEALTHY'
                    
                    robot_health[topic] = {
                        'publishers': publishers,
                        'subscribers': subscribers,
                        'status': status
                    }
                
                topic_health[robot] = robot_health
            
            temp_node.destroy_node()
            # Don't shutdown ROS as other parts might need it
            
            return topic_health
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_callback_group_conflicts(self, bt_processes):
        """Analyze process resource usage without providing recommendations"""
        analysis = {
            'total_processes': len(bt_processes),
            'thread_distribution': {},
            'resource_pressure': {}
        }
        
        # Analyze thread distribution
        for proc in bt_processes:
            robot_id = proc['robot_id']
            analysis['thread_distribution'][robot_id] = {
                'threads': proc['threads'],
                'cpu_percent': proc['cpu_percent'],
                'memory_mb': proc['memory_mb']
            }
        
        # Calculate total resource usage (for informational purposes only)
        total_threads = sum(proc['threads'] for proc in bt_processes)
        analysis['total_threads'] = total_threads
        
        return analysis
    
    def get_detailed_cpu_analysis(self, bt_processes):
        """Get detailed CPU analysis including per-thread CPU usage and top functions"""
        cpu_analysis = {}
        
        for proc_info in bt_processes:
            pid = proc_info['pid']
            robot_id = proc_info['robot_id']
            
            try:
                proc = psutil.Process(pid)
                
                # Store current CPU sample and calculate delta if we have previous samples
                current_time = time.time()
                current_cpu_times = proc.cpu_times()
                
                if pid in self.previous_cpu_samples:
                    prev_sample = self.previous_cpu_samples[pid]
                    time_delta = current_time - prev_sample['time']
                    
                    # Calculate CPU time delta in seconds
                    user_delta = current_cpu_times.user - prev_sample['cpu_times'].user
                    system_delta = current_cpu_times.system - prev_sample['cpu_times'].system
                    
                    # Calculate percentage of CPU time
                    if time_delta > 0:
                        user_percent = (user_delta / time_delta) * 100
                        system_percent = (system_delta / time_delta) * 100
                    else:
                        user_percent = system_percent = 0
                        
                    # Store for analysis
                    cpu_analysis[robot_id] = {
                        'user_cpu_percent': user_percent,
                        'system_cpu_percent': system_percent,
                        'total_cpu_percent': user_percent + system_percent,
                        'time_delta': time_delta,
                    }
                    
                    # Store history for trending
                    self.cpu_history[robot_id].append(user_percent + system_percent)
                    # Keep only last 10 samples
                    if len(self.cpu_history[robot_id]) > 10:
                        self.cpu_history[robot_id].pop(0)
                    
                # Update the previous sample
                self.previous_cpu_samples[pid] = {
                    'time': current_time,
                    'cpu_times': current_cpu_times
                }
                
                # Get thread-specific CPU usage if available
                try:
                    threads = []
                    for thread in proc.threads():
                        thread_id = thread.id
                        # Calculate thread CPU if we have previous data
                        threads.append({
                            'id': thread_id,
                            'user_time': thread.user_time,
                            'system_time': thread.system_time,
                            'total_time': thread.user_time + thread.system_time
                        })
                    
                    # Sort threads by total CPU time
                    threads.sort(key=lambda x: x['total_time'], reverse=True)
                    
                    # Add the top 5 threads to the analysis
                    if robot_id in cpu_analysis:
                        top_threads = threads[:5]
                        cpu_analysis[robot_id]['top_threads'] = top_threads
                        
                        # Try to get stack traces for the top 2 CPU-intensive threads
                        if top_threads:
                            try:
                                top_thread_ids = [t['id'] for t in top_threads[:2]]
                                stack_traces = self.get_thread_stack_traces(pid, top_thread_ids)
                                cpu_analysis[robot_id]['stack_traces'] = stack_traces
                            except Exception as e:
                                print(f"Failed to get stack traces: {e}")
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    if robot_id in cpu_analysis:
                        cpu_analysis[robot_id]['top_threads'] = []
                
                # Try to get command line for the process to identify its role
                try:
                    cmdline = proc.cmdline()
                    if robot_id in cpu_analysis:
                        cpu_analysis[robot_id]['cmdline'] = ' '.join(cmdline)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Error analyzing CPU for process {pid}: {e}")
                continue
        
        # Add trend analysis
        for robot_id, history in self.cpu_history.items():
            if robot_id in cpu_analysis and len(history) > 1:
                # Calculate trend (positive = increasing, negative = decreasing)
                trend = history[-1] - history[0]
                cpu_analysis[robot_id]['cpu_trend'] = trend
                
                # Calculate average
                avg = sum(history) / len(history)
                cpu_analysis[robot_id]['cpu_average'] = avg
        
        self.sample_count += 1
        return cpu_analysis
    
    def run_comprehensive_diagnostic(self):
        """Run comprehensive diagnostic and generate report"""
        print("🔍 Starting Behavior Tree Diagnostic Analysis...")
        print("=" * 80)
        
        # 1. Scan behavior tree processes
        print("\n📊 BEHAVIOR TREE PROCESSES:")
        bt_processes = self.scan_behavior_tree_processes()
        
        if not bt_processes:
            print("❌ No behavior tree processes found!")
            return
        
        for proc in bt_processes:
            uptime_min = proc['uptime_seconds'] / 60
            print(f"   🤖 Robot {proc['robot_id']} (PID: {proc['pid']})")
            print(f"      CPU: {proc['cpu_percent']:5.1f}% | Memory: {proc['memory_mb']:6.1f}MB | "
                  f"Threads: {proc['threads']:2d} | Uptime: {uptime_min:.1f}min")
        
        # 2. System thread analysis
        print(f"\n🧵 SYSTEM THREAD ANALYSIS:")
        thread_info = self.get_system_thread_info()
        if 'error' not in thread_info:
            print(f"   Total system threads: {thread_info['total_system_threads']}")
            print(f"   Python threads: {thread_info['python_threads']}")
            print(f"   Thread limit: {thread_info.get('thread_limit', 'unknown')}")
        else:
            print(f"   Error: {thread_info['error']}")
        
        # 3. ROS node health
        print(f"\n🔧 ROS NODE & TOPIC HEALTH:")
        node_health = self.check_ros_node_health()
        if 'error' not in node_health:
            print(f"   Total ROS nodes: {node_health['total_nodes']}")
            print(f"   Tree nodes: {node_health['tree_nodes']}")
            print(f"   Total topics: {node_health['total_topics']}")
            print(f"   BT-related topics: {node_health['bt_related_topics']}")
            
            # Show robot-specific topic counts
            for robot, topics in node_health['robot_topics'].items():
                print(f"   {robot}: {len(topics)} topics")
        else:
            print(f"   Error: {node_health['error']}")
        
        # 4. Topic publisher/subscriber health
        print(f"\n📡 TOPIC PUBLISHER/SUBSCRIBER HEALTH:")
        topic_health = self.monitor_topic_publishers_subscribers()
        if 'error' not in topic_health:
            for robot, topics in topic_health.items():
                print(f"   🤖 {robot}:")
                for topic, health in topics.items():
                    status_emoji = {"HEALTHY": "✅", "NO_PUBLISHER": "❌", "NO_SUBSCRIBERS": "⚠️"}
                    emoji = status_emoji.get(health['status'], "❓")
                    topic_short = topic.split('/')[-1]
                    print(f"      {emoji} {topic_short:15} | P:{health['publishers']} S:{health['subscribers']} | {health['status']}")
        else:
            print(f"   Error: {topic_health['error']}")
        
        # 5. Process resource usage analysis
        print(f"\n📊 PROCESS RESOURCE USAGE:")
        resource_analysis = self.analyze_callback_group_conflicts(bt_processes)
        print(f"   Total BT processes: {resource_analysis['total_processes']}")
        print(f"   Total threads: {resource_analysis.get('total_threads', 0)}")
        
        for robot_id, stats in resource_analysis['thread_distribution'].items():
            print(f"   Robot {robot_id}: {stats['threads']} threads, {stats['cpu_percent']:.1f}% CPU, {stats['memory_mb']:.1f}MB")
        
        # 6. Detailed CPU analysis
        print(f"\n🖥️ DETAILED CPU ANALYSIS:")
        cpu_analysis = self.get_detailed_cpu_analysis(bt_processes)
        
        for robot_id, stats in cpu_analysis.items():
            print(f"   Robot {robot_id}:")
            print(f"      Total CPU: {stats['total_cpu_percent']:.1f}%")
            print(f"      User CPU: {stats['user_cpu_percent']:.1f}%")
            print(f"      System CPU: {stats['system_cpu_percent']:.1f}%")
            if 'cpu_trend' in stats:
                trend = "increasing" if stats['cpu_trend'] > 0 else "decreasing"
                print(f"      Trend: {trend} ({stats['cpu_trend']:.1f})")
            if 'cpu_average' in stats:
                print(f"      Average CPU: {stats['cpu_average']:.1f}%")
            if 'top_threads' in stats:
                print(f"      Top threads by CPU usage:")
                for i, thread in enumerate(stats['top_threads'][:3]):  # Show top 3 threads
                    print(f"         #{i+1} Thread {thread['id']}: {thread['total_time']:.1f}s total "
                          f"(User: {thread['user_time']:.1f}s, System: {thread['system_time']:.1f}s)")
                
                # Show stack traces for top threads if available
                if 'stack_traces' in stats:
                    print("\n      Stack traces for top CPU threads:")
                    for thread_id, trace in stats['stack_traces'].items():
                        print(f"         Thread {thread_id} stack:")
                        # Extract the useful parts of the stack trace (limit to first 15 lines)
                        trace_lines = trace.split('\n')
                        relevant_lines = [line for line in trace_lines if 'Thread' in line or '#' in line][:15]
                        for line in relevant_lines:
                            print(f"            {line.strip()}")
            if 'cmdline' in stats:
                print(f"      Command line: {stats['cmdline']}")
        
        print("=" * 80)
        print("🔍 Diagnostic analysis complete")
    
    
    def start_continuous_monitoring(self, interval=10):
        """Start continuous monitoring with periodic reports"""
        print(f"🔄 Starting continuous monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while self.monitoring:
                self.run_comprehensive_diagnostic()
                print(f"\n⏰ Next analysis in {interval} seconds...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n🛑 Continuous monitoring stopped")
        except Exception as e:
            print(f"\n❌ Error in continuous monitoring: {e}")
            import traceback
            traceback.print_exc()
    
    def get_thread_stack_traces(self, pid, thread_ids):
        """Get stack traces for specific threads in a process"""
        stack_traces = {}
        try:
            # Use gdb to get stack traces (requires root)
            for thread_id in thread_ids:
                cmd = [
                    'gdb', '-ex', f'attach {pid}', 
                    '-ex', f'thread {thread_id}', 
                    '-ex', 'bt', 
                    '-ex', 'detach', 
                    '-ex', 'quit', 
                    '--batch'
                ]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    stack_traces[thread_id] = result.stdout
                except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
                    stack_traces[thread_id] = f"Error getting stack trace: {e}"
        except Exception as e:
            print(f"Error getting stack traces: {e}")
        return stack_traces


def main():
    """Main function"""
    import sys
    
    diagnostic = BehaviorTreeDiagnostic()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        diagnostic.start_continuous_monitoring(interval)
    else:
        diagnostic.run_comprehensive_diagnostic()


if __name__ == '__main__':
    main()
