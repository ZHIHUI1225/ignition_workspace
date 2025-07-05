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


class BehaviorTreeDiagnostic:
    """Diagnostic tool for behavior tree nodes and callback issues"""
    
    def __init__(self):
        self.node_info = {}
        self.callback_group_stats = defaultdict(dict)
        self.subscription_health = defaultdict(dict)
        self.monitoring = True
        
    def scan_behavior_tree_processes(self):
        """Scan for behavior tree processes and extract information"""
        bt_processes = []
        
        try:
            # Use subprocess to get process information as a fallback
            import subprocess
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
        """Analyze potential callback group conflicts"""
        analysis = {
            'total_processes': len(bt_processes),
            'thread_distribution': {},
            'potential_conflicts': [],
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
        
        # Check for potential conflicts
        total_threads = sum(proc['threads'] for proc in bt_processes)
        if total_threads > 50:  # Arbitrary threshold
            analysis['potential_conflicts'].append(f"High thread count: {total_threads} total threads")
        
        # Check for resource pressure
        high_cpu_processes = [proc for proc in bt_processes if proc['cpu_percent'] > 50]
        if high_cpu_processes:
            analysis['potential_conflicts'].append(f"High CPU usage processes: {len(high_cpu_processes)}")
        
        high_memory_processes = [proc for proc in bt_processes if proc['memory_mb'] > 200]
        if high_memory_processes:
            analysis['potential_conflicts'].append(f"High memory usage processes: {len(high_memory_processes)}")
        
        return analysis
    
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
        
        # 5. Callback group conflict analysis
        print(f"\n⚠️  CALLBACK GROUP CONFLICT ANALYSIS:")
        conflict_analysis = self.analyze_callback_group_conflicts(bt_processes)
        print(f"   Total BT processes: {conflict_analysis['total_processes']}")
        
        for robot_id, stats in conflict_analysis['thread_distribution'].items():
            print(f"   Robot {robot_id}: {stats['threads']} threads, {stats['cpu_percent']:.1f}% CPU, {stats['memory_mb']:.1f}MB")
        
        if conflict_analysis['potential_conflicts']:
            print(f"\n🚨 POTENTIAL ISSUES DETECTED:")
            for conflict in conflict_analysis['potential_conflicts']:
                print(f"   • {conflict}")
        else:
            print(f"   ✅ No obvious callback group conflicts detected")
        
        # 6. Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        self.generate_recommendations(bt_processes, conflict_analysis, topic_health)
        
        print("=" * 80)
        print("🔍 Diagnostic analysis complete")
    
    def generate_recommendations(self, bt_processes, conflict_analysis, topic_health):
        """Generate recommendations based on diagnostic results"""
        recommendations = []
        
        # Check for high resource usage
        for proc in bt_processes:
            if proc['cpu_percent'] > 30:
                recommendations.append(f"Robot {proc['robot_id']}: High CPU usage ({proc['cpu_percent']:.1f}%) - consider reducing control frequency")
            
            if proc['threads'] > 15:
                recommendations.append(f"Robot {proc['robot_id']}: High thread count ({proc['threads']}) - check for callback group proliferation")
            
            if proc['memory_mb'] > 150:
                recommendations.append(f"Robot {proc['robot_id']}: High memory usage ({proc['memory_mb']:.1f}MB) - check for memory leaks")
        
        # Check topic health issues
        if 'error' not in topic_health:
            unhealthy_robots = []
            for robot, topics in topic_health.items():
                unhealthy_topics = [topic for topic, health in topics.items() if health['status'] != 'HEALTHY']
                if unhealthy_topics:
                    unhealthy_robots.append(f"{robot}: {len(unhealthy_topics)} unhealthy topics")
            
            if unhealthy_robots:
                recommendations.append("Topic connectivity issues detected - check Gazebo simulation and topic publishers")
        
        # General recommendations
        if len(bt_processes) >= 3:
            recommendations.append("Consider using staggered startup delays for behavior tree nodes")
            recommendations.append("Ensure each robot uses isolated callback groups and executors")
        
        if not recommendations:
            recommendations.append("✅ System appears healthy - no specific recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
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
