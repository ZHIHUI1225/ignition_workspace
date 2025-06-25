#!/usr/bin/env python3
"""
Diagnostic Launcher Script
Provides easy access to various diagnostic tools for behavior tree monitoring.
"""

import sys
import subprocess
import time
import threading
import os


def print_menu():
    """Print the diagnostic menu"""
    print("\n" + "="*60)
    print("🔍 BEHAVIOR TREE DIAGNOSTIC TOOLS")
    print("="*60)
    print("1. Run comprehensive system monitor (full monitoring)")
    print("2. Run behavior tree specific diagnostic (one-shot)")
    print("3. Run continuous BT diagnostic (every 10 seconds)")
    print("4. Run continuous BT diagnostic (every 30 seconds)")
    print("5. Monitor ROS topics only")
    print("6. Quick system resource check")
    print("7. Test ROS connectivity")
    print("0. Exit")
    print("="*60)


def run_system_monitor():
    """Run the comprehensive system monitor"""
    print("🚀 Starting comprehensive system monitor...")
    print("This will monitor CPU, memory, threads, and ROS topics.")
    print("Press Ctrl+C to stop.")
    
    try:
        subprocess.run([sys.executable, "/root/workspace/system_monitor.py"])
    except KeyboardInterrupt:
        print("\n✅ System monitor stopped")


def run_bt_diagnostic(continuous=False, interval=10):
    """Run behavior tree diagnostic"""
    script = "/root/workspace/bt_diagnostic.py"
    
    if continuous:
        print(f"🔄 Starting continuous BT diagnostic (every {interval}s)...")
        print("Press Ctrl+C to stop.")
        subprocess.run([sys.executable, script, "--continuous", str(interval)])
    else:
        print("📊 Running one-shot BT diagnostic...")
        subprocess.run([sys.executable, script])


def monitor_ros_topics():
    """Monitor ROS topics specifically"""
    print("📡 Monitoring ROS topics...")
    print("This will show topic publishers, subscribers, and data flow.")
    
    try:
        # Use ros2 command line tools
        print("\n1. Listing all topics:")
        subprocess.run(["ros2", "topic", "list"], timeout=10)
        
        print("\n2. Checking behavior tree related topics:")
        topics_to_check = [
            "/turtlebot0/odom_map",
            "/turtlebot1/odom_map", 
            "/turtlebot2/odom_map",
            "/parcel0/pose",
            "/parcel1/pose",
            "/parcel2/pose"
        ]
        
        for topic in topics_to_check:
            print(f"\n   Checking {topic}:")
            try:
                result = subprocess.run(["ros2", "topic", "info", topic], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"      ✅ {topic} - Active")
                    # Extract publisher/subscriber count
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Publisher count:' in line or 'Subscription count:' in line:
                            print(f"      {line.strip()}")
                else:
                    print(f"      ❌ {topic} - Not found or no publishers")
            except subprocess.TimeoutExpired:
                print(f"      ⏰ {topic} - Timeout checking topic")
    
    except FileNotFoundError:
        print("❌ ros2 command not found. Make sure ROS2 is properly sourced.")
    except Exception as e:
        print(f"❌ Error monitoring topics: {e}")


def quick_resource_check():
    """Quick system resource check"""
    print("⚡ Quick system resource check...")
    
    import psutil
    
    # CPU and memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"💻 System Resources:")
    print(f"   CPU Usage: {cpu_percent:.1f}%")
    print(f"   Memory Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    print(f"   Available Memory: {memory.available / 1024**3:.1f}GB")
    
    # Check for behavior tree processes
    bt_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'num_threads']):
        try:
            if (proc.info['name'] == 'python3' and 
                proc.info['cmdline'] and 
                any('behaviour_tree' in arg for arg in proc.info['cmdline'])):
                bt_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    print(f"\n🤖 Behavior Tree Processes ({len(bt_processes)}):")
    if bt_processes:
        for proc in bt_processes:
            memory_mb = proc['memory_info'].rss / 1024 / 1024
            print(f"   PID {proc['pid']:5d}: CPU {proc['cpu_percent']:5.1f}% | "
                  f"Memory {memory_mb:6.1f}MB | Threads {proc['num_threads']:2d}")
    else:
        print("   ❌ No behavior tree processes found")


def test_ros_connectivity():
    """Test basic ROS connectivity"""
    print("🔗 Testing ROS connectivity...")
    
    try:
        # Test if ROS is running
        result = subprocess.run(["ros2", "node", "list"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            nodes = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            print(f"✅ ROS2 is running with {len(nodes)} nodes:")
            
            # Show behavior tree related nodes
            bt_nodes = [node for node in nodes if 'tree' in node]
            if bt_nodes:
                print(f"   🌳 Behavior tree nodes ({len(bt_nodes)}):")
                for node in bt_nodes:
                    print(f"      • {node}")
            else:
                print(f"   ⚠️ No behavior tree nodes found")
            
            # Show all nodes if not too many
            if len(nodes) <= 10:
                print(f"   📋 All nodes:")
                for node in nodes:
                    print(f"      • {node}")
            else:
                print(f"   📋 Total nodes: {len(nodes)} (showing BT nodes only)")
        else:
            print("❌ ROS2 not running or not accessible")
            print(f"Error: {result.stderr}")
    
    except FileNotFoundError:
        print("❌ ros2 command not found. Please source ROS2 setup.")
    except subprocess.TimeoutExpired:
        print("⏰ Timeout testing ROS connectivity")
    except Exception as e:
        print(f"❌ Error testing ROS connectivity: {e}")


def main():
    """Main menu loop"""
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect option (0-7): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                run_system_monitor()
            elif choice == '2':
                run_bt_diagnostic(continuous=False)
            elif choice == '3':
                run_bt_diagnostic(continuous=True, interval=10)
            elif choice == '4':
                run_bt_diagnostic(continuous=True, interval=30)
            elif choice == '5':
                monitor_ros_topics()
            elif choice == '6':
                quick_resource_check()
            elif choice == '7':
                test_ros_connectivity()
            else:
                print("❌ Invalid choice. Please select 0-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")


if __name__ == '__main__':
    main()
