#!/usr/bin/env python3
"""
Simple test to verify stopping constraints are applied correctly
"""

import sys
import json
sys.path.append('/root/workspace/src/behaviour_tree')

from behaviour_tree.behaviors.Replan_behaviors import load_trajectory_parameters_individual

def analyze_trajectory_structure(case, robot_id):
    """Analyze the structure of trajectory segments"""
    
    print(f"=== Analyzing Trajectory Structure for Robot {robot_id} in {case} ===")
    
    # Load trajectory data
    trajectory_data = load_trajectory_parameters_individual(case, robot_id)
    
    if not trajectory_data:
        print(f"No trajectory data found for Robot {robot_id}")
        return
    
    time_segments = trajectory_data.get('time_segments', [])
    
    print(f"Number of segments: {len(time_segments)}")
    
    for i, segment in enumerate(time_segments):
        arc_times = segment.get('arc', [])
        line_times = segment.get('line', [])
        
        has_arc = len(arc_times) > 0
        has_line = len(line_times) > 0
        
        print(f"Segment {i}:")
        print(f"  - Has arc: {has_arc} ({len(arc_times)} subsegments)")
        print(f"  - Has line: {has_line} ({len(line_times)} subsegments)")
        
        if i == len(time_segments) - 1:  # Final segment
            print(f"  - This is the FINAL segment")
            if has_arc and not has_line:
                print(f"  - ✓ Arc stopping constraint should apply (ends with arc)")
            elif has_line:
                print(f"  - ✓ Line stopping constraint should apply (ends with line)")
            else:
                print(f"  - ⚠ No arc or line segments found")
    
    return trajectory_data

def main():
    # Test multiple robots to see different trajectory structures
    test_cases = [
        ("simple_maze", 0),
        ("simple_maze", 1),
        ("simple_maze", 2),
    ]
    
    for case, robot_id in test_cases:
        try:
            analyze_trajectory_structure(case, robot_id)
            print()
        except Exception as e:
            print(f"Error analyzing {case} robot {robot_id}: {e}")
            print()

if __name__ == "__main__":
    main()
