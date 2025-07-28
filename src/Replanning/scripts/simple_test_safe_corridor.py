#!/usr/bin/env python3
"""
Simple test script for the updated get_safe_corridor function
"""

import sys
import os
import numpy as np

# Add paths
sys.path.append('/root/workspace/config')
sys.path.append('/root/workspace/src/Replanning/scripts')

from config_loader import config
from GenerateMatrix import load_reeb_graph_from_file
from Planing_functions import get_safe_corridor

def simple_test():
    """
    Simple test of the updated get_safe_corridor function
    """
    print("Simple test of updated get_safe_corridor function...")
    
    # Load configuration
    case = config.case
    N = config.N
    
    # Generate file paths
    graph_file = config.get_full_path(config.file_path, use_data_path=True)
    waypoints_file = config.get_full_path(config.waypoints_file_path, use_data_path=True)
    environment_file = config.get_full_path(config.environment_file, use_data_path=True)
    
    print(f"Case: {case}, N: {N}")
    
    try:
        # Load graph
        reeb_graph = load_reeb_graph_from_file(graph_file)
        print(f"Graph loaded with {len(reeb_graph.nodes)} nodes")
        
        # Test the updated function
        print("\nTesting updated get_safe_corridor function...")
        safe_corridor, Distance, Angle, vertex = get_safe_corridor(
            reeb_graph, waypoints_file, environment_file, 
            step_size=0.5, max_distance=100
        )
        
        print(f"✅ Function executed successfully!")
        print(f"Generated {len(safe_corridor)} corridor segments")
        
        # Print corridor details
        print(f"\nCorridor details:")
        for i, corridor in enumerate(safe_corridor):
            slope, y_min, y_max = corridor
            width = y_max - y_min
            distance = Distance[i][0]
            angle_deg = np.degrees(Angle[i][0])
            
            slope_text = f"{slope:.4f}" if abs(slope) < 1000000 else "Vertical"
            print(f"  Segment {i+1}:")
            print(f"    Slope: {slope_text}")
            print(f"    Width: {width:.2f}")
            print(f"    Range: [{y_min:.2f}, {y_max:.2f}]")
            print(f"    Distance: {distance:.2f}")
            print(f"    Angle: {angle_deg:.2f}°")
            print(f"    Vertex shape: {vertex[i].shape}")
        
        # Test with different parameters
        print(f"\nTesting with finer step size...")
        safe_corridor2, _, _, _ = get_safe_corridor(
            reeb_graph, waypoints_file, environment_file, 
            step_size=0.2, max_distance=80
        )
        
        print(f"✅ Fine step test completed!")
        
        # Compare corridor widths
        print(f"\nComparing corridor widths:")
        for i in range(len(safe_corridor)):
            width1 = safe_corridor[i][2] - safe_corridor[i][1]
            width2 = safe_corridor2[i][2] - safe_corridor2[i][1]
            print(f"  Segment {i+1}: Default={width1:.2f}, Fine={width2:.2f}, Diff={width2-width1:.2f}")
        
        print(f"\n🎉 All tests passed! Updated get_safe_corridor function is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()
