#!/usr/bin/env python3
"""
Generate normalization parameters using get_normalization_prams function
"""

import sys
import os

# Add paths
sys.path.append('/root/workspace/config')
sys.path.append('/root/workspace/src/Replanning/scripts')

from config_loader import config
from GenerateMatrix import load_reeb_graph_from_file
from Planing_functions import get_normalization_prams

def generate_normalization_using_function():
    """Generate normalization file using the get_normalization_prams function"""
    
    print("Generating normalization parameters using get_normalization_prams...")
    
    # Load configuration
    case = config.case
    N = config.N
    
    # Get file paths
    graph_file = config.get_full_path(config.file_path, use_data_path=True)
    waypoints_file = config.get_full_path(config.waypoints_file_path, use_data_path=True)
    normalization_file = config.get_full_path(config.Normalization_planning_path, use_data_path=True)  # Use planning-specific path
    
    print(f"Configuration:")
    print(f"  Case: {case}")
    print(f"  N: {N}")
    print(f"  Graph file: {graph_file}")
    print(f"  Waypoints file: {waypoints_file}")
    print(f"  Output normalization file: {normalization_file}")
    
    # Check if input files exist
    for file_path, name in [(graph_file, "Graph"), (waypoints_file, "Waypoints")]:
        if not os.path.exists(file_path):
            print(f"❌ ERROR: {name} file not found: {file_path}")
            return False
    
    try:
        # Load graph
        print(f"\nLoading reeb graph...")
        reeb_graph = load_reeb_graph_from_file(graph_file)
        print(f"Graph loaded with {len(reeb_graph.nodes)} nodes")
        
        # Call the get_normalization_prams function
        print(f"\nCalling get_normalization_prams...")
        result = get_normalization_prams(
            waypoints_file, 
            normalization_file, 
            reeb_graph, 
            None  # Initial_Guess_file_path not needed for this function
        )
        
        if result:
            print(f"✅ Normalization file generated successfully!")
            
            # Verify the generated file
            if os.path.exists(normalization_file):
                print(f"File saved to: {normalization_file}")
                
                # Read and display the contents
                import json
                with open(normalization_file, 'r') as f:
                    data = json.load(f)
                
                print(f"\nGenerated normalization data:")
                print(f"  Keys: {list(data.keys())}")
                print(f"  Number of segments (al): {len(data['al'])}")
                print(f"  Number of segments (length_min): {len(data['length_min'])}")
                print(f"  ac value: {data['ac']:.6f}")
                print(f"  curvature_max: {data['curvature_max']:.6f}")
                print(f"  curvature_min: {data['curvature_min']}")
                
                print(f"\nSegment details:")
                for i in range(len(data['al'])):
                    print(f"    Segment {i+1}: al={data['al'][i]:.6f}, length_min={data['length_min'][i]:.2f}, length_max={data['length_max'][i]:.2f}")
                
                return True
            else:
                print(f"❌ ERROR: File was not created at {normalization_file}")
                return False
        else:
            print(f"❌ ERROR: get_normalization_prams returned False")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_normalization_using_function()
    if success:
        print(f"\n🎉 Normalization file generation completed successfully!")
        print(f"The GA_planning.py script should now work correctly.")
    else:
        print(f"\n💥 Normalization file generation failed!")
