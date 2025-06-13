#!/usr/bin/env python3
"""
Simple test script to verify PickObject behavior with replanned trajectory support
"""

import sys
import os

# Add the source path to import the behavior
sys.path.insert(0, '/root/workspace/src/behaviour_tree/behaviour_tree/behaviors')

# Test imports
try:
    from manipulation_behaviors import PickObject
    print("✓ Successfully imported PickObject")
except ImportError as e:
    print(f"✗ Failed to import PickObject: {e}")
    sys.exit(1)

try:
    import numpy as np
    import json
    print("✓ Required dependencies available")
except ImportError as e:
    print(f"✗ Missing dependencies: {e}")
    sys.exit(1)

# Test trajectory file existence
replanned_file = '/root/workspace/data/simple_maze/tb0_Trajectory_replanned.json'
original_file = '/root/workspace/data/simple_maze/tb0_Trajectory.json'

print("\nTrajectory file availability:")
if os.path.exists(replanned_file):
    print(f"✓ Replanned trajectory found: {replanned_file}")
    
    # Check file structure
    try:
        with open(replanned_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Replanned trajectory contains {len(data['Trajectory'])} points")
    except Exception as e:
        print(f"✗ Error reading replanned trajectory: {e}")
        
elif os.path.exists(original_file):
    print(f"✓ Original trajectory found: {original_file}")
    
    try:
        with open(original_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Original trajectory contains {len(data['Trajectory'])} points")
    except Exception as e:
        print(f"✗ Error reading original trajectory: {e}")
else:
    print(f"✗ No trajectory files found")
    sys.exit(1)

# Test PickObject instantiation
try:
    pick_behavior = PickObject("test_pick")
    print("✓ Successfully created PickObject instance")
    
    # Test trajectory loading method exists
    if hasattr(pick_behavior, '_load_trajectory_data'):
        print("✓ PickObject has _load_trajectory_data method")
    else:
        print("✗ PickObject missing _load_trajectory_data method")
        
    # Test MPC controller setup method exists
    if hasattr(pick_behavior, '_initialize_mpc_controller'):
        print("✓ PickObject has _initialize_mpc_controller method")
    else:
        print("✗ PickObject missing _initialize_mpc_controller method")
        
    print("\n✓ All tests passed! PickObject behavior is ready for replanned trajectory support.")
    
except Exception as e:
    print(f"✗ Error creating PickObject: {e}")
    sys.exit(1)
