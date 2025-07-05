#!/usr/bin/env python3
"""
PATCH: Improved Tree Reset Logic
This patch makes the reset behavior more configurable and prevents wrong resets.
"""

import py_trees
import time

# IMPROVED reset configuration - can be modified via ROS parameters
IMPROVED_TREE_RESET_CONFIG = {
    'auto_reset_on_success': False,        # Changed: Don't auto-reset on success by default
    'auto_reset_on_failure': False,        # Keep: Don't auto-reset on failure
    'max_iterations': 3,                   # Changed: Lower limit for debugging
    'reset_on_specific_failures': [        # Only reset on these specific failures
        'timeout',
        'minor_error',
        'trajectory_deviation',
        'path_blocked'
    ],
    'no_reset_on_critical_failures': [     # Never reset on these
        'system_failure',
        'hardware_error', 
        'safety_violation',
        'mpc_initialization_failure',
        'trajectory_loading_failure',
        'setup_failure'
    ],
    'manual_reset_only': False,            # NEW: If True, only manual resets allowed
    'reset_delay': 2.0,                   # Increased delay to observe state
    'debug_mode': True,                   # Enable detailed logging
    'require_manual_confirmation': False   # NEW: Require user confirmation for resets
}

def check_tree_reset_conditions_improved(tree, robot_namespace, completion_info):
    """
    IMPROVED version of reset condition checking with better control
    """
    try:
        # Use improved configuration
        config = IMPROVED_TREE_RESET_CONFIG.copy()
        
        status = completion_info['status']
        iteration = completion_info['iteration']
        
        if config['debug_mode']:
            print(f"\n🔍 [{robot_namespace}] RESET ANALYSIS:")
            print(f"   Status: {status}")
            print(f"   Iteration: {iteration}")
            print(f"   Max iterations: {config['max_iterations']}")
            print(f"   Manual reset only: {config['manual_reset_only']}")
        
        # Check if manual reset only mode is enabled
        if config['manual_reset_only']:
            print(f"🔒 [{robot_namespace}] Manual reset only mode - no automatic resets")
            return False
        
        # Check iteration limit
        if iteration >= config['max_iterations']:
            print(f"🛑 [{robot_namespace}] Maximum iterations ({config['max_iterations']}) reached - stopping")
            return False
        
        # Handle SUCCESS case
        if status == py_trees.common.Status.SUCCESS:
            if config['auto_reset_on_success']:
                print(f"✅ [{robot_namespace}] Success - auto-reset enabled")
                if config['require_manual_confirmation']:
                    response = input(f"Reset tree for {robot_namespace}? (y/n): ")
                    return response.lower() == 'y'
                return True
            else:
                print(f"✅ [{robot_namespace}] Success - auto-reset disabled (staying completed)")
                return False
        
        # Handle FAILURE case
        elif status == py_trees.common.Status.FAILURE:
            # Get failure context from blackboard
            try:
                blackboard = py_trees.blackboard.Client()
                blackboard.register_key(
                    key=f"{robot_namespace}/failure_context",
                    access=py_trees.common.Access.READ
                )
                failure_context = getattr(blackboard, f"{robot_namespace.replace('/', '_')}_failure_context", "unknown")
                
                if config['debug_mode']:
                    print(f"   Failure context: {failure_context}")
                
                # Check if this is a critical failure that should not be reset
                if failure_context in config['no_reset_on_critical_failures']:
                    print(f"🚨 [{robot_namespace}] Critical failure ({failure_context}) - NO RESET allowed")
                    return False
                
                # Check if this is a recoverable failure
                if failure_context in config['reset_on_specific_failures']:
                    print(f"🔄 [{robot_namespace}] Recoverable failure ({failure_context}) - reset allowed")
                    if config['require_manual_confirmation']:
                        response = input(f"Reset tree for {robot_namespace} after {failure_context}? (y/n): ")
                        return response.lower() == 'y'
                    return True
                
                # Default failure behavior - more conservative
                if config['auto_reset_on_failure']:
                    print(f"🔄 [{robot_namespace}] General failure - auto-reset enabled")
                    return True
                else:
                    print(f"🛑 [{robot_namespace}] Failure ({failure_context}) - auto-reset disabled")
                    return False
                    
            except Exception as bb_error:
                print(f"⚠️ [{robot_namespace}] Could not read failure context: {bb_error}")
                # Conservative: don't reset on unknown failures
                return False
        
        # Unknown status
        print(f"❓ [{robot_namespace}] Unknown completion status: {status} - no reset")
        return False
        
    except Exception as e:
        print(f"🚨 [{robot_namespace}] Error checking reset conditions: {e}")
        # Conservative approach: don't reset on errors
        return False

def log_tree_reset_event_improved(robot_namespace, should_reset, reason, iteration):
    """Improved logging for reset events"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if should_reset:
        print(f"\n🔄 [{timestamp}] TREE RESET EVENT:")
        print(f"   Robot: {robot_namespace}")
        print(f"   Iteration: {iteration}")
        print(f"   Reason: {reason}")
        print(f"   Action: RESETTING TREE")
        print(f"   Next iteration will be: {iteration + 1}")
    else:
        print(f"\n🛑 [{timestamp}] TREE COMPLETION EVENT:")
        print(f"   Robot: {robot_namespace}")
        print(f"   Iteration: {iteration}")
        print(f"   Reason: {reason}")
        print(f"   Action: STOPPING (no reset)")

def apply_improved_reset_patch(module_path="/root/workspace/src/behaviour_tree/behaviour_tree/my_behaviour_tree_modular.py"):
    """
    Apply the improved reset logic patch
    """
    print("🔧 Applying improved tree reset patch...")
    
    # The patch would replace the existing check_tree_reset_conditions function
    # with check_tree_reset_conditions_improved
    
    patch_info = {
        'function_to_replace': 'check_tree_reset_conditions',
        'new_function': 'check_tree_reset_conditions_improved',
        'config_to_replace': 'TREE_RESET_CONFIG',
        'new_config': 'IMPROVED_TREE_RESET_CONFIG'
    }
    
    print(f"✅ Patch ready - functions to replace: {patch_info}")
    return patch_info

# Quick configuration presets
def configure_for_debugging():
    """Configure for debugging - no automatic resets"""
    config = IMPROVED_TREE_RESET_CONFIG.copy()
    config.update({
        'auto_reset_on_success': False,
        'auto_reset_on_failure': False,
        'max_iterations': 1,
        'manual_reset_only': True,
        'debug_mode': True
    })
    return config

def configure_for_testing():
    """Configure for testing - controlled resets"""
    config = IMPROVED_TREE_RESET_CONFIG.copy()
    config.update({
        'auto_reset_on_success': True,
        'auto_reset_on_failure': False,
        'max_iterations': 5,
        'manual_reset_only': False,
        'debug_mode': True,
        'reset_delay': 0.5
    })
    return config

def configure_for_production():
    """Configure for production - conservative resets"""
    config = IMPROVED_TREE_RESET_CONFIG.copy()
    config.update({
        'auto_reset_on_success': True,
        'auto_reset_on_failure': False,
        'max_iterations': 20,
        'manual_reset_only': False,
        'debug_mode': False,
        'reset_delay': 1.0
    })
    return config

if __name__ == "__main__":
    print("=== Tree Reset Patch Demo ===")
    
    # Test the improved reset logic
    test_cases = [
        {
            'status': py_trees.common.Status.SUCCESS,
            'iteration': 1,
            'robot': 'turtlebot0'
        },
        {
            'status': py_trees.common.Status.FAILURE,
            'iteration': 2,
            'robot': 'turtlebot0'
        }
    ]
    
    for case in test_cases:
        completion_info = {
            'status': case['status'],
            'iteration': case['iteration'],
            'robot': case['robot'],
            'timestamp': time.time()
        }
        
        should_reset = check_tree_reset_conditions_improved(
            None, case['robot'], completion_info
        )
        
        reason = f"Status: {case['status']}"
        log_tree_reset_event_improved(
            case['robot'], should_reset, reason, case['iteration']
        )
