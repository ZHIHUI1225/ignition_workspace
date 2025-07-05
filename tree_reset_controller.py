#!/usr/bin/env python3
"""
Tree Reset Controller - Utility to control and monitor behavior tree resets
"""

import py_trees
import time

class TreeResetController:
    """Controller for managing behavior tree reset behavior"""
    
    def __init__(self, robot_namespace="turtlebot0"):
        self.robot_namespace = robot_namespace
        self.reset_config = {
            'auto_reset_on_success': True,     # Auto-reset when tree succeeds
            'auto_reset_on_failure': False,    # Don't auto-reset on failure
            'max_iterations': 10,              # Maximum iterations
            'reset_on_specific_failures': [    # Reset only on specific failure types
                'timeout',
                'minor_error',
                'trajectory_deviation'
            ],
            'no_reset_on_critical_failures': [ # Never reset on these
                'system_failure',
                'hardware_error',
                'safety_violation',
                'mpc_initialization_failure',
                'trajectory_loading_failure'
            ],
            'reset_delay': 1.0,               # Delay before reset
            'debug_mode': True                # Enable detailed logging
        }
    
    def should_reset(self, tree_status, iteration_count, failure_context=None):
        """
        Determine if tree should be reset
        
        Args:
            tree_status: py_trees.common.Status
            iteration_count: Current iteration number
            failure_context: Optional failure context from blackboard
            
        Returns:
            tuple: (should_reset: bool, reason: str)
        """
        
        # Check iteration limit
        if iteration_count >= self.reset_config['max_iterations']:
            return False, f"Maximum iterations ({self.reset_config['max_iterations']}) reached"
        
        # Handle SUCCESS
        if tree_status == py_trees.common.Status.SUCCESS:
            if self.reset_config['auto_reset_on_success']:
                return True, "Success - auto-reset enabled"
            else:
                return False, "Success - auto-reset disabled"
        
        # Handle FAILURE
        elif tree_status == py_trees.common.Status.FAILURE:
            if failure_context:
                # Check critical failures
                if failure_context in self.reset_config['no_reset_on_critical_failures']:
                    return False, f"Critical failure: {failure_context} - no reset allowed"
                
                # Check recoverable failures
                if failure_context in self.reset_config['reset_on_specific_failures']:
                    return True, f"Recoverable failure: {failure_context} - reset allowed"
            
            # Default failure behavior
            if self.reset_config['auto_reset_on_failure']:
                return True, "General failure - auto-reset enabled"
            else:
                return False, "General failure - auto-reset disabled"
        
        return False, f"Unknown status: {tree_status}"
    
    def configure_reset_behavior(self, **kwargs):
        """Update reset configuration"""
        for key, value in kwargs.items():
            if key in self.reset_config:
                old_value = self.reset_config[key]
                self.reset_config[key] = value
                print(f"🔧 Reset config updated: {key} = {value} (was {old_value})")
            else:
                print(f"⚠️  Unknown config key: {key}")
    
    def disable_all_resets(self):
        """Disable all automatic resets - useful for debugging"""
        self.configure_reset_behavior(
            auto_reset_on_success=False,
            auto_reset_on_failure=False
        )
        print("🛑 All automatic resets disabled")
    
    def enable_conservative_resets(self):
        """Enable only conservative resets (success only)"""
        self.configure_reset_behavior(
            auto_reset_on_success=True,
            auto_reset_on_failure=False
        )
        print("🔄 Conservative reset mode enabled (success only)")
    
    def enable_aggressive_resets(self):
        """Enable resets on both success and failure"""
        self.configure_reset_behavior(
            auto_reset_on_success=True,
            auto_reset_on_failure=True
        )
        print("🔄 Aggressive reset mode enabled (success and failure)")
    
    def report_failure_context(self, node_name, error_type, robot_namespace=None):
        """Report failure context to blackboard for reset decision"""
        if robot_namespace is None:
            robot_namespace = self.robot_namespace
            
        try:
            blackboard_client = py_trees.blackboard.Client(name="reset_controller")
            blackboard_client.register_key(
                key=f"{robot_namespace}/failure_context",
                access=py_trees.common.Access.WRITE
            )
            
            blackboard_client.set(f"{robot_namespace}/failure_context", error_type)
            print(f"📊 Failure context reported: {node_name} -> {error_type}")
            
        except Exception as e:
            print(f"❌ Failed to report failure context: {e}")

# Example usage functions
def create_debug_configuration():
    """Create a configuration for debugging (no automatic resets)"""
    controller = TreeResetController()
    controller.disable_all_resets()
    controller.configure_reset_behavior(
        max_iterations=3,  # Stop after 3 iterations for debugging
        debug_mode=True
    )
    return controller

def create_production_configuration():
    """Create a configuration for production use"""
    controller = TreeResetController()
    controller.enable_conservative_resets()
    controller.configure_reset_behavior(
        max_iterations=50,  # Allow more iterations
        debug_mode=False
    )
    return controller

def create_test_configuration():
    """Create a configuration for testing"""
    controller = TreeResetController()
    controller.enable_aggressive_resets()
    controller.configure_reset_behavior(
        max_iterations=5,
        reset_delay=0.1,  # Faster resets for testing
        debug_mode=True
    )
    return controller

if __name__ == "__main__":
    # Example: Create debug controller
    print("=== Tree Reset Controller Demo ===")
    
    # Debug mode - no automatic resets
    debug_controller = create_debug_configuration()
    
    # Test different scenarios
    scenarios = [
        (py_trees.common.Status.SUCCESS, 1, None),
        (py_trees.common.Status.FAILURE, 2, "timeout"),
        (py_trees.common.Status.FAILURE, 3, "system_failure"),
        (py_trees.common.Status.FAILURE, 4, "minor_error")
    ]
    
    for status, iteration, context in scenarios:
        should_reset, reason = debug_controller.should_reset(status, iteration, context)
        print(f"Status: {status}, Iteration: {iteration}, Context: {context}")
        print(f"  → Reset: {should_reset}, Reason: {reason}")
        print()
