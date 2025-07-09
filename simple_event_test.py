#!/usr/bin/env python3
"""
Simple test for event-driven behavior concept.
Focuses on the core concept without ROS dependencies.
"""

import time
import threading
import math
import psutil
import os

class Position:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Object:
    def __init__(self, name, position=None):
        self.name = name
        self.position = position or Position(0, 0)


class PollingBehavior:
    """Traditional polling-based behavior"""
    
    def __init__(self, name, distance_threshold=0.14):
        self.name = name
        self.distance_threshold = distance_threshold
        self.parcel = Object("parcel", Position(1.0, 1.0))
        self.relay = Object("relay", Position(2.0, 2.0))
        self.check_count = 0
        self.operations_count = 0
    
    def calculate_distance(self):
        """Calculate distance between parcel and relay"""
        dx = self.parcel.position.x - self.relay.position.x
        dy = self.parcel.position.y - self.relay.position.y
        self.operations_count += 1  # Count distance calculation as an operation
        return math.sqrt(dx*dx + dy*dy)
    
    def check_condition(self):
        """Check if parcel is within range of relay"""
        self.check_count += 1
        distance = self.calculate_distance()
        is_in_range = distance <= self.distance_threshold
        
        return is_in_range, distance
    
    def update(self):
        """Traditional polling update - checks every time"""
        is_in_range, distance = self.check_condition()
        
        if is_in_range:
            print(f"[{self.name}] SUCCESS: parcel in range of relay (distance={distance:.3f}m)")
            return "SUCCESS"
        else:
            print(f"[{self.name}] RUNNING: waiting for parcel to enter range (distance={distance:.3f}m)")
            return "RUNNING"


class EventDrivenBehavior:
    """Event-driven behavior"""
    
    def __init__(self, name, distance_threshold=0.14, delta_threshold=0.01):
        self.name = name
        self.distance_threshold = distance_threshold
        self.delta_threshold = delta_threshold
        self.parcel = Object("parcel", Position(1.0, 1.0))
        self.relay = Object("relay", Position(2.0, 2.0))
        self.prev_distance = float('inf')
        self.last_distance_to_threshold = float('inf')
        self.distance_changed = False
        self.check_count = 0
        self.operations_count = 0
        self.last_check_time = 0
        self.cached_in_range = False
    
    def calculate_distance(self):
        """Calculate distance between parcel and relay"""
        dx = self.parcel.position.x - self.relay.position.x
        dy = self.parcel.position.y - self.relay.position.y
        self.operations_count += 1  # Count distance calculation as an operation
        return math.sqrt(dx*dx + dy*dy)
    
    def update_parcel_position(self, x, y):
        """Update parcel position and check for threshold crossing"""
        self.parcel.position.x = x
        self.parcel.position.y = y
        
        # Calculate new distance
        current_distance = self.calculate_distance()
        
        # Calculate how far from threshold
        distance_to_threshold = current_distance - self.distance_threshold
        
        # Check if we've crossed the threshold
        if self.prev_distance != float('inf'):
            # Check for threshold crossing (sign change)
            threshold_crossed = (distance_to_threshold * self.last_distance_to_threshold) <= 0
            
            # Check for significant approach to threshold
            delta = abs(distance_to_threshold - self.last_distance_to_threshold)
            significant_change = delta > self.delta_threshold
            
            if threshold_crossed or significant_change:
                self.distance_changed = True
                event_type = "threshold crossing" if threshold_crossed else "significant approach"
                print(f"[{self.name}] ⚡ Event: {event_type} detected (distance: {current_distance:.3f}m)")
        
        # Update stored values
        self.prev_distance = current_distance
        self.last_distance_to_threshold = distance_to_threshold
    
    def should_check_condition(self):
        """Determine if we should perform a full check"""
        current_time = time.time()
        time_since_last_check = current_time - self.last_check_time
        
        # Check at least once per second
        forced_check = time_since_last_check > 1.0
        
        # Check if distance changed significantly
        should_check = self.distance_changed or forced_check
        
        if should_check:
            self.distance_changed = False
            self.last_check_time = current_time
        
        return should_check
    
    def check_condition(self):
        """Check if parcel is within range of relay"""
        self.check_count += 1
        distance = self.calculate_distance()
        is_in_range = distance <= self.distance_threshold
        
        # Cache the result
        self.cached_in_range = is_in_range
        
        return is_in_range, distance
    
    def update(self):
        """Event-driven update - only checks when conditions change"""
        # Check if we need to perform a full condition check
        if self.should_check_condition():
            is_in_range, distance = self.check_condition()
            if is_in_range:
                print(f"[{self.name}] ✓ SUCCESS: parcel in range (distance={distance:.3f}m)")
                return "SUCCESS"
            else:
                print(f"[{self.name}] RUNNING: waiting for parcel (distance={distance:.3f}m)")
                return "RUNNING"
        else:
            # Use cached result
            self.operations_count += 1  # Count the cache lookup as an operation
            if self.cached_in_range:
                print(f"[{self.name}] ✓ SUCCESS: parcel in range (cached result - no recalculation)")
                return "SUCCESS"
            else:
                print(f"[{self.name}] RUNNING: waiting for parcel (cached result - no recalculation)")
                return "RUNNING"


def test_polling():
    """Test polling behavior"""
    behavior = PollingBehavior("PollingTest")
    
    print("\n=== Testing Polling Behavior ===")
    print("Running with standard polling - checks on every update...\n")
    
    # Run for iterations
    for i in range(10):
        status = behavior.update()
        
        # Halfway through, move the parcel closer to the relay point
        if i == 5:
            behavior.parcel.position.x = 1.95
            behavior.parcel.position.y = 1.95
            print(f"Updated parcel position to near relay point [1.95, 1.95]")
        
        time.sleep(0.5)
    
    return behavior.check_count, behavior.operations_count


def test_event_driven():
    """Test event-driven behavior"""
    behavior = EventDrivenBehavior("EventDrivenTest")
    
    print("\n=== Testing Event-Driven Behavior ===")
    print("Running with event-driven approach - only checks on significant changes...\n")
    
    # Run for iterations
    for i in range(10):
        status = behavior.update()
        
        # Make small moves that don't cross threshold (noise)
        if i == 2:
            behavior.update_parcel_position(1.01, 1.01)
            print(f"Updated parcel position slightly [1.01, 1.01] (noise)")
        
        # Make a significant move that approaches threshold
        if i == 4:
            behavior.update_parcel_position(1.5, 1.5)
            print(f"Updated parcel position toward relay [1.5, 1.5] (approaching)")
        
        # Cross the threshold
        if i == 6:
            behavior.update_parcel_position(1.95, 1.95)
            print(f"Updated parcel position to near relay [1.95, 1.95] (crossing threshold)")
        
        time.sleep(0.5)
    
    return behavior.check_count, behavior.operations_count


def main():
    """Main function to run tests"""
    print("\nTesting two behavior implementations:")
    print("  1. Traditional polling - checks conditions on every update")
    print("  2. Event-driven - only checks when conditions might have changed\n")
    
    print("Running tests...")
    p_checks, p_ops = test_polling()
    time.sleep(1)
    e_checks, e_ops = test_event_driven()
    
    print("\n=== Results ===")
    print(f"Polling behavior:      {p_checks} checks, {p_ops} operations performed")
    print(f"Event-driven behavior: {e_checks} checks, {e_ops} operations performed")
    
    check_reduction = (1 - e_checks / p_checks) * 100 if p_checks > 0 else 0
    op_reduction = (1 - e_ops / p_ops) * 100 if p_ops > 0 else 0
    
    print(f"\nReduction in condition checks: {check_reduction:.1f}%")
    print(f"Reduction in operations:      {op_reduction:.1f}%")
    
    print("\nThe event-driven approach reduces CPU usage by only performing calculations")
    print("when something significant has changed or when the threshold is crossed.")
    print("This makes the behavior tree more efficient during waiting periods.")


if __name__ == "__main__":
    main()
