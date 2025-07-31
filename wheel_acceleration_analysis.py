#!/usr/bin/env python3
"""
E-puck2 Wheel Acceleration Analysis
Calculates proper acceleration limits based on motor specifications and wheel dynamics
"""

import math

def calculate_wheel_dynamics():
    """Calculate wheel acceleration limits from motor specifications"""
    
    # Hardware specifications from e-puck2 documentation
    wheel_radius = 0.0205  # m (20.5mm radius)
    wheelbase = 0.053      # m (53mm between wheels)
    max_steps_per_second = 1200
    steps_per_revolution = 20
    gear_reduction_ratio = 50
    
    print("E-PUCK2 WHEEL ACCELERATION ANALYSIS")
    print("=" * 50)
    
    # Step 1: Calculate wheel angular velocity
    wheel_steps_per_second = max_steps_per_second / gear_reduction_ratio
    wheel_revolutions_per_second = wheel_steps_per_second / steps_per_revolution
    wheel_angular_velocity_max = wheel_revolutions_per_second * 2 * math.pi
    
    print(f"Motor specifications:")
    print(f"  Max motor steps/s: {max_steps_per_second}")
    print(f"  Gear reduction: {gear_reduction_ratio}:1")
    print(f"  Steps per revolution: {steps_per_revolution}")
    print()
    
    print(f"Wheel dynamics:")
    print(f"  Wheel steps/s: {wheel_steps_per_second}")
    print(f"  Wheel revolutions/s: {wheel_revolutions_per_second:.3f}")
    print(f"  Wheel angular velocity max: {wheel_angular_velocity_max:.3f} rad/s")
    print()
    
    # Step 2: Calculate linear velocity from wheel rotation
    linear_velocity_max = wheel_angular_velocity_max * wheel_radius
    print(f"Linear motion:")
    print(f"  Linear velocity max: {linear_velocity_max:.4f} m/s ({linear_velocity_max*100:.1f} cm/s)")
    
    # Step 3: Calculate robot angular velocity (differential drive)
    robot_angular_velocity_max = 2 * linear_velocity_max / wheelbase
    print(f"  Robot angular velocity max: {robot_angular_velocity_max:.3f} rad/s ({math.degrees(robot_angular_velocity_max):.1f}°/s)")
    print()
    
    # Step 4: Calculate acceleration limits
    # Assumption: reasonable acceleration time (stepper motors can change speed quickly)
    acceleration_times = [0.1, 0.2, 0.5, 1.0]  # seconds
    
    print("Acceleration limits for different acceleration times:")
    for accel_time in acceleration_times:
        wheel_angular_accel = wheel_angular_velocity_max / accel_time
        linear_accel = wheel_angular_accel * wheel_radius
        robot_angular_accel = robot_angular_velocity_max / accel_time
        
        print(f"  {accel_time}s to max speed:")
        print(f"    Wheel angular acceleration: {wheel_angular_accel:.2f} rad/s²")
        print(f"    Linear acceleration: {linear_accel:.3f} m/s²")
        print(f"    Robot angular acceleration: {robot_angular_accel:.2f} rad/s²")
        print()
    
    # Step 5: Physics constraints
    friction_coefficient = 0.7
    safety_factor = 0.8
    gravity = 9.81
    friction_limited_accel = friction_coefficient * safety_factor * gravity
    
    print("Physics constraints:")
    print(f"  Friction coefficient: {friction_coefficient}")
    print(f"  Safety factor: {safety_factor}")
    print(f"  Friction-limited acceleration: {friction_limited_accel:.3f} m/s²")
    print()
    
    # Recommended values (conservative approach)
    recommended_accel_time = 0.5  # seconds
    recommended_linear_accel = (wheel_angular_velocity_max / recommended_accel_time) * wheel_radius
    recommended_angular_accel = robot_angular_velocity_max / recommended_accel_time
    
    # Use the more conservative between wheel limit and friction limit
    final_linear_accel = min(recommended_linear_accel, friction_limited_accel)
    
    print("RECOMMENDED VALUES:")
    print(f"  Linear acceleration: {final_linear_accel:.3f} m/s² (wheel-limited)")
    print(f"  Angular acceleration: {recommended_angular_accel:.2f} rad/s²")
    print(f"  Acceleration time: {recommended_accel_time}s")
    
    return {
        'wheel_angular_velocity_max': wheel_angular_velocity_max,
        'linear_velocity_max': linear_velocity_max,
        'robot_angular_velocity_max': robot_angular_velocity_max,
        'linear_acceleration_max': final_linear_accel,
        'angular_acceleration_max': recommended_angular_accel,
        'wheel_angular_acceleration_max': wheel_angular_velocity_max / recommended_accel_time
    }

if __name__ == "__main__":
    results = calculate_wheel_dynamics()
