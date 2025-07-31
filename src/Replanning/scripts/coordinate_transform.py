#!/usr/bin/env python3
"""
Coordinate Transformation Utilities

This module provides coordinate transformation functions for converting between
camera pixel coordinates and world meter coordinates.

Camera coordinate system: Origin at left-upper corner, X-right, Y-down (pixels)
World coordinate system: Origin at left-lower corner, X-right, Y-up (meters)
Conversion factor: 0.0023 (pixels to meters)
Image dimensions: 600x1100 (cropped from 665-65=600 height, 1150-50=1100 width)
"""

import numpy as np

# Image dimensions for coordinate transformation
IMG_HEIGHT = 600  # Cropped image height (665-65=600)
IMG_WIDTH = 1100   # Cropped image width (1150-50=1100)
PIXEL_TO_METER_SCALE = 0.0023  # Conversion factor from pixels to meters

def convert_pixel_to_world_coordinates(pixel_pos):
    """
    Convert a single position from camera pixel coordinates to world meter coordinates.
    
    Camera coordinate system: Origin at left-upper corner, X-right, Y-down (pixels)
    World coordinate system: Origin at left-lower corner, X-right, Y-up (meters)
    
    Args:
        pixel_pos: [x, y] position in pixel coordinates (can be list, tuple, or numpy array)
    
    Returns:
        numpy array: [x, y] position in world meter coordinates
    """
    if isinstance(pixel_pos, (list, tuple)) and len(pixel_pos) >= 2:
        pixel_x = pixel_pos[0]
        pixel_y = pixel_pos[1]
    elif isinstance(pixel_pos, np.ndarray) and pixel_pos.size >= 2:
        pixel_x = pixel_pos[0]
        pixel_y = pixel_pos[1]
    else:
        # Handle malformed position data
        return np.array([0.0, 0.0])
    
    # Transform from camera pixel coordinates to world coordinates
    # Camera origin: left-upper corner, World origin: left-lower corner
    # Camera X-right → World X-right (same direction)
    # Camera Y-down → World Y-up (flip and offset)
    
    # World coordinates in pixels (relative to left-lower corner)
    world_x_px = pixel_x  # X direction is the same
    world_y_px = IMG_HEIGHT - pixel_y  # Flip Y axis for world coordinates
    
    # Convert to world coordinates in meters
    world_x = world_x_px * PIXEL_TO_METER_SCALE  # Convert pixels to meters
    world_y = world_y_px * PIXEL_TO_METER_SCALE  # Convert pixels to meters
    
    return np.array([world_x, world_y])

def convert_camera_angle_to_world_angle(camera_angle):
    """
    Convert angle from camera coordinate frame to world coordinate frame.
    
    Camera coordinate system: Origin at left-upper corner, X-right, Y-down
    World coordinate system: Origin at left-lower corner, X-right, Y-up
    
    Since the Y-axis is flipped, angles need to be negated.
    
    Args:
        camera_angle: Angle in radians in camera coordinate frame
    
    Returns:
        float: Angle in radians in world coordinate frame
    """
    # When Y-axis is flipped, angles are negated
    # This maintains the correct orientation relative to the coordinate system
    world_angle = -camera_angle
    
    # Normalize angle to [-π, π] range
    while world_angle > np.pi:
        world_angle -= 2 * np.pi
    while world_angle < -np.pi:
        world_angle += 2 * np.pi
    
    return world_angle

def convert_world_angle_to_camera_angle(world_angle):
    """
    Convert angle from world coordinate frame to camera coordinate frame.
    
    World coordinate system: Origin at left-lower corner, X-right, Y-up
    Camera coordinate system: Origin at left-upper corner, X-right, Y-down
    
    Args:
        world_angle: Angle in radians in world coordinate frame
    
    Returns:
        float: Angle in radians in camera coordinate frame
    """
    # When Y-axis is flipped, angles are negated
    camera_angle = -world_angle
    
    # Normalize angle to [-π, π] range
    while camera_angle > np.pi:
        camera_angle -= 2 * np.pi
    while camera_angle < -np.pi:
        camera_angle += 2 * np.pi
    
    return camera_angle

def convert_pixel_positions_to_world_meters(pixel_positions):
    """
    Convert multiple positions from camera pixel coordinates to world meter coordinates.
    
    Args:
        pixel_positions: List of [x, y] positions in pixel coordinates
    
    Returns:
        List of [x, y] positions in world meter coordinates
    """
    world_positions = []
    for pos in pixel_positions:
        world_pos = convert_pixel_to_world_coordinates(pos)
        world_positions.append(world_pos.tolist())
    
    return world_positions

def convert_world_to_pixel_coordinates(world_pos):
    """
    Convert a single position from world meter coordinates to camera pixel coordinates.
    
    World coordinate system: Origin at left-lower corner, X-right, Y-up (meters)
    Camera coordinate system: Origin at left-upper corner, X-right, Y-down (pixels)
    
    Args:
        world_pos: [x, y] position in world meter coordinates (can be list, tuple, or numpy array)
    
    Returns:
        numpy array: [x, y] position in camera pixel coordinates
    """
    if isinstance(world_pos, (list, tuple)) and len(world_pos) >= 2:
        world_x = world_pos[0]
        world_y = world_pos[1]
    elif isinstance(world_pos, np.ndarray) and world_pos.size >= 2:
        world_x = world_pos[0]
        world_y = world_pos[1]
    else:
        # Handle malformed position data
        return np.array([0.0, 0.0])
    
    # Convert from world coordinates in meters to pixels
    world_x_px = world_x / PIXEL_TO_METER_SCALE
    world_y_px = world_y / PIXEL_TO_METER_SCALE
    
    # Transform from world coordinates to camera pixel coordinates
    # World origin: left-lower corner, Camera origin: left-upper corner
    # World X-right → Camera X-right (same direction)
    # World Y-up → Camera Y-down (flip and offset)
    
    pixel_x = world_x_px  # X direction is the same
    pixel_y = IMG_HEIGHT - world_y_px  # Flip Y axis back to camera coordinates
    
    return np.array([pixel_x, pixel_y])

def convert_pixel_data_to_meters(phi, l, r):
    """
    Convert planning data from pixels to meters where needed and transform angles to world frame.
    
    Args:
        phi: Angular data in camera frame (radians) - will be converted to world frame
        l: Length data in pixels (converted to meters)
        r: Radius data in pixels (converted to meters)
    
    Returns:
        Tuple (phi_world, l_meters, r_meters) with converted units and transformed angles
    """
    # Convert angles from camera frame to world frame
    if isinstance(phi, (list, tuple)):
        phi_world = [convert_camera_angle_to_world_angle(angle) for angle in phi]
    elif isinstance(phi, np.ndarray):
        phi_world = np.array([convert_camera_angle_to_world_angle(angle) for angle in phi])
    else:
        phi_world = convert_camera_angle_to_world_angle(phi)
    
    # Convert lengths and radii from pixels to meters
    if isinstance(l, (list, tuple)):
        l_meters = [val * PIXEL_TO_METER_SCALE for val in l]
    elif isinstance(l, np.ndarray):
        l_meters = l * PIXEL_TO_METER_SCALE
    else:
        l_meters = l * PIXEL_TO_METER_SCALE
    
    if isinstance(r, (list, tuple)):
        r_meters = [val * PIXEL_TO_METER_SCALE for val in r]
    elif isinstance(r, np.ndarray):
        r_meters = r * PIXEL_TO_METER_SCALE
    else:
        r_meters = r * PIXEL_TO_METER_SCALE
        
    return phi_world, l_meters, r_meters

def get_image_dimensions():
    """
    Get the image dimensions used for coordinate transformation.
    
    Returns:
        tuple: (height, width) in pixels
    """
    return IMG_HEIGHT, IMG_WIDTH

def get_pixel_to_meter_scale():
    """
    Get the pixel to meter conversion scale factor.
    
    Returns:
        float: Scale factor (pixels to meters)
    """
    return PIXEL_TO_METER_SCALE

# For backward compatibility
def convert_pixel_positions_to_world_meters_legacy(pixel_positions):
    """
    Legacy function name for backward compatibility.
    Use convert_pixel_positions_to_world_meters instead.
    """
    return convert_pixel_positions_to_world_meters(pixel_positions)
