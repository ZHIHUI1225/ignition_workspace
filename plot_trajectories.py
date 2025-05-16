#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Circle

def load_trajectory(file_path):
    """Load trajectory data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'Trajectory' in data:
                return data['Trajectory']
            else:
                print(f"Warning: 'Trajectory' key not found in {file_path}")
                return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def main():
    # Set up the figure
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Color map for the robots
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    relay_points = []
    
    # Plot each robot's trajectory
    for i in range(5):
        file_path = f"/root/workspace/data/tb{i}_DiscreteTrajectory.json"
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        trajectory = load_trajectory(file_path)
        
        if not trajectory:
            continue
            
        x_coords = [point[0] for point in trajectory]
        y_coords = [point[1] for point in trajectory]
        
        # Plot the trajectory
        plt.plot(x_coords, y_coords, color=colors[i], label=f'TB{i}', linewidth=2)
        
        # Mark start point with a circle
        plt.plot(x_coords[0], y_coords[0], 'o', color=colors[i], markersize=8)
        
        # Mark end point with a star and save as relay point
        plt.plot(x_coords[-1], y_coords[-1], '*', color=colors[i], markersize=12)
        relay_points.append((x_coords[-1], y_coords[-1]))
    
    # Add relay points visualization
    for i, (x, y) in enumerate(relay_points):
        circle = Circle((x, y), 0.08, fill=False, linestyle='--', color='black', alpha=0.7)
        ax.add_patch(circle)
        plt.text(x+0.1, y+0.1, f'RP{i}', fontsize=12)
    
    # Set plot properties
    plt.title('Turtlebot Trajectories with Relay Points', fontsize=16)
    plt.xlabel('X Coordinate (m)', fontsize=14)
    plt.ylabel('Y Coordinate (m)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Equal aspect ratio to maintain proper scale
    plt.axis('equal')
    
    # Add annotations
    plt.text(0.02, 0.02, 'o: Start Point\n*: End Point\n--: Relay Point Range',
             transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Save the plot
    plt.savefig('/root/workspace/trajectories_plot.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to /root/workspace/trajectories_plot.png")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
