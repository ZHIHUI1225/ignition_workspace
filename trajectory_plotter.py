#!/usr/bin/env python3
"""
Trajectory Plotting Tool for Robot Navigation Analysis

This script plots trajectory data from both original and replanned trajectory files.
It visualizes x, y, theta (orientation), velocity, and angular velocity information.

Author: GitHub Copilot
Date: 2024
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

class TrajectoryPlotter:
    """A comprehensive trajectory plotter for robot navigation analysis."""
    
    def __init__(self, original_file=None, replanned_file=None):
        """
        Initialize the trajectory plotter.
        
        Args:
            original_file (str): Path to original trajectory JSON file
            replanned_file (str): Path to replanned trajectory JSON file
        """
        self.original_file = original_file
        self.replanned_file = replanned_file
        self.original_data = None
        self.replanned_data = None
        
    def load_trajectory_data(self, file_path):
        """
        Load trajectory data from JSON file.
        
        Args:
            file_path (str): Path to trajectory JSON file
            
        Returns:
            dict: Parsed trajectory data with x, y, theta, v, omega arrays
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            trajectory = data['Trajectory']
            
            # Extract components: [x, y, theta, v, omega]
            x = [point[0] for point in trajectory]
            y = [point[1] for point in trajectory]
            theta = [point[2] for point in trajectory]
            v = [point[3] for point in trajectory]
            omega = [point[4] for point in trajectory]
            
            return {
                'x': np.array(x),
                'y': np.array(y), 
                'theta': np.array(theta),
                'v': np.array(v),
                'omega': np.array(omega),
                'time': np.arange(len(x))  # Assuming uniform time steps
            }
            
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {file_path}")
            return None
        except KeyError:
            print(f"Error: 'Trajectory' key not found in {file_path}")
            return None
    
    def load_data(self):
        """Load both original and replanned trajectory data."""
        if self.original_file:
            print(f"Loading original trajectory from: {self.original_file}")
            self.original_data = self.load_trajectory_data(self.original_file)
            
        if self.replanned_file:
            print(f"Loading replanned trajectory from: {self.replanned_file}")
            self.replanned_data = self.load_trajectory_data(self.replanned_file)
    
    def plot_xy_trajectory(self, ax=None, show_arrows=True, arrow_interval=20):
        """
        Plot 2D trajectory in x-y plane with orientation arrows.
        
        Args:
            ax: Matplotlib axis object (optional)
            show_arrows (bool): Whether to show orientation arrows
            arrow_interval (int): Interval between orientation arrows
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot original trajectory
        if self.original_data is not None:
            ax.plot(self.original_data['x'], self.original_data['y'], 
                   'b-', linewidth=2, label='Original Trajectory', alpha=0.8)
            
            # Add orientation arrows for original trajectory
            if show_arrows:
                for i in range(0, len(self.original_data['x']), arrow_interval):
                    x, y, theta = self.original_data['x'][i], self.original_data['y'][i], self.original_data['theta'][i]
                    dx = 0.05 * np.cos(theta)
                    dy = 0.05 * np.sin(theta)
                    ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.01, 
                            fc='blue', ec='blue', alpha=0.6)
        
        # Plot replanned trajectory
        if self.replanned_data is not None:
            ax.plot(self.replanned_data['x'], self.replanned_data['y'], 
                   'r--', linewidth=2, label='Replanned Trajectory', alpha=0.8)
            
            # Add orientation arrows for replanned trajectory
            if show_arrows:
                for i in range(0, len(self.replanned_data['x']), arrow_interval):
                    x, y, theta = self.replanned_data['x'][i], self.replanned_data['y'][i], self.replanned_data['theta'][i]
                    dx = 0.05 * np.cos(theta)
                    dy = 0.05 * np.sin(theta)
                    ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.01, 
                            fc='red', ec='red', alpha=0.6)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Robot Trajectory Comparison (X-Y Plane)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
        
        return ax
    
    def plot_theta_evolution(self, ax=None):
        """
        Plot theta (orientation) evolution over time.
        
        Args:
            ax: Matplotlib axis object (optional)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        if self.original_data is not None:
            ax.plot(self.original_data['time'], self.original_data['theta'], 
                   'b-', linewidth=2, label='Original θ', alpha=0.8)
        
        if self.replanned_data is not None:
            ax.plot(self.replanned_data['time'], self.replanned_data['theta'], 
                   'r--', linewidth=2, label='Replanned θ', alpha=0.8)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Orientation θ (rad)')
        ax.set_title('Robot Orientation Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_velocity_profiles(self, ax=None):
        """
        Plot linear and angular velocity profiles.
        
        Args:
            ax: Matplotlib axis object (optional)
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        else:
            ax1, ax2 = ax
        
        # Linear velocity
        if self.original_data is not None:
            ax1.plot(self.original_data['time'], self.original_data['v'], 
                    'b-', linewidth=2, label='Original v', alpha=0.8)
        
        if self.replanned_data is not None:
            ax1.plot(self.replanned_data['time'], self.replanned_data['v'], 
                    'r--', linewidth=2, label='Replanned v', alpha=0.8)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Linear Velocity v (m/s)')
        ax1.set_title('Linear Velocity Profile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Angular velocity
        if self.original_data is not None:
            ax2.plot(self.original_data['time'], self.original_data['omega'], 
                    'b-', linewidth=2, label='Original ω', alpha=0.8)
        
        if self.replanned_data is not None:
            ax2.plot(self.replanned_data['time'], self.replanned_data['omega'], 
                    'r--', linewidth=2, label='Replanned ω', alpha=0.8)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Angular Velocity ω (rad/s)')
        ax2.set_title('Angular Velocity Profile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        return ax1, ax2
    
    def plot_comprehensive_analysis(self, save_path=None):
        """
        Create a comprehensive analysis plot with all trajectory aspects.
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. X-Y trajectory (spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self.plot_xy_trajectory(ax1)
        
        # 2. Theta evolution
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_theta_evolution(ax2)
        
        # 3. Linear velocity
        ax3 = fig.add_subplot(gs[1, 2])
        if self.original_data is not None:
            ax3.plot(self.original_data['time'], self.original_data['v'], 
                    'b-', linewidth=2, label='Original v', alpha=0.8)
        if self.replanned_data is not None:
            ax3.plot(self.replanned_data['time'], self.replanned_data['v'], 
                    'r--', linewidth=2, label='Replanned v', alpha=0.8)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Linear Velocity (m/s)')
        ax3.set_title('Linear Velocity')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Angular velocity
        ax4 = fig.add_subplot(gs[2, 0])
        if self.original_data is not None:
            ax4.plot(self.original_data['time'], self.original_data['omega'], 
                    'b-', linewidth=2, label='Original ω', alpha=0.8)
        if self.replanned_data is not None:
            ax4.plot(self.replanned_data['time'], self.replanned_data['omega'], 
                    'r--', linewidth=2, label='Replanned ω', alpha=0.8)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Angular Velocity (rad/s)')
        ax4.set_title('Angular Velocity')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. X position over time
        ax5 = fig.add_subplot(gs[2, 1])
        if self.original_data is not None:
            ax5.plot(self.original_data['time'], self.original_data['x'], 
                    'b-', linewidth=2, label='Original X', alpha=0.8)
        if self.replanned_data is not None:
            ax5.plot(self.replanned_data['time'], self.replanned_data['x'], 
                    'r--', linewidth=2, label='Replanned X', alpha=0.8)
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('X Position (m)')
        ax5.set_title('X Position Evolution')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Y position over time
        ax6 = fig.add_subplot(gs[2, 2])
        if self.original_data is not None:
            ax6.plot(self.original_data['time'], self.original_data['y'], 
                    'b-', linewidth=2, label='Original Y', alpha=0.8)
        if self.replanned_data is not None:
            ax6.plot(self.replanned_data['time'], self.replanned_data['y'], 
                    'r--', linewidth=2, label='Replanned Y', alpha=0.8)
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Y Position (m)')
        ax6.set_title('Y Position Evolution')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.suptitle('Comprehensive Robot Trajectory Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def print_trajectory_statistics(self):
        """Print statistical analysis of the trajectories."""
        print("\n" + "="*60)
        print("TRAJECTORY ANALYSIS STATISTICS")
        print("="*60)
        
        if self.original_data is not None:
            print("\nORIGINAL TRAJECTORY:")
            print(f"  Length: {len(self.original_data['x'])} points")
            print(f"  X range: [{self.original_data['x'].min():.3f}, {self.original_data['x'].max():.3f}] m")
            print(f"  Y range: [{self.original_data['y'].min():.3f}, {self.original_data['y'].max():.3f}] m")
            print(f"  θ range: [{self.original_data['theta'].min():.3f}, {self.original_data['theta'].max():.3f}] rad")
            print(f"  Velocity range: [{self.original_data['v'].min():.3f}, {self.original_data['v'].max():.3f}] m/s")
            print(f"  Angular velocity range: [{self.original_data['omega'].min():.3f}, {self.original_data['omega'].max():.3f}] rad/s")
            
            # Calculate path length
            dx = np.diff(self.original_data['x'])
            dy = np.diff(self.original_data['y'])
            path_length = np.sum(np.sqrt(dx**2 + dy**2))
            print(f"  Total path length: {path_length:.3f} m")
        
        if self.replanned_data is not None:
            print("\nREPLANNED TRAJECTORY:")
            print(f"  Length: {len(self.replanned_data['x'])} points")
            print(f"  X range: [{self.replanned_data['x'].min():.3f}, {self.replanned_data['x'].max():.3f}] m")
            print(f"  Y range: [{self.replanned_data['y'].min():.3f}, {self.replanned_data['y'].max():.3f}] m")
            print(f"  θ range: [{self.replanned_data['theta'].min():.3f}, {self.replanned_data['theta'].max():.3f}] rad")
            print(f"  Velocity range: [{self.replanned_data['v'].min():.3f}, {self.replanned_data['v'].max():.3f}] m/s")
            print(f"  Angular velocity range: [{self.replanned_data['omega'].min():.3f}, {self.replanned_data['omega'].max():.3f}] rad/s")
            
            # Calculate path length
            dx = np.diff(self.replanned_data['x'])
            dy = np.diff(self.replanned_data['y'])
            path_length = np.sum(np.sqrt(dx**2 + dy**2))
            print(f"  Total path length: {path_length:.3f} m")
        
        print("\n" + "="*60)


def main():
    """Main function to run the trajectory plotter."""
    parser = argparse.ArgumentParser(description='Plot robot trajectory data')
    parser.add_argument('--original', '-o', 
                       help='Path to original trajectory JSON file')
    parser.add_argument('--replanned', '-r', 
                       help='Path to replanned trajectory JSON file')
    parser.add_argument('--output', '-out', 
                       help='Output path for saving the plot')
    parser.add_argument('--show', action='store_true', 
                       help='Show the plot interactively')
    parser.add_argument('--stats', action='store_true', 
                       help='Print trajectory statistics')
    
    args = parser.parse_args()
    robot_id=0
    # Default file paths if not provided
    if not args.original and not args.replanned:
        # Try to find default files in the workspace
        workspace_data = "/root/workspace/data/simple_maze"
        
        default_original = os.path.join(workspace_data, f"tb{robot_id}_Trajectory.json")
        default_replanned = os.path.join(workspace_data, f"tb{robot_id}_Trajectory_replanned.json")
        
        if os.path.exists(default_original):
            args.original = default_original
        if os.path.exists(default_replanned):
            args.replanned = default_replanned
    
    # Create plotter
    plotter = TrajectoryPlotter(args.original, args.replanned)
    
    # Load data
    plotter.load_data()
    
    # Print statistics if requested
    if args.stats:
        plotter.print_trajectory_statistics()
    
    # Create plot
    if plotter.original_data is not None or plotter.replanned_data is not None:
        fig = plotter.plot_comprehensive_analysis(args.output)
        
        if args.show:
            plt.show()
        else:
            # Save to default location if no output specified
            if not args.output:
                default_output =f"/root/workspace/trajectory{robot_id}_analysis.png"
                plt.savefig(default_output, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {default_output}")
    else:
        print("No trajectory data loaded. Please check file paths.")


if __name__ == "__main__":
    main()
