#!/usr/bin/env python3
"""
Safe Corridor Generator and Visualizer
This script generates and visualizes safe corridors between waypoints for path planning.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from shapely.geometry import LineString, Polygon

# Add paths and load configuration
sys.path.append('/root/workspace/config')
sys.path.append('/root/workspace/src/Replanning/scripts')
try:
    from config_loader import config
except ImportError:
    print("Error: Could not import config_loader. Please ensure config files are in place.")
    sys.exit(1)

# Import required modules
from GenerateMatrix import load_reeb_graph_from_file
from Planing_functions import load_WayPointFlag_from_file, check_collision, get_safe_corridor
from Environment import Environment

def generate_safe_corridor_detailed(reeb_graph, waypoints_file_path, environment_file, step_size=0.5, max_distance=100):
    """
    Generate safe corridor with detailed visualization information using the existing get_safe_corridor function
    
    Parameters:
    - reeb_graph: The loaded graph
    - waypoints_file_path: Path to waypoints file
    - environment_file: Path to environment file
    - step_size: Step size for collision checking
    - max_distance: Maximum distance to check for corridor bounds
    
    Returns:
    - safe_corridor: List of corridor bounds [slope, y_min, y_max]
    - corridor_info: Detailed information for visualization
    - loaded_environment: Environment object
    """
    
    print("Using get_safe_corridor function from Planing_functions...")
    
    # Use the existing get_safe_corridor function
    safe_corridor, Distance, Angle, vertex = get_safe_corridor(reeb_graph, waypoints_file_path, environment_file, step_size, max_distance)
    
    # Load waypoints and environment for additional information
    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
    loaded_environment = Environment.load_from_file(environment_file)
    
    print(f"Generated {len(safe_corridor)} corridor segments")
    
    # Create detailed corridor information for visualization
    corridor_info = []
    for i in range(len(safe_corridor)):
        # Get waypoint positions
        start_pos = reeb_graph.nodes[Waypoints[i]].configuration
        end_pos = reeb_graph.nodes[Waypoints[i+1]].configuration
        
        # Calculate distance and angle
        distance = Distance[i][0] if hasattr(Distance[i], '__getitem__') else Distance[i]
        angle = Angle[i][0] if hasattr(Angle[i], '__getitem__') else Angle[i]
        
        # Get corridor parameters
        slope = safe_corridor[i][0]
        y_min = safe_corridor[i][1]
        y_max = safe_corridor[i][2]
        
        # Get corridor boundary vertices
        corridor_bounds = vertex[i] if vertex[i] is not None else []
        
        print(f"Corridor {i+1}: slope={slope:.4f}, y_range=[{y_min:.2f}, {y_max:.2f}], distance={distance:.2f}")
        
        corridor_info.append({
            'segment_id': i,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'distance': distance,
            'angle': angle,
            'slope': slope,
            'y_min': y_min,
            'y_max': y_max,
            'corridor_bounds': corridor_bounds.tolist() if hasattr(corridor_bounds, 'tolist') else corridor_bounds,
            'waypoint_flag': Flagb[i] if i < len(Flagb) else 0
        })
    
    return safe_corridor, corridor_info, loaded_environment

def plot_comprehensive_results(reeb_graph, corridor_info, loaded_environment, Waypoints, Flagb, 
                             ga_result_file=None, planning_result_file=None, save_path=None):
    """
    Plot the safe corridors with GA planning results and Planning_Path optimization results
    """
    print("\nGenerating comprehensive visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw environment boundary from coord_bounds
    coord_bounds = loaded_environment.coord_bounds
    x_min, x_max, y_min, y_max = coord_bounds
    
    # Draw boundary walls as thick black lines
    boundary_width = 3
    ax.plot([x_min, x_max], [y_min, y_min], 'k-', linewidth=boundary_width, label='Environment Boundary')  
    ax.plot([x_min, x_max], [y_max, y_max], 'k-', linewidth=boundary_width)  
    ax.plot([x_min, x_min], [y_min, y_max], 'k-', linewidth=boundary_width)  
    ax.plot([x_max, x_max], [y_min, y_max], 'k-', linewidth=boundary_width)  
    
    # Draw environment obstacles
    loaded_environment.draw("gray")
    
    # Plot safe corridors
    for info in corridor_info:
        i = info['segment_id']
        corridor_bounds = info['corridor_bounds']
        
        # Only plot if corridor_bounds is available and not empty
        if corridor_bounds and len(corridor_bounds) > 0:
            corridor_bounds = np.array(corridor_bounds)
            
            # Plot corridor boundary
            ax.fill(corridor_bounds[:, 0], corridor_bounds[:, 1], 
                    alpha=0.2, color='lightblue', 
                    label='Safe Corridor' if i == 0 else "")
            ax.plot(corridor_bounds[:, 0], corridor_bounds[:, 1], 
                    'b--', linewidth=1, alpha=0.5)
        else:
            # Fallback: plot a simple corridor representation
            start_pos = info['start_pos']
            end_pos = info['end_pos']
            width = info['y_max'] - info['y_min']
            
            # Create a simple rectangular corridor representation
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Unit vector along the line
                ux = dx / length
                uy = dy / length
                
                # Perpendicular vector
                px = -uy * width / 2
                py = ux * width / 2
                
                # Create corridor bounds
                bounds = np.array([
                    [start_pos[0] + px, start_pos[1] + py],
                    [end_pos[0] + px, end_pos[1] + py],
                    [end_pos[0] - px, end_pos[1] - py],
                    [start_pos[0] - px, start_pos[1] - py],
                    [start_pos[0] + px, start_pos[1] + py]  # Close polygon
                ])
                
                ax.fill(bounds[:, 0], bounds[:, 1], 
                        alpha=0.2, color='lightblue', 
                        label='Safe Corridor' if i == 0 else "")
                ax.plot(bounds[:, 0], bounds[:, 1], 
                        'b--', linewidth=1, alpha=0.5)
    
    # Plot waypoint connection lines (reference path)
    for info in corridor_info:
        i = info['segment_id']
        start_pos = info['start_pos']
        end_pos = info['end_pos']
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                'k:', linewidth=1, alpha=0.6, 
                label='Reference Path' if i == 0 else "")
    
    # Load and plot GA planning result
    if ga_result_file and os.path.exists(ga_result_file):
        print(f"Loading GA planning result from: {ga_result_file}")
        try:
            with open(ga_result_file, 'r') as f:
                ga_data = json.load(f)
            
            phi_ga = np.array(ga_data['Initial_guess_phi'])
            plot_optimized_path(ax, reeb_graph, Waypoints, Flagb, phi_ga, 
                              color='red', linestyle='-', linewidth=2, 
                              label='GA Planning Result', alpha=0.8)
        except Exception as e:
            print(f"Error loading GA result: {e}")
    
    # Load and plot Planning_Path optimization result
    if planning_result_file and os.path.exists(planning_result_file):
        print(f"Loading Planning_Path result from: {planning_result_file}")
        try:
            with open(planning_result_file, 'r') as f:
                planning_data = json.load(f)
            
            # Check for different possible key names
            phi_key = None
            if 'Optimization_phi' in planning_data:
                phi_key = 'Optimization_phi'
            elif 'Initial_guess_phi' in planning_data:
                phi_key = 'Initial_guess_phi'
            
            if phi_key:
                phi_planning = np.array(planning_data[phi_key])
                plot_optimized_path(ax, reeb_graph, Waypoints, Flagb, phi_planning, 
                                  color='blue', linestyle='-', linewidth=2, 
                                  label='Planning_Path Result', alpha=0.8)
            else:
                print(f"Error: No recognized phi angles key found in planning result")
                print(f"Available keys: {list(planning_data.keys())}")
        except Exception as e:
            print(f"Error loading Planning_Path result: {e}")
    
    # Plot waypoints
    for i in range(len(Waypoints)):
        pos = reeb_graph.nodes[Waypoints[i]].configuration
        if i < len(Flagb) and Flagb[i] != 0:
            ax.plot(pos[0], pos[1], 'o', color='darkred', markersize=8, 
                   markeredgecolor='white', markeredgewidth=1,
                   label='Relay Point' if i == 0 or (i > 0 and Flagb[i-1] == 0) else "")
            ax.text(pos[0]+8, pos[1]+8, f'R{i}', fontsize=9, color='darkred', fontweight='bold')
        else:
            ax.plot(pos[0], pos[1], 'o', color='darkgreen', markersize=8, 
                   markeredgecolor='white', markeredgewidth=1,
                   label='Waypoint' if i == 0 else "")
            ax.text(pos[0]+8, pos[1]+8, f'W{i}', fontsize=9, color='darkgreen', fontweight='bold')
    
    ax.set_title('Comprehensive Path Planning Results with Safe Corridors', fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # Keep x and y with same scale
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive plot saved to: {save_path}")
    
    plt.show()
    
    return fig, ax

def plot_optimized_path(ax, reeb_graph, Waypoints, Flagb, phi_solution, color='blue', 
                       linestyle='-', linewidth=2, label='Optimized Path', alpha=0.8):
    """
    Plot the optimized path given phi angles
    """
    N = len(Waypoints)
    
    # Calculate distances and angles between waypoints
    Distance = np.zeros(N-1)
    Angle = np.zeros(N-1)
    
    for i in range(N-1):
        start_pos = reeb_graph.nodes[Waypoints[i]].configuration
        end_pos = reeb_graph.nodes[Waypoints[i+1]].configuration
        Distance[i] = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        Angle[i] = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
    
    # Plot the optimized path
    for i in range(N-1):
        start_pos = reeb_graph.nodes[Waypoints[i]].configuration
        
        phi_new_opt = phi_solution[i] + Flagb[i] * np.pi/2
        sigma_opt = -1 + np.cos(phi_new_opt - phi_solution[i+1])
        
        if abs(sigma_opt) < 1e-6:
            # Straight line case
            r0_opt = 100000000
            l_opt = Distance[i]
            
            # Plot straight line
            x_line = np.linspace(0, l_opt * np.cos(phi_solution[i+1]), 100)
            y_line = np.linspace(0, l_opt * np.sin(phi_solution[i+1]), 100)
            x_line = x_line + start_pos[0]
            y_line = y_line + start_pos[1]
            ax.plot(x_line, y_line, color=color, linestyle=linestyle, 
                   linewidth=linewidth, alpha=alpha, 
                   label=label if i == 0 else "")
        else:
            # Arc + line case
            l_opt = Distance[i] * (np.cos(Angle[i] - phi_new_opt) - np.cos(Angle[i] - phi_solution[i+1])) / sigma_opt
            r0_opt = -Distance[i] * np.sin(Angle[i] - phi_solution[i+1]) / sigma_opt
            
            # Plot the arc
            theta_start = phi_new_opt + np.pi/2
            theta_end = phi_solution[i+1] + np.pi/2
            center = (r0_opt * np.cos(theta_start), r0_opt * np.sin(theta_start))
            
            theta = np.linspace(theta_start, theta_end, 100)
            x_arc = r0_opt * np.cos(theta) - center[0] + start_pos[0]
            y_arc = r0_opt * np.sin(theta) - center[1] + start_pos[1]
            ax.plot(x_arc, y_arc, color=color, linestyle=linestyle, 
                   linewidth=linewidth, alpha=alpha,
                   label=label if i == 0 else "")
            
            # Plot the line segment
            x_line = np.linspace(r0_opt * np.cos(theta_end) - center[0],
                               r0_opt * np.cos(theta_end) - center[0] + l_opt * np.cos(phi_solution[i+1]), 100)
            y_line = np.linspace(r0_opt * np.sin(theta_end) - center[1],
                               r0_opt * np.sin(theta_end) - center[1] + l_opt * np.sin(phi_solution[i+1]), 100)
            x_line = x_line + start_pos[0]
            y_line = y_line + start_pos[1]
            ax.plot(x_line, y_line, color=color, linestyle=linestyle, 
                   linewidth=linewidth, alpha=alpha)

def save_corridor_data(safe_corridor, corridor_info, output_file):
    """
    Save corridor data to JSON file
    """
    data = {
        'safe_corridor': safe_corridor,
        'corridor_details': [
            {
                'segment_id': info['segment_id'],
                'start_pos': [float(x) for x in info['start_pos']],
                'end_pos': [float(x) for x in info['end_pos']],
                'distance': float(info['distance']),
                'angle': float(info['angle']),
                'slope': float(info['slope']) if info['slope'] != float('inf') else 'vertical',
                'y_min': float(info['y_min']),
                'y_max': float(info['y_max']),
                'corridor_bounds': [[float(x) for x in point] for point in info['corridor_bounds']],
                'waypoint_flag': int(info['waypoint_flag'])
            }
            for info in corridor_info
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Corridor data saved to: {output_file}")

def main():
    """
    Main function to generate and visualize safe corridors
    """
    print("Safe Corridor Generator and Visualizer")
    print("="*50)
    
    # Load configuration
    case = config.case
    N = config.N
    
    print(f"Configuration:")
    print(f"  Case: {case}")
    print(f"  Number of robots: {N}")
    
    # Generate file paths
    graph_file = config.get_full_path(config.file_path, use_data_path=True)
    waypoints_file = config.get_full_path(config.waypoints_file_path, use_data_path=True)
    environment_file = config.get_full_path(config.environment_file, use_data_path=True)
    
    print(f"\nFile paths:")
    print(f"  Graph: {graph_file}")
    print(f"  Waypoints: {waypoints_file}")
    print(f"  Environment: {environment_file}")
    
    # Check if files exist
    for file_path, name in [(graph_file, "Graph"), (waypoints_file, "Waypoints"), (environment_file, "Environment")]:
        if not os.path.exists(file_path):
            print(f"ERROR: {name} file not found: {file_path}")
            return
    
    try:
        # Load graph
        print("\nLoading reeb graph...")
        reeb_graph = load_reeb_graph_from_file(graph_file)
        print(f"Graph loaded with {len(reeb_graph.nodes)} nodes")
        
        # Load waypoints for plotting
        Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file)
        print(f"Loaded {len(Waypoints)} waypoints")
        
        # Generate safe corridors
        print("\nGenerating safe corridors...")
        safe_corridor, corridor_info, loaded_environment = generate_safe_corridor_detailed(
            reeb_graph, waypoints_file, environment_file, 
            step_size=0.5, max_distance=100
        )
        
        # Get result file paths
        ga_result_file = config.get_full_path(config.Result_file, use_data_path=True)
        planning_result_file = config.get_full_path(f"Optimization_withSC_path{N}{case}.json", use_data_path=True)
        
        print(f"\nLooking for result files:")
        print(f"  GA Result: {ga_result_file} {'✓' if os.path.exists(ga_result_file) else '✗'}")
        print(f"  Planning Result: {planning_result_file} {'✓' if os.path.exists(planning_result_file) else '✗'}")
        
        # Plot comprehensive results
        output_plot = config.get_full_path(f"comprehensive_results_{case}_N{N}.png", use_data_path=True)
        plot_comprehensive_results(reeb_graph, corridor_info, loaded_environment, Waypoints, Flagb, 
                                  ga_result_file=ga_result_file,
                                  planning_result_file=planning_result_file,
                                  save_path=output_plot)
        
        # Save corridor data
        output_data = config.get_full_path(f"safe_corridors_{case}_N{N}.json", use_data_path=True)
        save_corridor_data(safe_corridor, corridor_info, output_data)
        
        print(f"\nVisualization complete!")
        print(f"Output files:")
        print(f"  Comprehensive Plot: {output_plot}")
        print(f"  Corridor Data: {output_data}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
