
# generate the graph and save the environment for experiments

from BarriersOriginal import generate_barriers_reeb_graph
import matplotlib.pyplot as plt
from Graph import save_reeb_graph_to_file
from Environment import Environment
import json
import os
import sys

# Add config path to sys.path
sys.path.append('/root/workspace/config')
from config_loader import config

# Load configuration parameters
case = config.case
N = config.N
start = config.start_pose
goal = config.goal_pose
arc_range = config.arc_range

print(f"Configuration loaded:")
print(f"  Case: {case}")
print(f"  Number of robots (N): {N}")
print(f"  Arc range: {arc_range}")
print(f"  Start pose: {start}")
print(f"  Goal pose: {goal}")

# Load environment from Enviro_experiments.json
env_file_path = "/root/workspace/data/Enviro_experiments.json"
print(f"Loading environment from {env_file_path}")

if not os.path.exists(env_file_path):
    print(f"Error: Environment file {env_file_path} not found!")
    exit(1)

environment = Environment.load_from_file(env_file_path)
print("Environment loaded successfully")

# IMPORTANT: Preserve the original coordinate bounds from Enviro_experiments.json
# The Environment class recalculates bounds based only on polygon vertices, which is incorrect
# We need to use the full workspace bounds from the rectangle detector
with open(env_file_path, 'r') as f:
    original_env_data = json.load(f)
    
# Override the incorrectly calculated bounds with the original correct bounds
if 'coord_bounds' in original_env_data:
    environment.coord_bounds = tuple(original_env_data['coord_bounds'])
    environment.width = original_env_data.get('width', environment.width)
    environment.height = original_env_data.get('height', environment.height)
    print(f"Preserved original coordinate bounds: {environment.coord_bounds}")
    print(f"Environment dimensions: {environment.width}x{environment.height}")
else:
    print("Warning: No coord_bounds found in original file, using calculated bounds")

print(f"Start: {start}, Goal: {goal}")
print(f"Environment bounds: {environment.coord_bounds}")

# Save the environment to a JSON file with start and goal information
env_data = environment.to_dict()
env_data['start_pose'] = start
env_data['goal_pose'] = goal

output_env_file = f"/root/workspace/data/environment_{case}.json"
with open(output_env_file, 'w') as file:
    json.dump(env_data, file, indent=4)

print(f"Environment saved to {output_env_file}")
print(f"Start pose saved: {start}")
print(f"Goal pose saved: {goal}")

# Generate the Reeb graph (or other graph representation)
# Assuming Environment_expansion is just the environment itself for this case
Environment_expansion = environment
print("Generating Reeb graph...")
reeb_graph = generate_barriers_reeb_graph(Environment_expansion)
print("Reeb graph generated.")

# Clear any repeat neighbors in the graph
reeb_graph.clear_repeat_neighbors()
print("Cleared repeat neighbors.")

# Option to improve the graph quality using rebuild_graph from Rebuild_large.py
use_rebuild = False  # Set to True if you want to use the rebuild_graph functionality
if use_rebuild:
    try:
        from Rebuild_large import rebuild_graph
        print("Improving graph quality with rebuild_graph...")
        reeb_graph = rebuild_graph(reeb_graph, environment, start, goal)
        print("Graph improvement completed.")
    except ImportError:
        print("Warning: Rebuild_large module not found, skipping graph improvement step.")

# Save the Reeb graph to the specified JSON file with start and goal information
output_graph_file = f"/root/workspace/data/Graph_{case}.json"

# Create the graph data manually to include start and goal
from Graph import Node

node_id_map = {node: idx for idx, node in enumerate(reeb_graph.nodes)}
graph_data = {
    'nodes': [(node_id_map[node], node.configuration.tolist(), node.parent, node.is_goal) for node in reeb_graph.nodes],
    'in_neighbors': {node_id_map[node]: [node_id_map[neighbor] for neighbor in neighbors] for node, neighbors in reeb_graph.in_neighbors.items()},
    'out_neighbors': {node_id_map[node]: [node_id_map[neighbor] for neighbor in neighbors] for node, neighbors in reeb_graph.out_neighbors.items()},
    'start_pose': start,
    'goal_pose': goal
}

with open(output_graph_file, 'w') as file:
    json.dump(graph_data, file, indent=4)

print(f"Reeb graph saved to {output_graph_file}")
print(f"Start pose saved in graph: {start}")
print(f"Goal pose saved in graph: {goal}")

# Draw the graph (optional, for visualization)
print("Drawing graph...")

# Draw boundary walls first
coord_bounds = environment.coord_bounds
x_min, x_max, y_min, y_max = coord_bounds

# Draw boundary walls as thick black lines
boundary_width = 3
plt.plot([x_min, x_max], [y_min, y_min], 'k-', linewidth=boundary_width, label='Boundary Wall')  # Bottom wall
plt.plot([x_min, x_max], [y_max, y_max], 'k-', linewidth=boundary_width)  # Top wall
plt.plot([x_min, x_min], [y_min, y_max], 'k-', linewidth=boundary_width)  # Left wall
plt.plot([x_max, x_max], [y_min, y_max], 'k-', linewidth=boundary_width)  # Right wall

# Draw environment polygons/obstacles
environment.draw('lightgray') # Draw environment for context

# Draw the graph
reeb_graph.draw("purple") # Changed color for distinction

# Draw start and goal positions
plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

plt.title(f"Reeb Graph for {case.capitalize()} Environment with Boundary Walls")
plt.xlabel("X coordinate (pixels)")
plt.ylabel("Y coordinate (pixels)")
plt.legend()
plt.axis('equal')
plt.autoscale()
print("Displaying plot...")
plt.show()
print("Script finished.")
