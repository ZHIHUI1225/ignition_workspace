from Drawing import draw_problem_configuration
from Node import Node
import numpy as np


def construct_approximate_reeb_graph(environment, grid_x, grid_y):
    """Returns columns of points on a grid corresponding to the contracted level sets for a Reeb Graph.

    :param environment: Environment to construct Reeb Graph of
    :param grid_x: number of grid points in x direction
    :param grid_y: number of grid points in y direction
    :return: List[List[Node]]
    """
    # Use the environment's coordinate bounds directly
    min_x, max_x, min_y, max_y = environment.bounds()
    
    print(f"Reeb graph generation bounds: ({min_x}, {max_x}, {min_y}, {max_y})")
    
    # Apply small epsilon to avoid boundary issues, but respect coordinate bounds
    eps_x = min(0.01 * environment.width, 20)  # Max 20 pixels epsilon
    eps_y = min(0.01 * environment.height, 20)  # Max 20 pixels epsilon
    
    # Ensure we stay within the coordinate bounds
    min_x = max(min_x + eps_x, min_x + 10)  # At least 10 pixels from boundary
    min_y = max(min_y + eps_y, min_y + 10)  # At least 10 pixels from boundary  
    max_x = min(max_x - eps_x, max_x - 10)  # At least 10 pixels from boundary
    max_y = min(max_y - eps_y, max_y - 10)  # At least 10 pixels from boundary
    
    print(f"Adjusted bounds for node generation: ({min_x:.1f}, {max_x:.1f}, {min_y:.1f}, {max_y:.1f})")

    draw_problem_configuration(environment, None, None, None, draw_robot=False)
    step_y = (max_y - min_y) / grid_y

    columns = []
    for x in np.linspace(min_x, max_x, grid_x):
        current_column = []
   
        y = min_y
        while y < max_y:
            start_y = y
            while environment.clear_coords(x, y) and y < max_y:
                y += step_y
            end_y = y - step_y

            if environment.clear_coords(x, start_y) and environment.clear_coords(x, end_y):
                # Create node at the center of the free space segment
                node_y = (start_y + end_y) / 2
                
                # Validate that the node is within coordinate bounds
                env_min_x, env_max_x, env_min_y, env_max_y = environment.bounds()
                if (env_min_x <= x <= env_max_x) and (env_min_y <= node_y <= env_max_y):
                    current_column.append(Node(np.array([x, node_y])))
                else:
                    print(f"Warning: Node ({x:.1f}, {node_y:.1f}) outside coordinate bounds")

            while not environment.clear_coords(x, y) and y < max_y:
                y += step_y

        columns.append(current_column)

    # Log column generation statistics
    total_nodes = sum(len(col) for col in columns)
    print(f"Generated {len(columns)} columns with {total_nodes} total nodes")

    redundant_columns = set()
    for i in range(1,len(columns) - 1):
        if {n.configuration[1] for n in columns[i]} == {n.configuration[1] for n in columns[i + 1]} and {n.configuration[1] for n in columns[i]} == {n.configuration[1] for n in columns[i - 1]}:
            redundant_columns.add(i)
    columns = [columns[i] for i in range(len(columns)) if i not in redundant_columns and len(columns[i]) > 0]
    # expend this node to the more free space
    bias=0
    for i in range(1,len(columns) - 1):
        # if the nodes reduce, the space become bigger. back up the nodes
        if len(columns[i]) > len(columns[i+1]):
            for n in columns[i]:
                n.configuration[0] = n.configuration[0]+bias
            for n in columns[i+1]:
                n.configuration[0] = n.configuration[0]+bias
        elif len(columns[i]) < len(columns[i+1]):
            for n in columns[i+1]:
                n.configuration[0] = n.configuration[0]-bias
            for n in columns[i]:
                n.configuration[0] = n.configuration[0]-bias
        elif len(columns[i]) ==1 and len(columns[i+1])==1:
                if abs(columns[i][0].configuration[1]-(max_y+min_y)/2)<10:
                    columns[i][0].configuration[0]=columns[i][0].configuration[0]-bias
                    columns[i+1][0].configuration[0]=columns[i+1][0].configuration[0]-bias
                if abs(columns[i+1][0].configuration[1]-(max_y+min_y)/2)<10:
                    columns[i][0].configuration[0]=columns[i][0].configuration[0]+bias
                    columns[i+1][0].configuration[0]=columns[i+1][0].configuration[0]+bias
    return columns
