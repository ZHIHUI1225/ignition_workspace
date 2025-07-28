
from Polygon import Polygon, rectangle
from Environment import Environment
import numpy as np
from ReebGraph import construct_approximate_reeb_graph
from Graph import Graph
import copy
def generate_barriers_simple():
    """Generates the 'Simple' problem configuration from OMPL.

    Returns (environment, robot, start configuration, goal configuration)."""
    border = [rectangle(28, 600, 52, 18), rectangle(28, 62, 650, 18),
              rectangle(600, 650, 650, 18), rectangle(28, 600, 650, 600)]
    column1 = [ rectangle(186, 226, 560, 300), rectangle(186, 226, 200, 18)]
    column2 = [ rectangle(380, 480, 500, 450), rectangle(380, 480, 260, 220)]

    barriers_environment = Environment(border + column1 + column2 )
    robot_geometry = Polygon([(-33, 63), (-33, -63), (33, -63), (33, -42), (-11, -42),
                              (-11, 42), (33, 42), (33, 63), (-33, 63)])

    start = np.array([100, 70, 0])
    goal = np.array([550, 535, np.pi])

    return barriers_environment, start, goal

def generate_barriers_problem():
    """Generates the 'Barriers' problem configuration from OMPL.

    Returns (environment, robot, start configuration, goal configuration)."""
    border = [rectangle(28, 1298, 52, 18), rectangle(28, 62, 865, 18),
              rectangle(1264, 1298, 865, 18), rectangle(28, 1298, 865, 832)]
    column1 = [rectangle(186, 226, 832, 650), rectangle(186, 226, 560, 300), rectangle(186, 226, 200, 18)]
    column2 = [rectangle(300, 420, 620, 590), rectangle(300, 420, 530, 490), rectangle(300, 420, 429, 390), rectangle(300, 420, 260, 220)]
    column3 = [rectangle(500, 540, 755, 650), rectangle(500, 540, 560, 447), rectangle(500, 540, 370, 200)]
    column4 = [rectangle(541, 720, 720, 690), rectangle(541, 720, 520, 490), rectangle(541, 720, 320, 250)]
    # column4 = [rectangle(682, 720, 624, 510), rectangle(682, 720, 234, 122)]
    column5 = [rectangle(800, 930, 832, 750), rectangle(800, 930, 650, 550), rectangle(800, 930, 450, 320), rectangle(800, 930, 250, 18)]
    column6 = [ rectangle(1000, 1100, 650, 600), rectangle(1000, 1100, 500, 400), rectangle(1000, 1100, 300, 250)]

    barriers_environment = Environment(border + column1 + column2 + column3 + column4 + column5 + column6)
    robot_geometry = Polygon([(-33, 63), (-33, -63), (33, -63), (33, -42), (-11, -42),
                              (-11, 42), (33, 42), (33, 63), (-33, 63)])
    robot = Robot(robot_geometry)
    start = np.array([115, 146, 0])
    goal = np.array([1180, 735, np.pi])

    return barriers_environment, robot, start, goal
#### generate the graph with bottle neck
def generate_bottle_neck_barriers_problem():
    windth=1500
    height=1500
    # rectangle(left, right, top, bottom)
    border = [rectangle(28, 62+windth, 52, 18), rectangle(28, 62, 52+height, 18),
              rectangle(28+windth, 62+windth, 52+height, 18), rectangle(28,62+windth, 52+height, 18+height)]
    x=28+100
    gap_min=90
    gap_max=150
    environment=border
    while x< 400:
        y=np.random.randint(50, 100)
        w_o=np.random.randint(gap_min, gap_max+20)
        h_o=np.random.randint(gap_min, gap_max)
        while y< height-100:
            environment.append(rectangle(x, x+w_o, y, y+h_o))
            y+=h_o+np.random.randint(gap_min, gap_max)
        x=x+w_o+np.random.randint(gap_min, gap_max)
    environment.append(rectangle(600, 680, 850, 18))
    environment.append(rectangle(600, 680, 52+height, 52+890))
    x=780
    y=np.random.randint(50, 100)
    w_o=120
    h_o=np.random.randint(gap_min, gap_max)
    while y< height-100:
        environment.append(rectangle(x, x+w_o, y, y+h_o))
        y+=h_o+np.random.randint(gap_min, gap_max)
    environment.append(rectangle(1050, 1150, 400, 18))
    environment.append(rectangle(1050, 1150, 52+height, 52+420))
    x=1300
    while x< windth-150:
        y=np.random.randint(50, 100)
        w_o=100
        h_o=np.random.randint(gap_min, gap_max)
        while y< height-100:
            environment.append(rectangle(x, x+w_o, y, y+h_o))
            y+=h_o+np.random.randint(gap_min, gap_max)
        x=x+w_o+np.random.randint(gap_min, gap_max)
    barriers_environment = Environment(environment)
    robot_geometry = Polygon([(-33, 63), (-33, -63), (33, -63),(33,63), (-33, 63)])
    robot = Robot(robot_geometry)
    start = np.array([100, 1400, 0])
    goal = np.array([1500, 200, np.pi])

    return barriers_environment, robot, start, goal

    
    


# graph with T shape obstacles
def generate_barriers_test():
    """Generates the 'Barriers' problem configuration from OMPL.

    Returns (environment, robot, start configuration, goal configuration)."""
    border = [rectangle(28, 1298, 52, 18), rectangle(28, 62, 865, 18),
              rectangle(1264, 1298, 865, 18), rectangle(28, 1298, 865, 832)]
    column1 = [rectangle(166, 206, 832, 650), rectangle(166, 206, 560, 300), rectangle(166, 206, 200, 18)]
    column2 = [rectangle(270, 380, 720, 690), rectangle(300, 420, 580, 530), rectangle(270, 380, 429, 390), rectangle(300, 420, 260, 220)]
    column3 = [rectangle(500, 540, 755, 650), rectangle(500, 540, 560, 447), rectangle(500, 540, 370, 200)]
    column4 = [rectangle(541, 620, 720, 690), rectangle(541, 690, 520, 490), rectangle(541, 670, 320, 250)]
    # column4 = [rectangle(682, 720, 624, 510), rectangle(682, 720, 234, 122)]
    column5 = [rectangle(750, 930, 832, 750), rectangle(800, 930, 650, 550), rectangle(800, 930, 450, 320), rectangle(750, 930, 250, 18)]
    column6 = [ rectangle(1000, 1200, 650, 600), rectangle(1050, 1150, 500, 400), rectangle(1000, 1200, 300, 250)]

    barriers_environment = Environment(border + column1 + column2 + column3 + column4 + column5 + column6)
    robot_geometry = Polygon([(-33, 63), (-33, -63), (33, -63), (33, -42), (-11, -42),
                              (-11, 42), (33, 42), (33, 63), (-33, 63)])
    robot = Robot(robot_geometry)
    start = np.array([115, 200, 0])
    goal = np.array([1180, 750, np.pi])

    return barriers_environment, robot, start, goal

def generate_large_scale_test():
    windth=1500
    height=1500

    border = [rectangle(28, 62+windth, 52, 18), rectangle(28, 62, 52+height, 18),
              rectangle(28+windth, 62+windth, 52+height, 18), rectangle(28,62+windth, 52+height, 18+height)]
    x=200
    gap_min=80
    gap_max=120
    environment=border
    while x<windth-150:
        y=np.random.randint(50, 100)
        w_o=np.random.randint(gap_min, gap_max)
        h_o=np.random.randint(gap_min, gap_max)
        while y< height-100:
            environment.append(rectangle(x, x+w_o, y, y+h_o))
            y+=h_o+np.random.randint(gap_min-20, gap_max-20)
        x=x+w_o+np.random.randint(gap_min, gap_max-5)
    barriers_environment = Environment(environment)
    robot_geometry = Polygon([(-33, 63), (-33, -63), (33, -63),(33,63), (-33, 63)])
    robot = Robot(robot_geometry)
    start = np.array([100, 1400, 0])
    goal = np.array([1500, 200, np.pi])

    return barriers_environment, robot, start, goal


# def expend_polygon(environment, distance):
#     new_polygons = []
#     for p in environment.polygons:
#         new_polygon = []
#         for vertex in p.vertices:
#             new_vertex = vertex.copy()
#             new_vertex[0] += distance
#             new_polygon.append(new_vertex)
#         new_polygons.append(Polygon(new_polygon))
#     return Environment(new_polygons)
from Node import Node
from shapely.geometry import LineString, Polygon, Point

def check_collision(edge, polygons):
    """
    Check if an edge collides with any polygon in the environment.
    
    Parameters:
    edge (tuple): A tuple of two nodes representing the edge.
    polygons (list): A list of polygons in the environment.
    
    Returns:
    bool: True if the edge collides with any polygon, False otherwise.
    """
    node1, node2 = edge
    node1_coords = (node1.configuration[0], node1.configuration[1])
    node2_coords = (node2.configuration[0], node2.configuration[1])
    
    # Skip if the two nodes are the same (self-loop)
    if node1_coords == node2_coords:
        return False
    
    # Create a line between the two nodes
    edge_line = LineString([node1_coords, node2_coords])
    
    # Use a small buffer for the edge line to account for numerical precision
    # This is important for cases where the line passes very close to a polygon edge
    edge_line_buffered = edge_line.buffer(0.1)
    
    # Check collision against each polygon
    for polygon in polygons:
        try:
            # Convert polygon vertices to proper format for shapely
            polygon_vertices = []
            for i in range(len(polygon.vertices)):
                vertex = polygon.vertices[i]
                if hasattr(vertex, '__len__') and len(vertex) >= 2:
                    polygon_vertices.append((float(vertex[0]), float(vertex[1])))
                else:
                    print(f"Warning: Invalid vertex format: {vertex}")
                    continue
            
            # Need at least 3 vertices to form a polygon
            if len(polygon_vertices) < 3:
                continue
                
            # Ensure polygon is closed (first and last vertices are the same)
            if polygon_vertices[0] != polygon_vertices[-1]:
                polygon_vertices.append(polygon_vertices[0])
            
            # Create shapely polygon from vertices
            try:
                polygon_shape = Polygon(polygon_vertices)
                
                # Fix any self-intersections or invalid geometries
                if not polygon_shape.is_valid:
                    polygon_shape = polygon_shape.buffer(0)
                    if not polygon_shape.is_valid:
                        continue
                
                # Check for intersection between edge line and polygon
                # We check both the original line and the buffered line
                if edge_line.intersects(polygon_shape) or edge_line_buffered.intersects(polygon_shape):
                    # Check if only the endpoints are inside the polygon (this is okay)
                    point1 = Point(node1_coords)
                    point2 = Point(node2_coords)
                    
                    if (point1.within(polygon_shape) and not point2.within(polygon_shape)) or \
                       (point2.within(polygon_shape) and not point1.within(polygon_shape)):
                        # One point is inside, one is outside, this is likely a collision
                        return True
                    elif point1.within(polygon_shape) and point2.within(polygon_shape):
                        # Both points are inside the same polygon - this is ok
                        continue
                    else:
                        # Line intersects the polygon but neither endpoint is inside
                        # This is definitely a collision
                        return True
            except Exception as e:
                print(f"Error creating polygon: {e}")
                continue
                
        except Exception as e:
            print(f"Warning: Error checking collision for polygon {polygon}: {e}")
            continue
            
    return False

def expend_polygon(environment, distance):
    expanded_polygons = copy.deepcopy(environment.polygons)
    new_polygons = []
    for p in expanded_polygons:
        vertex= p.vertices.copy()
        x_min=vertex[0][0]
        x_max=vertex[1][0]
        y_min=min(vertex[2][1],vertex[0][1])
        y_max=max(vertex[0][1],vertex[2][1])
        new_polygons.append(rectangle(x_min-distance, x_max+distance, y_max+distance, y_min-distance))
    expend_Environment = Environment(new_polygons)
    return expend_Environment

def expend_polygon_selective(environment, distance):
    """
    Expand polygons selectively, avoiding expansion of Wall_0 and Wall_2
    if the environment has no_expand_indices metadata.
    """
    expanded_polygons = copy.deepcopy(environment.polygons)
    new_polygons = []
    
    # Check if environment has selective expansion metadata
    no_expand_indices = getattr(environment, 'no_expand_indices', [])
    
    for i, p in enumerate(expanded_polygons):
        vertex = p.vertices.copy()
        x_min = vertex[0][0]
        x_max = vertex[1][0]
        y_min = min(vertex[2][1], vertex[0][1])
        y_max = max(vertex[0][1], vertex[2][1])
        
        # Don't expand Wall_0 and Wall_2 (if specified in no_expand_indices)
        if i in no_expand_indices:
            # Keep original size for walls that shouldn't be expanded
            new_polygons.append(rectangle(x_min, x_max, y_max, y_min))
            print(f"Polygon {i} (Wall) not expanded")
        else:
            # Expand obstacles and other walls normally
            new_polygons.append(rectangle(x_min-distance, x_max+distance, y_max+distance, y_min-distance))
            
    expend_Environment = Environment(new_polygons)
    return expend_Environment

def generate_barriers_reeb_graph(environment_original):
    """
    Generate a Reeb graph for the given environment.
    
    Parameters:
    environment_original: The original environment with correct coordinate bounds
    
    Returns:
    Graph: The generated Reeb graph with nodes within the coordinate bounds
    """
    # Use the original environment's coordinate bounds for grid generation
    coord_bounds = environment_original.coord_bounds
    print(f"Using coordinate bounds for graph generation: {coord_bounds}")
    
    # Calculate grid parameters based on coordinate bounds
    x_min, x_max, y_min, y_max = coord_bounds
    environment_width = x_max - x_min
    environment_height = y_max - y_min
    
    # Adaptive grid size based on environment dimensions
    # For larger environments, use more grid points for better resolution
    if environment_width > 800:
        grid_x = max(15, int(environment_width / 60))  # ~60 pixels per grid point
    else:
        grid_x = max(10, int(environment_width / 80))  # ~80 pixels per grid point
        
    if environment_height > 400:
        grid_y = max(10, int(environment_height / 40))  # ~40 pixels per grid point  
    else:
        grid_y = max(8, int(environment_height / 50))   # ~50 pixels per grid point
    
    print(f"Environment dimensions: {environment_width}x{environment_height}")
    print(f"Using grid size: {grid_x}x{grid_y}")
    
    # Expand polygons for robot clearance, but preserve coordinate bounds
    environment = expend_polygon_selective(environment_original, 50)
    
    # Override the expanded environment's bounds with the original correct bounds
    environment.coord_bounds = coord_bounds
    environment.width = environment_width  
    environment.height = environment_height
    
    print(f"Preserved coordinate bounds after expansion: {environment.coord_bounds}")
    
    # Generate columns using the construct_approximate_reeb_graph function
    columns = construct_approximate_reeb_graph(environment, grid_x, grid_y)

    # Manual correction for better visualization
    # new_columns = []
    # for c in columns:
    #     if len(c) == 1:
    #         new_columns.append(c)
    #     else:
    #         c_copy = [n.copy() for n in c]
    #         for n in c_copy:
    #             n.configuration[0] -= 20
    #         for n in c:
    #             n.configuration[0] += 20
    #         new_columns.append(c_copy)
    #         new_columns.append(c)
    # columns = new_columns
    # # leave the end nodes of the obetacles and for the blank space take the middle point
    # new_columns = []
    # for i in range(1,len(columns)):
    #     if len(columns[i - 1]) == len(columns[i]):          
    #         if len(columns[i]) == 1:
    #             c_copy =columns[i]
    #             c_copy[0].configuration[0] = (columns[i][0].configuration[0] + columns[i-1][0].configuration[0])/2
    #             new_columns.append(c_copy)
    #         else:
    #             for j in range(len(columns[i])):
    #                 if columns[i][j].configuration[1] != columns[i-1][j].configuration[1]:
    #                     c_copy =columns[i]
    #                     new_columns.append(c_copy)
    #         # else:
    #         #     for j in range(len(columns[i])):
    #         #         if columns[i][j].configuration[1] == columns[i-1][j].configuration[1]:
    #         #             c_copy[j].configuration[0] = (columns[i][j].configuration[0] + columns[i-1][j].configuration[0])/2
    #         #     c_copy[0].configuration[0] = (columns[i][0].configuration[0] +columns[i-1][0].configuration[0] )/2
    #     else:
    #         # c_copy = columns[i]
    #         new_columns.append(columns[i-1])
    #         if len(columns[i]) != 1:
    #             new_columns.append(columns[i])
    # columns = new_columns
    # for i in range(0, len(columns), 3):
    #     for n in columns[i]:
    #         n.configuration[0] += 50
    # columns[-1][0].configuration[0] += 50
    # End of manual correction

    reeb_graph = Graph([node for column in columns for node in column])

    # Connect nodes in graph with comprehensive collision checking
    total_edges_attempted = 0
    total_edges_added = 0
    collision_edges_removed = 0
    
    for i in range(1, len(columns)):
        for parent in columns[i - 1]:
            if len(columns[i - 1]) == 1 or len(columns[i]) == 1:
                # Fully connect chokepoints, but check for collisions
                for child in columns[i]:
                    total_edges_attempted += 1
                    edge = [parent, child]
                    if not check_collision(edge, environment_original.polygons):
                        reeb_graph.add_edge(parent, child)
                        total_edges_added += 1
                    else:
                        collision_edges_removed += 1
                        print(f"Collision detected: Removed edge between chokepoint nodes at ({parent.configuration[0]:.1f}, {parent.configuration[1]:.1f}) and ({child.configuration[0]:.1f}, {child.configuration[1]:.1f})")
            else:
                # Connect redundant columns directly with collision checking
                for j in range(len(columns[i-1])):
                    edge_flag=False
                    for k in range(len(columns[i])):
                        total_edges_attempted += 1
                        new_edge = [columns[i - 1][j], columns[i][k]]
                        if not check_collision(new_edge, environment_original.polygons):
                            reeb_graph.add_edge(columns[i - 1][j], columns[i][k])
                            total_edges_added += 1
                            edge_flag=True
                        else:
                            collision_edges_removed += 1
                    
                    # If no direct connection possible, try alternative connections
                    if edge_flag==False:
                        dis_min=1000
                        k_min=-1
                        # Check if we have enough columns before accessing i+1
                        if i + 1 < len(columns) and len(columns[i+1]) > 0:
                            for k in range(len(columns[i+1])):
                                if k < len(columns[i+1]) and j < len(columns[i-1]):  # Additional bounds check
                                    new_edge = [columns[i - 1][j], columns[i+1][k]]
                                    distance=np.linalg.norm(np.array(columns[i - 1][j].configuration)-np.array(columns[i+1][k].configuration))
                                    if not check_collision(new_edge, environment_original.polygons) and distance<dis_min:
                                        dis_min=distance
                                        k_min=k
                        if k_min!=-1:
                            reeb_graph.add_edge(columns[i - 1][j], columns[i+1][k_min])
                            total_edges_added += 1
                            print(f"Alternative connection: Connected node at column {i-1} to column {i+1} (skipping column {i})")
                        else:
                            k_min=-1
                            dis_min=1000
                            # Check if we have enough columns before accessing i-2
                            if i >= 2 and len(columns[i-2]) > 0:
                                for k in range(len(columns[i-2])):
                                    if k < len(columns[i-2]) and j < len(columns[i]):  # Additional bounds check
                                        new_edge = [columns[i - 2][k], columns[i][j]]
                                        distance=np.linalg.norm(np.array(columns[i - 2][k].configuration)-np.array(columns[i][j].configuration))
                                        if not check_collision(new_edge, environment_original.polygons) and distance<dis_min:
                                            k_min=k
                                            dis_min=distance
                                if k_min!=-1:
                                    reeb_graph.add_edge(columns[i - 2][k_min], columns[i][j])
                                    total_edges_added += 1
                                    print(f"Backward connection: Connected node at column {i-2} to column {i}")
                                else:
                                    print(f"Warning: No collision-free connection found for node at column {i-1}, position ({columns[i-1][j].configuration[0]:.1f}, {columns[i-1][j].configuration[1]:.1f})")
                            else:
                                print(f"Cannot access column i-2 when i={i}, skipping edge creation")

    print(f"\nEdge creation summary:")
    print(f"Total edges attempted: {total_edges_attempted}")
    print(f"Total edges added: {total_edges_added}")
    print(f"Edges removed due to collisions: {collision_edges_removed}")
    print(f"Success rate: {(total_edges_added/total_edges_attempted)*100:.1f}%" if total_edges_attempted > 0 else "No edges attempted")
    
    # Final validation: Use comprehensive validation function
    print("\nPerforming final graph validation...")
    additional_removed = validate_graph_edges(reeb_graph, environment_original.polygons)
    
    if additional_removed > 0:
        print(f"COLLISION WARNING: Final validation removed {additional_removed} additional collision edges")
        print(f"This may indicate that the collision detection algorithm missed some collisions during initial graph creation.")
    else:
        print("Final validation: No additional collision edges found - graph appears to be collision-free")

    return reeb_graph

def validate_graph_edges(reeb_graph, environment_polygons):
    """
    Validate all edges in the graph and remove any that collide with obstacles.
    
    Parameters:
    reeb_graph (Graph): The reeb graph to validate
    environment_polygons (list): List of polygons representing obstacles
    
    Returns:
    int: Number of edges removed due to collisions
    """
    edges_to_remove = []
    total_edges = 0
    
    # Check all edges in the graph by iterating through out_neighbors
    for node, neighbors in reeb_graph.out_neighbors.items():
        for neighbor in neighbors:
            total_edges += 1
            edge = [node, neighbor]
            if check_collision(edge, environment_polygons):
                edges_to_remove.append((node, neighbor))
                print(f"Collision detected: Edge between ({node.configuration[0]:.1f}, {node.configuration[1]:.1f}) and ({neighbor.configuration[0]:.1f}, {neighbor.configuration[1]:.1f})")
    
    # Remove collision edges
    for node1, node2 in edges_to_remove:
        reeb_graph.remove_edge(node1, node2)
    
    print(f"Graph validation complete: {len(edges_to_remove)} collision edges removed out of {total_edges} total edges")
    return len(edges_to_remove)
