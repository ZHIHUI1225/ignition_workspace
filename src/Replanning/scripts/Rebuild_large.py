# rebuild the graph:
# 1. revise the node of which connected line ha collision with walls
# 2. thicken the nodes by middle point of the connected line
# 3. spare it and reconnect it with neighbors
 
import sys
import os
import numpy as np
import json
from Graph import save_reeb_graph_to_file

# from ProblemConfigurations.Barriers import generate_barriers_problem, generate_barriers_reeb_graph
from BarriersOriginal import expend_polygon,check_collision, generate_large_scale_test, generate_barriers_problem, generate_barriers_reeb_graph,generate_barriers_test
import matplotlib.pyplot as plt
from DRRRT import augment_reeb_graph

# # Get the current script's directory
# current_directory = os.path.dirname(os.path.abspath(__file__))
# # Go up one directory level to the project root
# project_root = os.path.dirname(os.path.dirname(current_directory))

# # Add the relative path to sys.path
# sys.path.append(project_root )
from Node import Node
from shapely.geometry import LineString, Polygon

def calculate_edge_length(node1, node2):
    """
    Calculate the Euclidean distance between two nodes.
    
    Parameters:
    node1 (Node): The first node.
    node2 (Node): The second node.
    
    Returns:
    float: The Euclidean distance between the two nodes.
    """
    return np.linalg.norm(node1.configuration - node2.configuration)

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
    edge_line_buffered = edge_line.buffer(0.1)
    
    for polygon in polygons:
        try:
            # Convert polygon vertices to proper format
            polygon_array = [polygon.vertices[i] for i in range(len(polygon.vertices))]
            
            # Need at least 3 vertices to form a polygon
            if len(polygon_array) < 3:
                continue
            
            # Ensure polygon is closed (first and last vertices are the same)
            if polygon_array[0] != polygon_array[-1]:
                polygon_array.append(polygon_array[0])
            
            polygon_shape = Polygon(polygon_array)
            
            # Check if polygon is valid
            if not polygon_shape.is_valid:
                continue
                
            # Check for intersection with both the line and buffered line
            if edge_line.intersects(polygon_shape) or edge_line_buffered.intersects(polygon_shape):
                return True
                
        except Exception as e:
            print(f"Warning: Error checking collision for polygon: {e}")
            continue
            
    return False

def get_out_neighbors_of_out_neighbors(reeb_graph, node):
    """
    Get the out_neighbors of the out_neighbors of a given node.
    
    Parameters:
    reeb_graph: The graph object.
    node: The node whose out_neighbors' out_neighbors are to be retrieved.
    
    Returns:
    dict: A dictionary where keys are the out_neighbors of the given node and values are their out_neighbors.
    """
    result = {}
    if node in reeb_graph.out_neighbors:
        for out_neighbor in reeb_graph.out_neighbors[node]:
            if out_neighbor in reeb_graph.out_neighbors:
                result[out_neighbor] = reeb_graph.out_neighbors[out_neighbor]
            else:
                result[out_neighbor] = []
    return result

def check_repeat_node(reeb_graph):
    count=0
    node_previous=reeb_graph.nodes[0]
    for i in range(1,len(reeb_graph.nodes)):
        if calculate_edge_length(reeb_graph.nodes[i],node_previous)<10:
            count+=1
        node_previous=reeb_graph.nodes[i]
    return count


    # ################################################################
def find_near_edge(reeb_graph, environment,node,node_previous,dist):
    for neighbor in reeb_graph.out_neighbors[node]:
        new_edge=(node_previous,neighbor)
        if not check_collision(new_edge, environment.polygons) and calculate_edge_length(node_previous,neighbor)<dist:
            if neighbor not in reeb_graph.out_neighbors[node_previous]:
                reeb_graph.add_edge(node_previous,neighbor)
                reeb_graph.remove_edge(node_previous,node)
                print("reconnected the node")

# # environment, robot, start, goal =  generate_barriers_problem()
# environment, robot, start, goal =  generate_large_scale_test()
# reeb_graph = generate_barriers_reeb_graph(environment)

# # reeb_graph, start_node, goal_node = augment_reeb_graph(environment, start, goal,
# #                                                        link_step_size=0.001 * environment.width,
# #                                                        reeb_graph=reeb_graph,
# #                                                        connection_distance_threshold=0.25 * environment.width)
# reeb_graph.clear_repeat_neighbors()
# draw_problem_configuration(environment, robot, start, goal, draw_robot=False)
# reeb_graph.draw("red")
# plt.show()
# Enviroment_expend=expend_polygon(environment, 10)

def rebuild_graph(reeb_graph, environment_original,start, target):
###########################################################
    reeb_graph.clear_repeat_neighbors()
    environment = expend_polygon(environment_original, 10)
# conbine the near nodes: connect the nodes of which the distance is less than 10 and try to choose the middle point of them
# ## check the collision of nodes and edges
#     i=0
#     while i<(len(reeb_graph.nodes)-1):
#         #check the collision of the node with the environment.polygons
#         for out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
#             if check_collision((reeb_graph.nodes[i],out_neighbor), environment.polygons):
#                 new_node=reeb_graph.nodes[i]
#                 while check_collision((new_node,out_neighbor), environment.polygons):
#                     new_node.configuration[0]=new_node.configuration[0]-30
#                 collision_flag=False
#                 for in_neighbor in reeb_graph.in_neighbors[reeb_graph.nodes[i]]:
#                     if check_collision((in_neighbor,new_node), environment.polygons):
#                         collision_flag=True
#                         break
#                 for other_out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
#                     if other_out_neighbor is not out_neighbor:
#                         if check_collision((new_node,other_out_neighbor), environment.polygons):
#                             collision_flag=True
#                             break
#                 if not collision_flag:
#                     reeb_graph.nodes[i].configuration=new_node.configuration
#                     print("revised the node")
#         i=i+1
# check the collision of new edges and add the new edges to the reeb_graph
###############################################################
    combine_radius=70
    i=0
    while i< len(reeb_graph.nodes):
        for out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
            if calculate_edge_length(reeb_graph.nodes[i],out_neighbor)<combine_radius:
                node_new=Node((reeb_graph.nodes[i].configuration+out_neighbor.configuration)/2)
                collision_flag=False
                for in_neighbor in reeb_graph.in_neighbors[reeb_graph.nodes[i]]:
                    if check_collision((in_neighbor,node_new), environment.polygons):
                        collision_flag=True
                        break
                for other_out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
                    if other_out_neighbor is not out_neighbor:
                        if check_collision((node_new,other_out_neighbor), environment.polygons):
                            collision_flag=True
                            break
                if collision_flag:
                    break  
                else:
                    if out_neighbor in reeb_graph.out_neighbors:
                        for out_out_neighbor in reeb_graph.out_neighbors[out_neighbor]:
                            if check_collision((node_new,out_out_neighbor), environment.polygons):
                                collision_flag=True
                                break  
                    if out_neighbor in reeb_graph.in_neighbors:
                        for in_out_neighbor in reeb_graph.in_neighbors[out_neighbor]:
                            if check_collision((in_out_neighbor,node_new), environment.polygons):
                                collision_flag=True
                                break
                    if not collision_flag:
                        reeb_graph.combine_nodes(reeb_graph.nodes[i],out_neighbor,node_new)
                        i=i-1
                        print("combined the nodes")       
        i=i+1

    # ######check the near nodes that can be deleted

    reeb_graph.clear_repeat_neighbors()
    i=0
    #delete the out_neighbors of the node
    while i< len(reeb_graph.nodes):
        for out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
            if calculate_edge_length(reeb_graph.nodes[i],out_neighbor)<combine_radius:
                collision_flag=False
                for out_out_neighbor in reeb_graph.out_neighbors[out_neighbor]:
                    if check_collision((reeb_graph.nodes[i],out_out_neighbor), environment.polygons):
                        collision_flag=True
                        break
                    if not collision_flag:
                        for in_out_neighbor in reeb_graph.in_neighbors[out_neighbor]:
                            if check_collision((in_out_neighbor,out_out_neighbor), environment.polygons):
                                collision_flag=True
                                break
                    if not collision_flag:
                        for in_neighbor in reeb_graph.in_neighbors[out_neighbor]:
                            reeb_graph.add_edge(in_neighbor,out_out_neighbor )
                        for out_out_neighbor2 in reeb_graph.out_neighbors[out_neighbor]:
                            reeb_graph.add_edge(reeb_graph.nodes[i],out_out_neighbor2)
                        reeb_graph.remove_node(out_neighbor)
                        print("removed the node")
                        i=i-1
                        break
        i=i+1


    reeb_graph.clear_repeat_neighbors()
    ### delete the node  
    i=0
    while i< len(reeb_graph.nodes):
        for out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
            if calculate_edge_length(reeb_graph.nodes[i],out_neighbor)<combine_radius:
                collision_flag=False
                for in_neighbor in reeb_graph.in_neighbors[reeb_graph.nodes[i]]:
                    if check_collision((in_neighbor,out_neighbor), environment.polygons):
                        collision_flag=True
                        break
                if not collision_flag:
                    for out_out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
                        if out_out_neighbor is not out_neighbor:
                            if check_collision((out_neighbor,out_out_neighbor), environment.polygons):
                                collision_flag=True
                                break
                if not collision_flag:
                    for in_neighbor in reeb_graph.in_neighbors[reeb_graph.nodes[i]]:
                        reeb_graph.add_edge(in_neighbor,out_neighbor)
                    for out_out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
                        if out_out_neighbor is not out_neighbor:
                            reeb_graph.add_edge(out_neighbor,out_out_neighbor)
                    reeb_graph.remove_node(reeb_graph.nodes[i])
                    print("removed the node")
                    i=i-1
        i=i+1

        

    #######break and reconnect the node################
    dist=300
    reeb_graph.clear_repeat_neighbors()
    for i in range(len(reeb_graph.nodes)-2):
        #check the distance between the out_neighbors and the node
        for out_neighbor in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
            out_dist=calculate_edge_length(out_neighbor,reeb_graph.nodes[i])
            find_near_edge(reeb_graph, environment,out_neighbor,reeb_graph.nodes[i],dist)
            for out_out_neighbor in reeb_graph.out_neighbors[out_neighbor]:
                find_near_edge(reeb_graph,environment, out_out_neighbor,reeb_graph.nodes[i],dist)
            
    # def save_reeb_graph_to_file(reeb_graph, file_path):
    #     # Create a dictionary to store the graph data
    #     node_id_map = {node: idx for idx, node in enumerate(reeb_graph.nodes)}
    #     graph_data = {
    #         'nodes': [(node_id_map[node], node.configuration.tolist(), node.parent, node.is_goal) for node in reeb_graph.nodes],
    #         'in_neighbors': {node_id_map[node]: [node_id_map[neighbor] for neighbor in neighbors] for node, neighbors in reeb_graph.in_neighbors.items()},
    #         'out_neighbors': {node_id_map[node]: [node_id_map[neighbor] for neighbor in neighbors] for node, neighbors in reeb_graph.out_neighbors.items()},
    #     }
        
    #     # Convert the dictionary to a JSON string
    #     graph_data_str = json.dumps(graph_data)
        
    #     # Open the file in write mode
    #     with open(file_path, 'w') as file:
    #         # Write the JSON string to the file
    #         file.write(graph_data_str)

    reeb_graph, start_node, goal_node = augment_reeb_graph(environment, start, target,
                                                        link_step_size=0.001 * environment.width,
                                                        reeb_graph=reeb_graph,
                                                        connection_distance_threshold=0.4 * environment.width)
    
    # Check if the number of nodes exceeds 30, if so, reduce the node number
    if len(reeb_graph.nodes) > 30:
        print(f"Node count ({len(reeb_graph.nodes)}) exceeds 30, reducing nodes...")
        # Using default values for Distance_range and Angle_range
        # You may want to adjust these parameters based on your specific requirements
        Distance_range = 600
        Angle_range = 0.3
        reeb_graph = reduce_node_number(reeb_graph, Distance_range, Angle_range)
    else:
        print(f"Node count ({len(reeb_graph.nodes)}) is within limit (<=30), skipping node reduction")
        
    return reeb_graph

# # Save reeb_graph to a file
# save_reeb_graph_to_file(reeb_graph, "Graph_large.json")
# # Save the environment to a file
# environment.save_to_file("environment.json")
# reeb_graph.draw("brown")
# plt.show()

# Example usage
def reduce_node_number(reeb_graph,Distance_range,Angle_range):
    print(f"the number of the original node: {len(reeb_graph.nodes)}")
    R=len(reeb_graph.nodes)
    StartPoint=reeb_graph.nodes[R-2].configuration
    TargetPoint=reeb_graph.nodes[R-1].configuration
    # Distance_range=600
    i=0
    while i < R-2:
        vec_ij = reeb_graph.nodes[i].configuration - StartPoint
        vec_jk = TargetPoint - reeb_graph.nodes[i].configuration
        
        # Calculate the dot product and magnitudes
        dot_product = np.dot(vec_ij, vec_jk)
        magnitude_ij = np.linalg.norm(vec_ij)
        magnitude_jk = np.linalg.norm(vec_jk)
        
        # Calculate the angle in radians using arctan2
        angle = np.arctan2(np.linalg.norm(np.cross(vec_ij, vec_jk)), np.dot(vec_ij, vec_jk))

        # The angle will be in the range [0, π], use the sign of the z-component of the cross product
        # to determine the direction of the angle.
        cross_product = np.cross(vec_ij, vec_jk)
        if cross_product < 0:  # Assuming the 2D plane is the XY plane.
                angle = -angle
        # Angle_range=0.3
        if abs(angle)>Angle_range:
            if np.linalg.norm(reeb_graph.nodes[i].configuration-StartPoint)>Distance_range or np.linalg.norm(reeb_graph.nodes[i].configuration-TargetPoint)>Distance_range:
                reeb_graph.remove_node(reeb_graph.nodes[i])
                R-=1
        i+=1
    R_new=len(reeb_graph.nodes)
    reeb_graph.clear_repeat_neighbors()
    j=0
    while j < R_new-2:
        if reeb_graph.in_neighbors[reeb_graph.nodes[j]]==[] or reeb_graph.out_neighbors[reeb_graph.nodes[j]]==[]:
            reeb_graph.remove_node(reeb_graph.nodes[j])
            R_new-=1
            j=j-1
        j+=1
   
    reeb_graph.clear_repeat_neighbors()
    # j=R_new-4
    # while j >=0:
    #     j-=1
    #     if reeb_graph.in_neighbors[reeb_graph.nodes[j]]==[] or reeb_graph.out_neighbors[reeb_graph.nodes[j]]==[]:
    #         reeb_graph.remove_node(reeb_graph.nodes[j])
    #         R_new-=1
    #         j=j+1
        

    R=len(reeb_graph.nodes)
    print(f"the number of the new node: {R}")
    return reeb_graph