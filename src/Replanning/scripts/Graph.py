import numpy as np
import matplotlib.pyplot as plt
from Node import Node
import json
def path_distances(graph, target):
    """Inefficient method for finding workspace distances to a target node in a graph."""
    distances = {n: np.inf for n in graph.nodes}
    distances[target] = 0
    done = False
    while not done:
        done = True
        for n, current_dist in list(distances.items()):
            for neighbor in graph.out_neighbors[n]:
                update_dist = 1 + distances[neighbor]
                if update_dist < current_dist:
                    distances[n] = update_dist
                    done = False
    return distances


def load_reeb_graph_from_file(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the JSON string from the file
        graph_data_str = file.read()
    
    # Convert the JSON string back to a dictionary
    graph_data = json.loads(graph_data_str)
    
    # Reconstruct the reeb_graph object
    reeb_graph = Graph(nodes=[])
    node_map = {}
    
    # Reconstruct nodes
    for node_id, configuration, parent, is_goal in graph_data['nodes']:
        node = Node(np.array(configuration), parent, is_goal)
        node_map[node_id] = node
        reeb_graph.nodes.append(node)
    
    # Reconstruct in_neighbors
    for node_id, neighbor_ids in graph_data['in_neighbors'].items():
        node = node_map[int(node_id)]
        reeb_graph.in_neighbors[node] = [node_map[int(neighbor_id)] for neighbor_id in neighbor_ids]
    
    # Reconstruct out_neighbors
    for node_id, neighbor_ids in graph_data['out_neighbors'].items():
        node = node_map[int(node_id)]
        reeb_graph.out_neighbors[node] = [node_map[int(neighbor_id)] for neighbor_id in neighbor_ids]
    
    return reeb_graph

class Graph:
    """Graph of configurations to be connected by closest distance."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.in_neighbors = {n: [] for n in self.nodes}
        self.out_neighbors = {n: [] for n in self.nodes}

    def add_node(self, node):
        self.nodes.append(node)
        self.in_neighbors[node] = []
        self.out_neighbors[node] = []
    #delete node from the graph
    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
        if node in self.in_neighbors:
            self.in_neighbors.pop(node)
        if node in self.out_neighbors:
            self.out_neighbors.pop(node)
        for neighbors in self.in_neighbors.values():
            if node in neighbors:
                neighbors.remove(node)
        for neighbors in self.out_neighbors.values():
            if node in neighbors:
                neighbors.remove(node)

    #use node2 to replace node1
    def replace_node(self, node1,node2):
        if node1 not in self.nodes:
            print(f"Node {node1} does not exist in the graph.")
            return
        self.add_node(node2)

        if node1 in self.in_neighbors:
            for neighbor in self.in_neighbors[node1]:
                self.add_edge(neighbor, node2)
        if node1 in self.out_neighbors:
            for neighbor in self.out_neighbors[node1]:
                self.add_edge(node2, neighbor)
        # check if the node is in the in_neighbors of the node
        for node, neighbors in self.in_neighbors.items():
            if node1 in neighbors:
                self.in_neighbors[node].remove(node1)
                self.in_neighbors[node].append(node2)
        # check if the node is in the out_neighbors of the node
        for node, neighbors in self.out_neighbors.items():
            if node1 in neighbors:
                self.out_neighbors[node].remove(node1)
                self.out_neighbors[node].append(node2)

        self.remove_node(node1)

    def clear_repeat_neighbors(self):
        """
        Remove repeated neighbors in in_neighbors and out_neighbors of every node.
        """
        for node in self.nodes:
            if node in self.in_neighbors:
                self.in_neighbors[node] = list(set(self.in_neighbors[node]))
            if node in self.out_neighbors:
                self.out_neighbors[node] = list(set(self.out_neighbors[node]))
     
    #combine node1 and node2 to a new node, and connect the new node to the in_neighbors of node1 and out_neighbors of node2
    def combine_nodes(self, node1, node2, new_node):
        if node1 not in self.nodes:
            print(f"Node {node1} does not exist in the graph.")
            return
        if node2 not in self.nodes:
            print(f"Node {node2} does not exist in the graph.")
            return
        self.add_node(new_node)
        for in_neighbor in self.in_neighbors[node1]:
            self.add_edge(in_neighbor, new_node)
        for in_neighbor in self.in_neighbors[node2]:
            if in_neighbor is not node1:
                self.add_edge(in_neighbor, new_node)
        for out_neighbor in self.out_neighbors[node1]:
            if out_neighbor is not node2:
                self.add_edge(new_node, out_neighbor)
        for out_neighbor in self.out_neighbors[node2]:
            self.add_edge(new_node, out_neighbor)


        ### check if the node is in the in_neighbors of the node
        for node, neighbors in self.in_neighbors.items():
            while node1 in neighbors:
                self.in_neighbors[node].remove(node1)

            while node2 in neighbors:
                self.in_neighbors[node].remove(node2)

        ##### check if the node is in the out_neighbors of the node
        for node, neighbors in self.out_neighbors.items():
            while node1 in neighbors:
                self.out_neighbors[node].remove(node1)

            while node2 in neighbors:
                self.out_neighbors[node].remove(node2)


        self.remove_node(node1)
        self.remove_node(node2)
        


    def add_edge(self, node1, node2):
        if node1 not in self.out_neighbors:
            self.out_neighbors[node1] = []
        if node2 not in self.in_neighbors:
            self.in_neighbors[node2] = []
        self.out_neighbors[node1].append(node2)
        self.in_neighbors[node2].append(node1)
        # if node1 not in self.out_neighbors:
        #     self.out_neighbors.pop(node1)
        # self.out_neighbors[node1].append(node2)
        # if node2 not in self.in_neighbors:
        #     self.out_neighbors.pop(node2)
        # self.in_neighbors[node2].append(node1)

    def remove_edge(self, node1, node2):
        if node1 in self.out_neighbors[node1]:
            self.out_neighbors[node1].remove(node2)
        if node2 in self.in_neighbors[node2]:
            self.in_neighbors[node2].remove(node1)

    def draw(self, color):
        xs = [node.configuration[0] for node in self.nodes]
        ys = [node.configuration[1] for node in self.nodes]
        plt.scatter(xs, ys, s=15, color=color)
        for node, neighbors in self.in_neighbors.items():
            for neighbor in neighbors:
                xs = [node.configuration[0], neighbor.configuration[0]]
                ys = [node.configuration[1], neighbor.configuration[1]]
                plt.plot(xs, ys, color=color)

def save_reeb_graph_to_file(reeb_graph, file_path):
    # Create a dictionary to store the graph data
    node_id_map = {node: idx for idx, node in enumerate(reeb_graph.nodes)}
    graph_data = {
        'nodes': [(node_id_map[node], node.configuration.tolist(), node.parent, node.is_goal) for node in reeb_graph.nodes],
        'in_neighbors': {node_id_map[node]: [node_id_map[neighbor] for neighbor in neighbors] for node, neighbors in reeb_graph.in_neighbors.items()},
        'out_neighbors': {node_id_map[node]: [node_id_map[neighbor] for neighbor in neighbors] for node, neighbors in reeb_graph.out_neighbors.items()}
    }
    
    # Convert the dictionary to a JSON string
    graph_data_str = json.dumps(graph_data)
    
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the JSON string to the file
        file.write(graph_data_str)