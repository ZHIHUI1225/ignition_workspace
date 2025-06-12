# get the Estimated Matrix from the graph
import matplotlib.pyplot as plt
from Graph import Graph, load_reeb_graph_from_file
from Node import Node
import numpy as np
import json

def generate_matrix(reeb_graph,save_file_path):
    # Example usage
    # file_path = "Graph.json"
    # reeb_graph = load_reeb_graph_from_file(file_path)
    # reeb_graph.draw("brown")
    # plt.show()
    # Get the number of nodes in the graph
    print(len(reeb_graph.nodes))
    R=len(reeb_graph.nodes)
    # Get the Adjeacency Matrix and the distance matrix from the graph
    Ad=np.zeros((R,R))
    El=np.zeros((R,R))
    for i in range(R):
        for j in range(R):
            if i==j:
                El[i,j]=0
            elif reeb_graph.nodes[j] in reeb_graph.out_neighbors[reeb_graph.nodes[i]]:
                El[i,j]=np.linalg.norm(reeb_graph.nodes[i].configuration-reeb_graph.nodes[j].configuration)
                Ad[i,j]=1
    # Get the curvature matrix from the graph
    Ec=np.zeros((R,R,R))
    for i in range(R):
        for j in range(R):
            for k in range(R):
                if Ad[i,j]==1 and Ad[j,k]==1:
                    # Calculate the vectors
                    vec_ij = reeb_graph.nodes[j].configuration - reeb_graph.nodes[i].configuration
                    vec_jk = reeb_graph.nodes[k].configuration - reeb_graph.nodes[j].configuration
                    
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
                    
                    # Store the angle
                    Ec[i, j, k] = angle
    #Variables set
    delta=1.57 # the angle between two pushing sides (pi/2)
    # generate the cooresponding turn matrix
    # turn left s[k+1]-s[k]=-1
    Cl=Ec.copy()
    # turn right s[k+1]-s[k]=1
    Cr=Ec.copy()
    for i in range(R):
        for j in range(R):
            for z in range(R):
                    if Ec[i,j,z]!=0:
                        Cl[i,j,z]=Cl[i,j,z]-delta
                        Cr[i,j,z]=Cr[i,j,z]+delta
                        if Cl[i,j,z]< -np.pi:
                            Cl[i,j,z]=Cl[i,j,z]+2*np.pi
                        elif Cl[i,j,z]> np.pi:
                            Cl[i,j,z]=Cl[i,j,z]-2*np.pi
                        if Cr[i,j,z]< -np.pi:   
                            Cr[i,j,z]=Cr[i,j,z]+2*np.pi
                        elif Cr[i,j,z]> np.pi:
                            Cr[i,j,z]=Cr[i,j,z]-2*np.pi
    nonzero_indices = np.nonzero(Ec)
    # print(Ec[nonzero_indices])
    max_value = np.max(Ec[nonzero_indices])
    min_value = np.min(Ec[nonzero_indices])
    nonzero= np.nonzero(Cr)
    max_Cr = np.max(Cr[nonzero])
    min_Cr = np.min(Cr[nonzero])
    # print(Ec[nonzero_indices]-Cr[nonzero])

    # save the Matrices Ec, El, Ad to the file
    def save_matrices_to_file(file_path, Ec, El, Ad,Cr,Cl):
        np.savez(file_path, Ec=Ec, El=El, Ad=Ad,Cr=Cr,Cl=Cl)

    # Count the number of non-zero elements
    non_zero_elements = np.count_nonzero(Ec)

    # Count the number of positive elements
    positive_elements = np.sum(Ec > 0)

    # Count the number of negative elements
    negative_elements = np.sum(Ec < 0)


    # Example usage
    save_matrices_to_file(save_file_path, Ec, El, Ad,Cr=Cr,Cl=Cl)

def get_NormalizationMatrix(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    length_min=data['Length_min']
    length_max=data['Length_max']
    curvature_max=data['Curvature_max']
    curvature_min=data['Curvature_min']
    theta_c=data['theta_c']
    theta_l=data['theta_l']
    return theta_l,theta_c,length_min,length_max,curvature_min,curvature_max