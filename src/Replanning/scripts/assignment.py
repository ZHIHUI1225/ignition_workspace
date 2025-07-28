# run the whole process of the project
# from the environment to the IP
import matplotlib.pyplot as plt
from Graph import Graph
from Node import Node
import numpy as np
import json
from Graph import save_reeb_graph_to_file, load_reeb_graph_from_file
from Environment import Environment
from Rebuild_large import rebuild_graph,reduce_node_number
from GenerateMatrix import generate_matrix
from normalization import get_normalization_prameters
from IntegerProgramming import Assignment_IP
import sys
import os
sys.path.append('/root/workspace/config')
from config_loader import config

# Visualization
def Visualization(reeb_graph_new,loaded_environment,Result_file,figure_name):
    fig, ax = plt.subplots() 
    loaded_environment.draw("black")
    reeb_graph_new.draw("grey")
    # Function to load Waypoints and RelayPoints from a file
    def load_points_from_file(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data['Waypoints'], data['RelayPoints'],data["FlagB"]
    waypoints_file_path = Result_file
    Waypoints, RelayPoints,FlagB = load_points_from_file(waypoints_file_path)

    # generate the waypoints matrix and flag matrix
    OrginalWayPoints=Waypoints.copy()

    WayPointM=[RelayPoints[0][0]]
    FlagM=[0]
    RelayPointM=[]
    for Arc in RelayPoints:
        RelayPointM.append(Arc[0])

    while OrginalWayPoints!=[]:
        for Arc in OrginalWayPoints:
            if Arc[0]==WayPointM[-1]:
                WayPointM.append(Arc[1])
                OrginalWayPoints.remove(Arc)
                if Arc[1] in RelayPointM:
                    FlagM.append(1)
                else:
                    FlagM.append(0)
                break
    # Plot arrows of waypoints and relay points
    for i, j,k in Waypoints:
        start = reeb_graph_new.nodes[i].configuration
        end = reeb_graph_new.nodes[j].configuration
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  width=2, head_width=20, head_length=15, fc='g', ec='g',zorder=5)  # Green arrow for waypoints


    for i, j,k in RelayPoints:
        start = reeb_graph_new.nodes[i].configuration
        end = reeb_graph_new.nodes[j].configuration
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  width=2,linestyle='-.',head_width=20, head_length=15, fc='r', ec='r',zorder=5)  # Blue arrow for relay points

    # Save the figure
    plt.show()
    plt.savefig(figure_name)
    # plt.show()

# Function to load Waypoints and RelayPoints from a file
def load_points_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Waypoints'], data['RelayPoints'],data["FlagB"]

def load_matrices_from_file(file_path):
    data = np.load(file_path)
    Ec = data['Ec']
    El = data['El']
    Ad = data['Ad']
    Cr=data['Cr']
    Cl=data['Cl']
    return Ec, El, Ad, Cr, Cl

def GetWaypoints(Result_file,Matrix_file,Save_file):
    Waypoints, RelayPoints,FlagB = load_points_from_file(Result_file)
    Ec, El, Ad ,Cr, Cl= load_matrices_from_file(Matrix_file)
# generate the waypoints matrix and flag matrix
    OrginalWayPoints=Waypoints.copy()

    WayPointM=[RelayPoints[0][0]]
    FlagM=[0]
    RelayPointM=[]
    for Arc in RelayPoints:
        RelayPointM.append(Arc[0])

    while OrginalWayPoints!=[]:
        for Arc in OrginalWayPoints:
            if Arc[0]==WayPointM[-1]:
                WayPointM.append(Arc[1])
                OrginalWayPoints.remove(Arc)
                if Arc[1] in RelayPointM:
                    FlagM.append(1)
                else:
                    FlagM.append(0)
                break
    # print(WayPointM)
    # print(FlagM)
    Flagb=FlagM.copy()
    b=0
    for i in range(len(FlagM)):
        if FlagM[i]==1:
            # if abs(Cr[WayPointM[i-1],WayPointM[i],WayPointM[i+1]])<abs(Cl[WayPointM[i-1],WayPointM[i],WayPointM[i+1]]):# turn right:
            #     Flagb[i]=1
            # else:
            #     Flagb[i]=-1
            if FlagB[b]==1:# turn right
                print(f"Relay point {i} turn right, angle is {Ec[WayPointM[i-1],WayPointM[i],WayPointM[i+1]]}, new angle is {Cr[WayPointM[i-1],WayPointM[i],WayPointM[i+1]]},Cl={Cl[WayPointM[i-1],WayPointM[i],WayPointM[i+1]]}")
                Flagb[i]=-1
            else:
                Flagb[i]=1
                print(f"Relay point {i} turn left, angle is {Ec[WayPointM[i-1],WayPointM[i],WayPointM[i+1]]}, new angle is {Cl[WayPointM[i-1],WayPointM[i],WayPointM[i+1]]}，Cr={Cr[WayPointM[i-1],WayPointM[i],WayPointM[i+1]]}")
            b=b+1
                
                # save the result to a file
    data = {
        'Waypoints': WayPointM,
        'Flags': FlagM,
        'FlagB':Flagb
    }
    with open(Save_file, 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
# Load configuration from config file
    case = config.case
    N = config.N
    arc_range = config.arc_range
    
    print(f"Loaded configuration - Case: {case}, N: {N}, Arc range: {arc_range}")
    
    Orginal_graph_file_path = "Graph_" + case +".json"
    New_graph_file_path = "Graph_new_" + case +".json"
    matrix_file_path = "Estimated_matrices_" + case +".npz"
    parameter_file_name=f"Normalization{N}_" + case +".json"
    reeb_graph = load_reeb_graph_from_file("/root/workspace/data/" + Orginal_graph_file_path)

    loaded_environment = Environment.load_from_file("/root/workspace/data/environment_" + case +".json")

    # Load start and goal positions from the config or environment file
    start = config.start
    goal = config.goal
    
    # If start/goal are not defined in config, try to load from environment file
    if start is None or goal is None:
        with open("/root/workspace/data/environment_" + case +".json", 'r') as file:
            env_data = json.load(file)
            
        if 'start_pose' in env_data and 'goal_pose' in env_data:
            start = np.array(env_data['start_pose']) if start is None else np.array(start)
            goal = np.array(env_data['goal_pose']) if goal is None else np.array(goal)
            print(f"Loaded start pose from environment: {start}")
            print(f"Loaded goal pose from environment: {goal}")
        else:
            # Fallback to manual definition if not found in file
            print("Warning: start_pose or goal_pose not found in environment file, using manual values")
            start = np.array([100.0, 155.0,0.4*np.pi]) if start is None else np.array(start)
            goal = np.array([780.0, 455.0, np.pi]) if goal is None else np.array(goal)
    else:
        start = np.array(start)
        goal = np.array(goal)
        print(f"Using start pose from config: {start}")
        print(f"Using goal pose from config: {goal}")
    # rebuild the reeb graph
    reeb_graph_new = rebuild_graph(reeb_graph, loaded_environment,start, goal)
    # reeb_graph_new=reduce_node_number(reeb_graph_new,Distance_range=Distance_range,Angle_range=Angle_range)
    save_reeb_graph_to_file(reeb_graph_new, "/root/workspace/data/" + New_graph_file_path )
    reeb_graph_new.draw("steelblue")
    # reeb_graph.draw("grey")
    Environment.draw(loaded_environment,"black")
    plt.axis('equal')
    plt.autoscale()
    plt.show()
    # ########## generate the matrix and save it ############
    ######### only need to recalculate when graph change ########

    generate_matrix(reeb_graph_new,"/root/workspace/data/" + matrix_file_path)
    # ###### get the normalization parameters and save #########
    # #### only need to recalculate when N and graph change ####
    Flag=get_normalization_prameters(N,arc_range,"/root/workspace/data/" + parameter_file_name,"/root/workspace/data/" + matrix_file_path)
    # Flag=True
    if Flag:
    ########## Integer programming solve ##########
        Assignment_IP(N,arc_range,Matrices_file="/root/workspace/data/" + matrix_file_path,Parameter_file="/root/workspace/data/" + parameter_file_name,Result_file="/root/workspace/data/" + f"AssignmentResult{N}"+case+".json")
        Visualization(reeb_graph_new,loaded_environment,Result_file="/root/workspace/data/" + f"AssignmentResult{N}"+case+".json",figure_name=f"Assignment_Result{N}.png")
        GetWaypoints(Result_file="/root/workspace/data/" + f"AssignmentResult{N}"+case+".json",Matrix_file="/root/workspace/data/" + matrix_file_path,Save_file="/root/workspace/data/" + f"WayPointFlag{N}"+case+".json")
    else:
        print("infeasible")

