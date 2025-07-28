# calucalte the pthe_planning result under different number agents
import json
import os
import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from GenerateMatrix import load_reeb_graph_from_file
from Planing_functions import Initial_Guess,Planning_normalization,get_normalization_prams,get_safe_corridor,Planning_error_withinSC

# Add config path to sys.path and load configuration
sys.path.append('/root/workspace/config')
from config_loader import config
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

# Example usage
# Load configuration parameters
case = config.case
N = config.N
arc_range = config.arc_range
phi0 = config.phi0  # Load phi0 from config

print(f"Configuration loaded:")
print(f"  Case: {case}")
print(f"  Number of robots (N): {N}")
print(f"  Arc range: {arc_range}")
print(f"  Initial angle (phi0): {phi0:.4f} ({phi0/np.pi:.2f}π)")

# Generate file paths using configuration
file_path = config.get_full_path(config.file_path, use_data_path=True)
environment_file = config.get_full_path(config.environment_file, use_data_path=True)

# Load reeb graph
reeb_graph = load_reeb_graph_from_file(file_path)

# Calculate distances and parameters
NumNodes = len(reeb_graph.nodes)
Start_node = reeb_graph.nodes[NumNodes-2].configuration
End_node = reeb_graph.nodes[NumNodes-1].configuration
Distances = np.linalg.norm(End_node-Start_node)
X_distance = abs(End_node[0]-Start_node[0])
Y_distance = abs(End_node[1]-Start_node[1])
Arc_min = arc_range[0]  # Use config arc range
Arc_max = arc_range[1]  # Use config arc range
# N_min=int((X_distance+Y_distance)/Arc_max)
# N_max=int(Distances/Arc_min)-4
# phi0 now loaded from config above

# Generate file paths using config
assignment_result_file = config.get_full_path(config.assignment_result_file, use_data_path=True)
Initial_Guess_file_path = config.get_full_path(f"InitialGuess{N}{case}.json", use_data_path=True)
Initial_Guess_figure_file = config.get_full_path(f"InitialGuess{N}{case}.png", use_data_path=True)
waypoints_file_path = config.get_full_path(config.waypoints_file_path, use_data_path=True)
Normalization_path = config.get_full_path(config.Normalization_planning_path, use_data_path=True)

if os.path.exists(assignment_result_file):
    GetWaypoints(assignment_result_file, config.get_full_path(f"Estimated_matrices_{case}.npz", use_data_path=True), waypoints_file_path)
    # get_normalization_prams(waypoints_file_path, Normalization_path, reeb_graph, Initial_Guess_file_path)
    safe_corridor, Distance, Angle, vertex = get_safe_corridor(reeb_graph, waypoints_file_path, environment_file)
    # Initial_Guess(reeb_graph, phi0, waypoints_file_path, environment_file, safe_corridor, Normalization_path, Initial_Guess_file_path, Initial_Guess_figure_file)
    # get_normalization_prams(waypoints_file_path=waypoints_file_path, Normalization_path=Normalization_path, reeb_graph=reeb_graph, Initial_Guess_file_path=Initial_Guess_file_path)
    
    # Use GA planning result as initial guess for further optimization
    # config.Result_file = f"Optimization_GA_{N}_IG_norma{case}.json" - the output from GA_planning.py
    GA_result_file_path = config.get_full_path(config.Result_file, use_data_path=True)  # This is the GA result file
    
    print(f"Checking for GA result file: {GA_result_file_path}")
    if os.path.exists(GA_result_file_path):
        print(f"✅ Using GA result as initial guess: {GA_result_file_path}")
        Planning_error_withinSC(waypoints_file_path, Normalization_path, environment_file, safe_corridor, reeb_graph, phi0, GA_result_file_path, Result_file=config.get_full_path(f"Optimization_withSC_path{N}{case}.json", use_data_path=True), figure_file=config.get_full_path(f"Optimization_winthSC_path{N}{case}.png", use_data_path=True))
    else:
        print(f"❌ GA result file not found: {GA_result_file_path}")
        print(f"Using basic initial guess instead: {Initial_Guess_file_path}")
        Planning_error_withinSC(waypoints_file_path, Normalization_path, environment_file, safe_corridor, reeb_graph, phi0, Initial_Guess_file_path, Result_file=config.get_full_path(f"Optimization_withSC_path{N}{case}.json", use_data_path=True), figure_file=config.get_full_path(f"Optimization_winthSC_path{N}{case}.png", use_data_path=True))
    
    # Planning_normalization(waypoints_file_path,Normalization_path,environment_file,safe_corridor,reeb_graph,phi0,GA_result_file_path,Result_file=f"Optimization_normalized_path{N}"+case+".json",figure_file=f"Optimization_normalized_path{N}"+case+".png")
else:
    print(f"❌ Assignment result file not found: {assignment_result_file}")
    print(f"Please run the assignment optimization first to generate this file.")

