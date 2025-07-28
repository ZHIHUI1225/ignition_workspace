# integer programming
# get relay points and waypoints pushing side
import json
import time
import gurobipy as gp
from gurobipy import GRB
from gurobipy import abs_
import numpy as np
import matplotlib.pyplot as plt
def generate_matrix(R):
    if R < 2:
        raise ValueError("R must be at least 2 to have a center with four elements.")
    
    matrix = np.zeros((R, R), dtype=int)
    center = R // 2
    for i in range(R):
        for j in range(R):
            if j > i and j < i+3:
                matrix[i, j] = 1
    
    return matrix
# Define the data
# load the Normalization matrix of N agents.
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

def load_matrices_from_file(file_path):
    data = np.load(file_path)
    Ec = data['Ec']
    El = data['El']
    Ad = data['Ad']
    Cr=data['Cr']
    Cl=data['Cl']
    return Ec, El, Ad, Cr, Cl

def Assignment_IP(N,arc_range,Matrices_file,Parameter_file,Result_file):

    # pushing sides 
    Ns=4

    # Example usage
    Ec, El, Ad ,Cr, Cl= load_matrices_from_file(Matrices_file)

    # the number of candidat nodes 
    R=np.shape(Ad)[0]

    # Start timing
    start_time = time.time()
    #Variables set
    delta=np.pi/2 # the angle between two pushing sides (pi/2)
    # generate the cooresponding turn matrix
    # Create a new model
    model = gp.Model("Assignment")
    model.Params.PreSparsify = 2
    model.Params.SoftMemLimit = 32
    # model.Params.Threads = 8
    # model.Params.NodefileStart = 0.5
    # model.Params.NodefileStart = 2 * 1024 * 1024 * 1024
    model.Params.MIPGap=0.01
    # Create variables
    num_items = R*R*N
    # x_ij^k = 1 if the k-th arc is from the relay point i to the relay point j
    x= model.addVars(R,R,N, vtype=GRB.BINARY, name="x")
    #y_ij^k = 1 if from waypoint i to waypoint j belongs to the k-th arc
    y= model.addVars(R,R,N, vtype=GRB.BINARY, name="y")
    # s_k pushing side 1,2,3,4
    # s= model.addVars(N, vtype=GRB.INTEGER, name="s")
    auxiliary_v=model.addVars(R,R,R,N-1, vtype=GRB.BINARY, name="auxiliary_v")
    b = model.addVars(N-1,vtype=GRB.BINARY, name="b")
    #to show the turn direction at the relay points
    # auxiliary_t=model.addVars(N, vtype=GRB.BINARY, name="auxiliary_t")
    # weight parameters
    al=1# the weight of the length
    ac=1# the weight of the curvature
    # acr=1 # the weight of the curvature at relay points

    theta_l,theta_c,length_min,length_max,curvature_min,curvature_max=get_NormalizationMatrix(file_name=Parameter_file)

    # Set objective

    # the sum of the length of the path
    SumL=gp.quicksum(y[i,j,k]*El[i,j]*Ad[i,j] for i in range(R) for j in range(R) for k in range(N))

    # SumV=gp.quicksum((gp.quicksum(y[i,j,k]*El[i,j]*Ad[i,j] for i in range(R) for j in range(R))-SumL/N)**2 for k in range(N))
    # the sum of the curvature of the waypoints
    SumC=gp.quicksum( y[j,i,k]*y[i,z,k]*Ad[j,i]*Ad[i,z]*np.abs(Ec[j,i,z]) for i in range(R) for j in range(R)  for z in range(R) for k in range(N))
    # the sum of the curvature of the relay points 
    #SumCR= gp.quicksum(y[z, j, k] *y[j, r, k+1] * Ad[z, j] * Ad[j, r] * (1 + (s[k+1] - s[k]) * delta) for i in range(R) for j in range(R) for z in range(R) for r in range(R) for k in range(N-1))
    # add the equality constraints of auxiliary_v to avoid direct multiplication of the variables
    for j in range(R):
        for z in range(R):
            if Ad[j,z]==0:
                model.addConstr(gp.quicksum(auxiliary_v[j,z,r,k] for r in range(R) for k in range(N-1))==0)
            else:
                for r in range(R):
                    if Ad[z,r]==0:
                        model.addConstr(gp.quicksum(auxiliary_v[j,z,r,k] for j in range(R) for k in range(N-1))==0)
                    else:
                        for k in range(N-1):
                            model.addConstr(auxiliary_v[j, z, r, k] == y[j, z, k] * y[z, r, k+1])
                            # model.addConstr(auxiliary_v[j, z, r, k]*(b[k]*np.abs(Cl[j, z, r])+(1-b[k])*np.abs(Cr[j, z, r]))<=0.5*np.pi)


    # SumCR=gp.quicksum(auxiliary_v[j,z,r,k]*(Ec[j,z,r] + (s[k+1] - s[k]) * delta)  for j in range(R) for z in range(R) for r in range(R) for k in range(N-1))
    SumCR=gp.quicksum(auxiliary_v[z,j,r,k]*(b[k]*np.abs(Cl[z,j,r])+(1-b[k])*np.abs(Cr[z,j,r]))  for j in range(R) for z in range(R) for r in range(R) for k in range(N-1))
    # Get the absolute value of SumCR
    # SumCRAbs=gp.abs_(SumCR)
    # model.setObjective(al * SumL + ac * SumC + acr * SumCR, GRB.MINIMIZE)
    model.setObjective(al *theta_l* (SumL-length_min) + ac * theta_c*(SumC+SumCR-curvature_min), GRB.MINIMIZE)

    # Add constraint
   # angle constraints at waypoints
    for j in range(R):
        for z in range(R):
            for k in range(N):
                if Ad[j,z]==0:
                    model.addConstr(y[j,z,k]==0)
                else:
                    for r in range(R):
                        if Ad[z,r]==0:
                            model.addConstr(y[z,r,k]==0)
                        # else:
                        #     model.addConstr(y[j,z,k]*y[z,r,k]*Ad[j,z]*Ad[z,r]*np.abs(Ec[j,z,r])<=np.pi*0.8)

    # Unique kth arc
    model.addConstr(gp.quicksum(x[i,j,k] for i in range(R) for j in range(R) for k in range(N))== N , "Sum_x")
    for k in range(N):
        model.addConstr(gp.quicksum(x[i,j,k] for i in range(R) for j in range(R)) == 1, f"UniqueArc_x{k}")
        # constraints on the length of each arc to balance the length of each arc
        model.addConstr(gp.quicksum(y[i,j,k]*Ad[i,j]*El[i,j] for i in range(R) for j in range(R)) >= arc_range[0], "NumberLowB_y")
        model.addConstr(gp.quicksum(y[i,j,k]*Ad[i,j]*El[i,j] for i in range(R) for j in range(R)) <=arc_range[1], "NumberHighB_y")
    # Start node:
    model.addConstr(gp.quicksum(x[R-2,j,0] for j in range(R)) == 1, "StartNode_x")
    model.addConstr(gp.quicksum(y[R-2,j,0]*Ad[R-2,j] for j in range(R)) == 1, "StartNode_y")
    # End node:
    model.addConstr(gp.quicksum(x[j,R-1,N-1] for j in range(R)) == 1, "EndNode_x")
    model.addConstr(gp.quicksum(y[j,R-1,N-1]*Ad[j,R-1] for j in range(R)) == 1, "EndNode_y")
    # Flow balance:
    for j in range(R):
        for k in range(N):
            if k<N-1:
                model.addConstr(gp.quicksum(x[i,j,k] for i in range(R)) == gp.quicksum(x[j,z,k+1] for z in range(R)), "FlowBalance_x")
            model.addConstr(gp.quicksum(y[i,j,k]*Ad[i,j] for i in range(R) ) +gp.quicksum(x[j,i,k] for i in range(R) )== gp.quicksum(y[j,z,k]*Ad[j,z] for z in range(R))+gp.quicksum(x[z,j,k] for z in range(R)), "FlowBalance_y")    
    # The relation between x and y
    for k in range(N):
        model.addConstr(gp.quicksum(x[i,j,k]*y[i,z,k]*Ad[i,z] for i in range(R) for j in range(R) for z in range(R)) == 1, "Relation_x_y_out")
        model.addConstr(gp.quicksum(x[i,j,k]*y[z,j,k]*Ad[z,j] for i in range(R) for j in range(R) for z in range(R)) == 1, "Relation_x_y_in")
    # the flow of the waypoints
    for j in range(R):
        model.addConstr(gp.quicksum(y[i,j,k]*Ad[i,j] for i in range(R) for k in range(N)) <=1, "Flow_y_out")
        model.addConstr(gp.quicksum(y[j,i,k]*Ad[j,i] for i in range(R) for k in range(N)) <=1, "Flow_y_in")
    # two pushing side s cannot be the same
    eps = 0.0001
    M = 10 + eps
    # for k in range(N-1):
    #     model.addConstr(s[k+1]-s[k]<= -eps + M*b[k], name="bigM_constr1")
    #     model.addConstr(s[k+1]-s[k]>=eps-(1-b[k])*M, name="bigM_constr2")
    #     model.addConstr(s[k+1]-s[k]<= 1, name="bigM_constr3")
    #     model.addConstr(s[k+1]-s[k]>= -1, name="bigM_constr4")


    # Optimize model
    model.optimize()
    # End timing
    end_time = time.time()
    # Calculate the time taken
    time_taken = end_time - start_time

    Waypoints=[]
    RelayPoints=[]
    FlagB=[]
    # Get the indices of all non-zero elements in auxiliary_v
    nonzero_indices = []
    for i in range(R):
        for j in range(R):
            for k in range(R):
                for l in range(N-1):
                    if auxiliary_v[i, j, k, l].X > 0.5:  # Check if the value is non-zero
                        nonzero_indices.append((i, j, k, l))
                        angle=auxiliary_v[i, j, k, l].X *(b[l].X *np.abs(Cl[i, j, k])+(1-b[l].X )*np.abs(Cr[i, j, k]))
                        print(f" angle at point {i} through point {j} to point{k} is {angle}")
    print("Indices of non-zero elements in auxiliary_v:", nonzero_indices)
    # Print the results
    if model.status == GRB.OPTIMAL:
        print("Selected items:")
        for k in range(N):
            Sumx=0
            for i in range(R):
                for j in range(R):
                    if abs(y[i, j, k].X-1)<0.1:
                        print(f"Waypoint {i} to {j} in arc {k}")
                        Waypoints.append([i,j,k])
                        
                    if abs(x[i, j, k].X-1)<0.1:
                        print(f"Relay point {i} to Relay point {j} of arc {k}")
                        RelayPoints.append([i,j,k])
            if k<N-1:
                if b[k].X==0: #right
                    FlagB.append(1)
                else: #left
                    FlagB.append(-1)
            #             Sumx=Sumx+1
            # print(f"Sumx={Sumx} for arc {k}")
        print(f"Total value: {model.objVal}")
        model.printQuality()
        #make the waypoints in right sequence
        Waypoints_updated=[]
        while len(Waypoints)!=0:
            for Arc in Waypoints:
                if Arc[0]==RelayPoints[0][0]:
                    Waypoints_updated.append(Arc)
                    Waypoints.remove(Arc)
                    break
                if Waypoints_updated!=[] and Arc[0]==Waypoints_updated[-1][1]:
                    Waypoints_updated.append(Arc)
                    Waypoints.remove(Arc)
                    break
            
    else:
        print("No optimal solution found.")

    # Function to save Waypoints and RelayPoints to a file
    def save_points_to_file(file_path, waypoints, relay_points,flagB,value):
        data = {
            'Waypoints': waypoints,
            'RelayPoints': relay_points,
            'FlagB':flagB,
            'TotalValue':value
        }
        with open(file_path, 'w') as file:
            json.dump(data, file)

    # Example usage
    waypoints_file_path = Result_file
    save_points_to_file(waypoints_file_path, Waypoints_updated, RelayPoints,FlagB,model.objVal)

    print("Finished and saved the result to AssignmentResult.json")
    print(f"Solving time taken: {time_taken:.2f} seconds")
# # draw the path
# # Example usage
# file_path = "Graph.json"
# reeb_graph = load_reeb_graph_from_file(file_path)
# # reeb_graph.draw("brown")
# # plot arrows of waypoints and replay points
# for i in range(len(Waypoints)):
#     start = reeb_graph.nodes[Waypoints[i][0]].configuration
#     end =reeb_graph.nodes[Waypoints[i][1]].configuration
#     plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
#               head_width=20, head_length=25, fc='g', ec='g')  # Red arrow for waypoints
# for i in range(len(RelayPoints)):
#     start =  reeb_graph.nodes[RelayPoints[i][0]].configuration
#     end = reeb_graph.nodes[RelayPoints[i][1]].configuration
#     plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
#               head_width=20, head_length=25, fc='b', ec='b')  # Blue arrow for relay points
    
# plt.autoscale() 
# plt.show()