# normalization
# to get the Nadir and Utopia points
# length term:[1367.6,4518.71]
#curvature term:[0.777,38.99]
import json

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from GenerateMatrix import load_reeb_graph_from_file
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
# the number of relay points N

#the number of agents
def single_objective_solve(N,al,ac,Obs,arc_range,Matrix_file_path):

    # pushing sides 
    Ns=4

    def load_matrices_from_file(file_path):
        data = np.load(file_path)
        Ec = data['Ec']
        El = data['El']
        Ad = data['Ad']
        Cr=data['Cr']
        Cl=data['Cl']
        return Ec, El, Ad, Cr, Cl

    # Example usage
    Ec, El, Ad,Cr,Cl = load_matrices_from_file(Matrix_file_path)

    # the number of candidat nodes 
    R=np.shape(Ad)[0]


    # Create a new model
    model = gp.Model("Assignment")
    model.Params.PreSparsify = 2
    model.Params.SoftMemLimit = 30
    # Create variables
    num_items = R*R*N
    # x_ij^k = 1 if the k-th arc is from the relay point i to the relay point j
    x= model.addVars(R,R,N, vtype=GRB.BINARY, name="x")
    #y_ij^k = 1 if from waypoint i to waypoint j belongs to the k-th arc
    y= model.addVars(R,R,N, vtype=GRB.BINARY, name="y")
    # s_k pushing side 1,2,3,4
    s= model.addVars(N, vtype=GRB.INTEGER, name="s")
    auxiliary_v=model.addVars(R,R,R,N, vtype=GRB.BINARY, name="auxiliary_v")
    b = model.addVars(N-1,vtype=GRB.BINARY, name="b")
    #to show the turn direction at the relay points
    # auxiliary_t=model.addVars(N, vtype=GRB.BINARY, name="auxiliary_t")
    # weight parameters
    # al=0# the weight of the length
    # ac=1# the weight of the curvature
    # acr=1 # the weight of the curvature at relay points
    # Set objective

    # the sum of the length of the path
    SumL=gp.quicksum(y[i,j,k]*El[i,j]*Ad[i,j] for i in range(R) for j in range(R) for k in range(N))

    # SumV=gp.quicksum((gp.quicksum(y[i,j,k]*El[i,j]*Ad[i,j] for i in range(R) for j in range(R))-SumL/N)**2 for k in range(N))
    # the sum of the curvature of the waypoints
    SumC=gp.quicksum( y[j,i,k]*y[i,z,k]*Ad[j,i]*Ad[i,z]*np.abs(Ec[j,i,z]) for i in range(R) for j in range(R)  for z in range(R) for k in range(N))
    # the sum of the curvature of the relay points
    #SumCR= gp.quicksum(y[z, j, k] *y[j, r, k+1] * Ad[z, j] * Ad[j, r] * (1 + (s[k+1] - s[k]) * delta) for i in range(R) for j in range(R) for z in range(R) for r in range(R) for k in range(N-1))
    # add the equality constraints of auxiliary_v to avoid direct multiplication of the variables
    # for j in range(R):
    #     for z in range(R):
    #         for r in range(R):
    #             for k in range(N-1):
    #                 if Ad[j,z]==0 or Ad[z,r]==0:
    #                     model.addConstr(auxiliary_v[j,z,r,k]==0)
    #                 else:
    #                     model.addConstr(auxiliary_v[j, z, r, k] == y[j, z, k] * y[z, r, k+1])
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

    # SumCR=gp.quicksum(auxiliary_v[j,z,r,k]*(Ec[j,z,r] + (s[k+1] - s[k]) * delta)  for j in range(R) for z in range(R) for r in range(R) for k in range(N-1))
    SumCR=gp.quicksum(auxiliary_v[j,z,r,k]*(b[k]*np.abs(Cl[j,z,r])+(1-b[k])*np.abs(Cr[j,z,r]))  for j in range(R) for z in range(R) for r in range(R) for k in range(N-1))
    # Get the absolute value of SumCR
    # SumCRAbs=gp.abs_(SumCR)
    model.setObjective(al * SumL + ac * SumC + ac * SumCR, Obs)
    
    # Add constraint
    
    # Unique kth arc
    for k in range(N):
        model.addConstr(gp.quicksum(x[i,j,k] for i in range(R) for j in range(R)) == 1, "UniqueArc")
        model.addConstr(gp.quicksum(y[i,j,k]*Ad[i,j]*El[i,j] for i in range(R) for j in range(R)) >=arc_range[0], "NumberLowB_y")
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
    Waypoints=[]
    RelayPoints=[]
    # Print the results
    if model.status == GRB.OPTIMAL:
        # print("Selected items:")
        # for i in range(R):
        #     for j in range(R):
        #         for k in range(N):
        #             if y[i, j, k].X == 1:
        #                 print(f"Waypoint {i} to {j} in arc {k}")
        #                 Waypoints.append([i,j])
        #             if x[i, j, k].X == 1:
        #                 print(f"Relay point {i} to Relay point {j} of arc {k}")
        #                 RelayPoints.append([i,j])
        print(f"Total value: {model.objVal}")
        return model.objVal
    else:
        print("No optimal solution found.")
        return None
    

# N=7
# file_path=f"Normalization{N}_large.json"
# Matrix_file_path="Estimated_matrices_large.npz"
def get_normalization_prameters(N,arc_range,file_path,Matrix_file_path):
    Length_max=single_objective_solve(N,1,0,Obs=GRB.MAXIMIZE,arc_range=arc_range,Matrix_file_path=Matrix_file_path)
    if Length_max==None:
        return False
    Length_min=single_objective_solve(N,1,0,Obs=GRB.MINIMIZE,arc_range=arc_range,Matrix_file_path=Matrix_file_path)
    if Length_min==None:
        return False
    Curvature_max=single_objective_solve(N,0,1,Obs=GRB.MAXIMIZE,arc_range=arc_range,Matrix_file_path=  Matrix_file_path)
    if Curvature_max==None:
        return False
    Curvature_min=single_objective_solve(N,0,1,Obs=GRB.MINIMIZE,arc_range=arc_range,Matrix_file_path=  Matrix_file_path)
    if Curvature_min==None:
        return False
    theta_l=1/(Length_max-Length_min)
    theta_c=1/(Curvature_max-Curvature_min)
    data = {
            'Length_max': Length_max,
            'Length_min': Length_min,
            'Curvature_max': Curvature_max,
            'Curvature_min': Curvature_min,
            'theta_l': theta_l,
            'theta_c': theta_c,
                }
    with open(file_path, 'w') as file:
                json.dump(data, file)
    return True

