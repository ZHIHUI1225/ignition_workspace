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

# Add coordinate transformation functions
sys.path.append('/root/workspace/src/Replanning/scripts')
from coordinate_transform import get_frame_info
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

# Get coordinate frame information
try:
    frame_info = get_frame_info()
    print("\n=== Coordinate Frame Information ===")
    for frame_name, frame_data in frame_info.items():
        print(f"{frame_data['name']}: Units={frame_data['units']}")
    print("✓ All path planning calculations will be performed in world_pixel coordinates")
except Exception as e:
    print(f"⚠️  Warning: Could not load coordinate frame info: {e}")

# Generate file paths using configuration
file_path = config.get_full_path(config.file_path, use_data_path=True)
environment_file = config.get_full_path(config.environment_file, use_data_path=True)

print(f"\n=== File Paths ===")
print(f"Graph file: {file_path}")
print(f"Environment file: {environment_file}")

# Load reeb graph with error handling
try:
    reeb_graph = load_reeb_graph_from_file(file_path)
    print(f"✓ Successfully loaded reeb graph with {len(reeb_graph.nodes)} nodes")
except Exception as e:
    print(f"❌ Error loading reeb graph: {e}")
    sys.exit(1)

# Calculate distances and parameters
NumNodes = len(reeb_graph.nodes)
Start_node = reeb_graph.nodes[NumNodes-2].configuration
End_node = reeb_graph.nodes[NumNodes-1].configuration
Distances = np.linalg.norm(End_node-Start_node)
X_distance = abs(End_node[0]-Start_node[0])
Y_distance = abs(End_node[1]-Start_node[1])
Arc_min = arc_range[0]  # Use config arc range
Arc_max = arc_range[1]  # Use config arc range

print(f"\n=== Path Planning Parameters ===")
print(f"Number of nodes in graph: {NumNodes}")
print(f"Start node: {Start_node} (world_pixel)")
print(f"End node: {End_node} (world_pixel)")
print(f"Total distance: {Distances:.2f} pixels")
print(f"X distance: {X_distance:.2f} pixels")
print(f"Y distance: {Y_distance:.2f} pixels")
print(f"Arc range: [{Arc_min}, {Arc_max}] pixels")

# Validate coordinates are reasonable
if Distances < 10 or Distances > 2000:
    print(f"⚠️  Warning: Unusual total distance: {Distances:.2f} pixels")
if X_distance > 1200 or Y_distance > 700:
    print(f"⚠️  Warning: Distance exceeds expected image bounds")

# Convert to meters for physical validation
pixel_to_meter = config.pixel_to_meter_scale
distance_meters = Distances * pixel_to_meter
print(f"Total distance in meters: {distance_meters:.3f}m")

if distance_meters > 5.0:  # More than 5 meters seems unreasonable
    print(f"⚠️  Warning: Very large distance in meters: {distance_meters:.3f}m")

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
    print(f"✓ Found assignment result file: {assignment_result_file}")
    
    # Process waypoints
    try:
        GetWaypoints(assignment_result_file, config.get_full_path(f"Estimated_matrices_{case}.npz", use_data_path=True), waypoints_file_path)
        print(f"✓ Successfully processed waypoints")
    except Exception as e:
        print(f"❌ Error processing waypoints: {e}")
        sys.exit(1)
    
    # Get safe corridor
    try:
        safe_corridor, Distance, Angle, vertex = get_safe_corridor(reeb_graph, waypoints_file_path, environment_file)
        print(f"✓ Successfully computed safe corridor")
        print(f"   Distance segments: {len(Distance) if hasattr(Distance, '__len__') else 'scalar'}")
        print(f"   Angle segments: {len(Angle) if hasattr(Angle, '__len__') else 'scalar'}")
        print(f"   Vertices: {len(vertex) if hasattr(vertex, '__len__') else 'scalar'}")
    except Exception as e:
        print(f"❌ Error computing safe corridor: {e}")
        sys.exit(1)
    
    # get_normalization_prams(waypoints_file_path, Normalization_path, reeb_graph, Initial_Guess_file_path)
    # Initial_Guess(reeb_graph, phi0, waypoints_file_path, environment_file, safe_corridor, Normalization_path, Initial_Guess_file_path, Initial_Guess_figure_file)
    # get_normalization_prams(waypoints_file_path=waypoints_file_path, Normalization_path=Normalization_path, reeb_graph=reeb_graph, Initial_Guess_file_path=Initial_Guess_file_path)
    
    # Use GA planning result as initial guess for further optimization
    # config.Result_file = f"Optimization_GA_{N}_IG_norma{case}.json" - the output from GA_planning.py
    GA_result_file_path = config.get_full_path(config.Result_file, use_data_path=True)  # This is the GA result file
    
    print(f"\n=== Optimization Setup ===")
    print(f"Checking for GA result file: {GA_result_file_path}")
    
    if os.path.exists(GA_result_file_path):
        print(f"✅ Using GA result as initial guess: {GA_result_file_path}")
        
        # Load and validate GA result
        try:
            with open(GA_result_file_path, 'r') as f:
                ga_data = json.load(f)
            
            # Check if GA result contains coordinate frame information
            ga_coordinate_frame = ga_data.get('coordinate_frame', 'unknown')
            print(f"GA result coordinate frame: {ga_coordinate_frame}")
            
            if ga_coordinate_frame not in ['World Pixel Frame', 'world_pixel']:
                print(f"⚠️  Warning: GA result may not be in world_pixel frame!")
            else:
                print(f"✓ GA result is in correct coordinate frame")
            
            # Validate GA result structure
            if 'Initial_guess_phi' in ga_data:
                phi_values = ga_data['Initial_guess_phi']
                print(f"GA result contains {len(phi_values)} phi values")
                print(f"Phi range: [{min(phi_values):.4f}, {max(phi_values):.4f}]")
                
                # Check for extreme values that might cause solver issues
                extreme_phi = [p for p in phi_values if abs(p) > 10]
                if extreme_phi:
                    print(f"⚠️  Warning: Found extreme phi values: {extreme_phi}")
            else:
                print(f"⚠️  Warning: GA result does not contain 'Initial_guess_phi' key")
                print(f"Available keys: {list(ga_data.keys())}")
            
        except Exception as e:
            print(f"❌ Error validating GA result: {e}")
        
        # Run optimization with error handling
        try:
            print(f"\n=== Starting Optimization ===")
            print(f"Solver: Ipopt (Interior Point Optimizer)")
            print(f"Coordinate frame: world_pixel")
            
            result_file_path = config.get_full_path(f"Optimization_withSC_path{N}{case}.json", use_data_path=True)
            figure_file_path = config.get_full_path(f"Optimization_winthSC_path{N}{case}.png", use_data_path=True)
            
            Planning_error_withinSC(
                waypoints_file_path, 
                Normalization_path, 
                environment_file, 
                safe_corridor, 
                reeb_graph, 
                phi0, 
                GA_result_file_path, 
                Result_file=result_file_path, 
                figure_file=figure_file_path
            )
            
            # Add coordinate frame information to result
            try:
                if os.path.exists(result_file_path):
                    with open(result_file_path, 'r') as f:
                        result_data = json.load(f)
                    
                    # Add coordinate frame information
                    result_data['coordinate_frames'] = frame_info
                    result_data['coordinate_frame'] = 'world_pixel'
                    result_data['data_coordinate_frame'] = 'world_pixel'
                    
                    with open(result_file_path, 'w') as f:
                        json.dump(result_data, f, indent=4)
                    
                    print(f"✓ Added coordinate frame information to result file")
            except Exception as e:
                print(f"⚠️  Warning: Could not add coordinate frame info to result: {e}")
                
        except Exception as e:
            print(f"❌ Optimization failed with error: {e}")
            print(f"\n=== Possible Causes of Solver Failure ===")
            print(f"1. Infeasible constraints: Safe corridor too narrow")
            print(f"2. Poor initial guess: GA result may have extreme values")
            print(f"3. Numerical issues: Very large or small coordinate values")
            print(f"4. Solver limits: Maximum iterations or time exceeded")
            print(f"5. Memory issues: Problem too large for available memory")
            print(f"\n=== Debugging Suggestions ===")
            print(f"- Check safe corridor dimensions")
            print(f"- Validate GA initial guess values")
            print(f"- Try different solver parameters")
            print(f"- Reduce problem complexity")
            sys.exit(1)
            
    else:
        print(f"❌ GA result file not found: {GA_result_file_path}")
        print(f"Using basic initial guess instead: {Initial_Guess_file_path}")
        
        try:
            Planning_error_withinSC(
                waypoints_file_path, 
                Normalization_path, 
                environment_file, 
                safe_corridor, 
                reeb_graph, 
                phi0, 
                Initial_Guess_file_path, 
                Result_file=config.get_full_path(f"Optimization_withSC_path{N}{case}.json", use_data_path=True), 
                figure_file=config.get_full_path(f"Optimization_winthSC_path{N}{case}.png", use_data_path=True)
            )
        except Exception as e:
            print(f"❌ Optimization with basic initial guess failed: {e}")
            sys.exit(1)
    
    # Planning_normalization(waypoints_file_path,Normalization_path,environment_file,safe_corridor,reeb_graph,phi0,GA_result_file_path,Result_file=f"Optimization_normalized_path{N}"+case+".json",figure_file=f"Optimization_normalized_path{N}"+case+".png")
else:
    print(f"❌ Assignment result file not found: {assignment_result_file}")
    print(f"Please run the assignment optimization first to generate this file.")

