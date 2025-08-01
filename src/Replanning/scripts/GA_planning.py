# using the metaheuristic Genetic Algorithm to solve the planning problem
from GenerateMatrix import load_reeb_graph_from_file
from Planing_functions import load_WayPointFlag_from_file,get_safe_corridor
import pygad
import numpy as np
import pandas as pd
import json
from sko.GA import GA
import matplotlib.pyplot as plt
from Environment import Environment
import sko
import time
import sys

# Add config path to sys.path and load configuration
sys.path.append('/root/workspace/config')
from config_loader import config

# Add coordinate frame import
sys.path.append('/root/workspace/src/Replanning/scripts')
from coordinate_transform import get_frame_info

# Load configuration parameters
case = config.case
N = config.N
r_lim = config.r_lim / 0.0023  # Convert from meters to pixels
arc_range = config.arc_range
phi0 = config.phi0  # Load phi0 from config

print(f"Configuration loaded:")
print(f"  Case: {case}")
print(f"  Number of robots (N): {N}")
print(f"  Radius limit: {r_lim:.2f} pixels (converted from {config.r_lim} meters)")
print(f"  Arc range: {arc_range}")
print(f"  Initial angle (phi0): {phi0:.4f} ({phi0/np.pi:.2f}π)")

# Generate file paths using configuration
file_path = config.file_path
environment_file = config.environment_file
assignment_result_file = config.assignment_result_file
waypoints_file_path = config.waypoints_file_path
Normalization_path = config.Normalization_planning_path  # Use planning-specific normalization

# Load graph and waypoints
reeb_graph = load_reeb_graph_from_file(config.get_full_path(file_path, use_data_path=True))
Waypoints, Flags, Flagb = load_WayPointFlag_from_file(config.get_full_path(waypoints_file_path, use_data_path=True))
Nw=len(Waypoints)

safe_corridor,Distance,Angle,vertex=get_safe_corridor(reeb_graph,config.get_full_path(waypoints_file_path, use_data_path=True),config.get_full_path(environment_file, use_data_path=True))
## solution phi Nw
phi_new=np.zeros(Nw-1)
sigma=np.zeros(Nw-1)
l=np.zeros(Nw-1)
r0=np.zeros(Nw-1)
x=np.zeros(Nw-1)
y=np.zeros(Nw-1)
y_new=np.zeros(Nw-1)
init_range_low=-np.pi
init_range_high=np.pi

def fitness_func(ga_instance, solution, solution_idx):
    penalty = 0
    L = 0
    
    # Reconstruct full solution with fixed phi0
    full_solution = np.zeros(Nw)
    full_solution[0] = phi0  # Fixed first angle
    full_solution[1:] = solution  # GA optimizes the remaining angles
    
    with open(config.get_full_path(Normalization_path, use_data_path=True), 'r') as file:
        data = json.load(file)
    al=data['al']
    ac=data['ac']
    length_min=data['length_min']
    
    for i in range(Nw-1):
        phi_new[i]=full_solution[i]+Flagb[i]*np.pi/2
        sigma[i]=-1+np.cos(phi_new[i]-full_solution[i+1])
        
        # Handle sigma near zero more robustly
        if abs(sigma[i]) < 1e-6:
            r0[i] = 100000000
            l[i] = Distance[i][0]
        else:
            l[i]=Distance[i][0]*(np.cos(Angle[i][0]-phi_new[i])-np.cos(Angle[i][0]-full_solution[i+1]))/sigma[i]
            r0[i]=-Distance[i][0]*np.sin(Angle[i][0]-full_solution[i+1])/sigma[i]
            
        length = abs((full_solution[i+1]- full_solution[i]-Flagb[i]*np.pi/2)*r0[i])+l[i]
        curvature = abs(Distance[i][0]*np.sin(Angle[i][0]-full_solution[i+1]))
        
        # EXTREMELY heavy penalties for hard constraints - make them infeasible
        cos_constraint = np.cos((phi_new[i]-full_solution[i+1])/4)
        if cos_constraint < 0:
            penalty += 1000000  # Massive penalty
            
        dir_constraint1 = (phi_new[i]-Angle[i][0])*(phi_new[i]+full_solution[i+1]-2*Angle[i][0])
        if dir_constraint1 < 0:
            penalty += 500000  # Very large penalty
            
        dir_constraint2 = (phi_new[i]-Angle[i][0])*(full_solution[i+1]-Angle[i][0])
        if dir_constraint2 > 0:
            penalty += 500000  # Very large penalty
            
        # Radius constraint - absolutely critical
        if abs(r0[i]) < r_lim:
            penalty += 1000000  # Massive penalty
            
        # Length constraints - absolutely critical
        if l[i] < 0:
            penalty += 1000000  # Massive penalty
            
        if l[i] > np.pi*Distance[i][0]/2:
            penalty += 1000000  # Massive penalty
         
        # Safe corridor constraints - very important
        theta_start = phi_new[i]+np.pi/2
        theta_end = full_solution[i+1]+np.pi/2
        center=(r0[i]*np.cos(theta_start),r0[i]*np.sin(theta_start))
        
        for k in range(3):
            theta=phi_new[i]+np.pi/2+(full_solution[i+1]-phi_new[i])/3*(k+1)    
            x_k = r0[i] * np.cos(theta)-center[0]
            y_k = r0[i] * np.sin(theta)-center[1]
            x_new=x_k*np.cos(Angle[i][0])+y_k*np.sin(Angle[i][0])
            y_new = -x_k*np.sin(Angle[i][0])+y_k*np.cos(Angle[i][0])
            
            # Safe corridor violations
            if y_new < safe_corridor[i][1]:
                penalty += 100000  # Large penalty
            if y_new > safe_corridor[i][2]:
                penalty += 100000  # Large penalty
                
            # Distance bounds
            if x_new < 0:
                penalty += 100000  # Large penalty
            if x_new > Distance[i][0]:
                penalty += 100000  # Large penalty
                
        L += al[i]*(length-length_min[i]) + ac*(curvature)
    
    return -(L + penalty)  # Return negative because GA maximizes
def constraint_check(solution):
    violations = 0
    violation_details = []
    
    # Reconstruct full solution with fixed phi0
    full_solution = np.zeros(Nw)
    full_solution[0] = phi0  # Fixed first angle
    full_solution[1:] = solution  # GA optimizes the remaining angles
    
    for i in range(Nw-1):
        phi_new[i] = full_solution[i] + Flagb[i]*np.pi/2
        sigma[i] = -1 + np.cos(phi_new[i] - full_solution[i+1])
        
        # Handle sigma near zero more robustly
        if abs(sigma[i]) < 1e-6:
            r0[i] = 100000000
            l[i] = Distance[i][0]
        else:
            l[i] = Distance[i][0]*(np.cos(Angle[i][0]-phi_new[i])-np.cos(Angle[i][0]-full_solution[i+1]))/sigma[i]
            r0[i] = -Distance[i][0]*np.sin(Angle[i][0]-full_solution[i+1])/sigma[i]
        
        print(f'Segment {i}: l={l[i]:.3f}, r0={r0[i]:.3f}')
        
        # Check angle constraint
        cos_check = np.cos((phi_new[i]-full_solution[i+1])/4)
        if cos_check < 0:
            violations += 1
            violation_details.append(f'Segment {i}: Angle constraint violated (cos={cos_check:.3f})')
        
        # Check direction constraints  
        dir_check1 = (phi_new[i]-Angle[i][0])*(phi_new[i]+full_solution[i+1]-2*Angle[i][0])
        if dir_check1 < 0:
            violations += 1
            violation_details.append(f'Segment {i}: Direction constraint 1 violated ({dir_check1:.3f})')
            
        dir_check2 = (phi_new[i]-Angle[i][0])*(full_solution[i+1]-Angle[i][0])
        if dir_check2 > 0:
            violations += 1
            violation_details.append(f'Segment {i}: Direction constraint 2 violated ({dir_check2:.3f})')
        
        # Check radius constraint
        if abs(r0[i]) < r_lim:
            violations += 1
            violation_details.append(f'Segment {i}: Radius too small (|r0|={abs(r0[i]):.1f} < {r_lim})')
        
        # Check length constraints
        if l[i] < 0:
            violations += 1
            violation_details.append(f'Segment {i}: Negative length (l={l[i]:.3f})')
            
        max_length = np.pi*Distance[i][0]/2
        if l[i] > max_length:
            violations += 1
            violation_details.append(f'Segment {i}: Length too long (l={l[i]:.3f} > {max_length:.3f})')
        
        # Check safe corridor constraints
        theta_start = phi_new[i] + np.pi/2
        theta_end = full_solution[i+1] + np.pi/2
        center = (r0[i]*np.cos(theta_start), r0[i]*np.sin(theta_start))
        
        for k in range(3):
            theta = phi_new[i] + np.pi/2 + (full_solution[i+1]-phi_new[i])/3*(k+1)
            x_k = r0[i] * np.cos(theta) - center[0]
            y_k = r0[i] * np.sin(theta) - center[1]
            x_new = x_k*np.cos(Angle[i][0]) + y_k*np.sin(Angle[i][0])
            y_new = -x_k*np.sin(Angle[i][0]) + y_k*np.cos(Angle[i][0])
            
            if y_new < safe_corridor[i][1]:
                violations += 1
                violation_details.append(f'Segment {i}, point {k}: Below corridor (y={y_new:.3f} < {safe_corridor[i][1]:.3f})')
                
            if y_new > safe_corridor[i][2]:
                violations += 1
                violation_details.append(f'Segment {i}, point {k}: Above corridor (y={y_new:.3f} > {safe_corridor[i][2]:.3f})')
                
            if x_new < 0:
                violations += 1
                violation_details.append(f'Segment {i}, point {k}: Negative x (x={x_new:.3f})')
                
            if x_new > Distance[i][0]:
                violations += 1
                violation_details.append(f'Segment {i}, point {k}: Exceeds distance (x={x_new:.3f} > {Distance[i][0]:.3f})')
    
    if violations > 0:
        print(f'\nTotal violations: {violations}')
        for detail in violation_details:
            print(f'  - {detail}')
    else:
        print('No constraint violations found!')
        
    return violations
lb = (-np.pi * np.ones(Nw)).tolist()
ub = (np.pi * np.ones(Nw)).tolist()

# Fix phi0 value exactly (same as Ipopt solver)
lb[0] = phi0
ub[0] = phi0
# Improved bounds calculation for intermediate waypoints
for i in range(1, Nw-2):
    if i < len(Angle) and i-1 < len(Angle):
        # Base bounds on angle constraints
        base_angle = Angle[i][0] if isinstance(Angle[i], list) else Angle[i]
        prev_angle = Angle[i-1][0] if isinstance(Angle[i-1], list) else Angle[i-1]
        
        # Conservative bounds to avoid violations
        bound1 = max(base_angle - np.pi/3 - Flagb[i]*np.pi/2, prev_angle - np.pi/3)
        bound2 = min(base_angle + np.pi/3 - Flagb[i]*np.pi/2, prev_angle + np.pi/3)
        
        lb[i] = max(-np.pi, min(bound1, bound2))
        ub[i] = min(np.pi, max(bound1, bound2))
    else:
        # Fallback bounds
        lb[i] = -np.pi/2
        ub[i] = np.pi/2

# Last waypoint bounds
if Nw-2 < len(Angle):
    last_angle = Angle[Nw-2][0] if isinstance(Angle[Nw-2], list) else Angle[Nw-2]
    lb[Nw-1] = max(-np.pi, last_angle - np.pi/3)
    ub[Nw-1] = min(np.pi, last_angle + np.pi/3)
else:
    lb[Nw-1] = -np.pi/2
    ub[Nw-1] = np.pi/2

gene_space = [{'low': lb[i], 'high': ub[i]} for i in range(1, Nw)]  # Skip index 0 (phi0)
# ga = GA(func=fitness_func, n_dim=Nw, size_pop=80, max_iter=2000, prob_mut=0.2,lb=lb,ub=ub, precision=1e-1)
# best_x, best_y = ga.run()
# print('best_x:', best_x, '\n', 'best_y:', best_y)
# Y_history = pd.DataFrame(ga.all_history_Y)
# fig, ax = plt.subplots(2, 1)istory.values, '.', color='red')
# Y_history.min(axis=1).cummin().plot(kind='line')
# plt.show()
# Get GA parameters from configuration
num_generations = config.num_generations
num_parents_mating = config.num_parents_mating
sol_per_pop = config.sol_per_pop
mutation_probability = config.mutation_probability
crossover_probability = config.crossover_probability
keep_elitism = config.keep_elitism
parent_selection_type = config.parent_selection_type
crossover_type = config.crossover_type
mutation_type = config.mutation_type

print(f"GA Parameters: {num_generations} generations, {sol_per_pop} population, {num_parents_mating} parents")

num_genes = Nw

last_fitness = 0
# Generate an initial population

# Improved initial population generation with much better feasibility
def create_feasible_individual():
    individual = np.zeros(Nw)
    
    # Start point - use config phi0
    individual[0] = phi0
    
    # Try to create a more feasible path with tighter constraints
    for i in range(1, Nw):
        if i < Nw-1 and i < len(Angle):
            # Use much more conservative angle-based initialization
            try:
                base_angle = float(Angle[i-1][0] if isinstance(Angle[i-1], (list, np.ndarray)) else Angle[i-1])
                # Much smaller random variation
                candidate = base_angle + np.random.uniform(-np.pi/8, np.pi/8)  # Reduced from pi/4
            except (IndexError, TypeError):
                candidate = np.random.uniform(-np.pi/4, np.pi/4)  # More conservative
        else:
            # Last point - very conservative
            try:
                if i >= 2:
                    prev_angle = float(Angle[i-2][0] if isinstance(Angle[i-2], (list, np.ndarray)) else Angle[i-2])
                    candidate = prev_angle + np.random.uniform(-np.pi/8, np.pi/8)
                else:
                    candidate = np.random.uniform(-np.pi/4, np.pi/4)
            except (IndexError, TypeError):
                candidate = np.random.uniform(-np.pi/4, np.pi/4)
        
        # Much tighter bounds clamping
        candidate = max(float(lb[i]), min(float(ub[i]), candidate))
        individual[i] = candidate
    
    return individual.tolist()

# Create multiple diverse initial populations for reduced dimension space
initial_population = []

# 60% constraint-aware individuals (excluding fixed phi0)
for _ in range(int(0.6 * sol_per_pop)):
    individual = create_feasible_individual()
    # Remove phi0 from the individual for reduced dimension optimization
    individual_reduced = individual[1:]  # Skip first element (phi0)
    initial_population.append(individual_reduced)

# 30% angle-based individuals (more direct angle initialization, excluding phi0)
for _ in range(int(0.3 * sol_per_pop)):
    individual = []
    for i in range(1, Nw):  # Start from index 1 (skip phi0)
        if i < len(Angle):
            try:
                base = float(Angle[i-1][0] if isinstance(Angle[i-1], (list, np.ndarray)) else Angle[i-1])
                val = np.clip(base + np.random.normal(0, np.pi/12), float(lb[i]), float(ub[i]))
                individual.append(float(val))
            except:
                individual.append(float(np.random.uniform(lb[i], ub[i])))
        else:
            individual.append(float(np.random.uniform(lb[i], ub[i])))
    initial_population.append(individual)

# 10% completely random but bounded (excluding phi0)
for _ in range(sol_per_pop - len(initial_population)):
    individual = []
    for i in range(1, Nw):  # Start from index 1 (skip phi0)
        individual.append(float(np.random.uniform(lb[i], ub[i])))
    initial_population.append(individual)

# Ensure all individuals have the correct reduced length (Nw-1)
expected_length = Nw-1
initial_population = [ind[:expected_length] if len(ind) > expected_length else ind + [0.0]*(expected_length-len(ind)) for ind in initial_population]
initial_population = np.array(initial_population, dtype=float)


def on_generation(ga_instance):
    """Monitor progress and check for feasible solutions"""
    generation = ga_instance.generations_completed
    solution, fitness, _ = ga_instance.best_solution(ga_instance.last_generation_fitness)
    
    # Print progress every 100 generations
    if generation % 100 == 0:
        print(f"Generation {generation}: Best fitness = {fitness:.2f}")
        
        # Reconstruct full solution with fixed phi0 for constraint checking
        full_solution = np.zeros(Nw)
        full_solution[0] = phi0  # Fixed first angle
        full_solution[1:] = solution  # GA optimizes the remaining angles
        
        # Quick feasibility check for current best solution
        violations = 0
        for i in range(Nw-1):
            phi_new_i = full_solution[i] + Flagb[i]*np.pi/2
            sigma_i = -1 + np.cos(phi_new_i - full_solution[i+1])
            
            if abs(sigma_i) < 1e-6:
                r0_i = 100000000
                l_i = Distance[i][0]
            else:
                l_i = Distance[i][0]*(np.cos(Angle[i][0]-phi_new_i)-np.cos(Angle[i][0]-full_solution[i+1]))/sigma_i
                r0_i = -Distance[i][0]*np.sin(Angle[i][0]-full_solution[i+1])/sigma_i
            
            # Check critical constraints
            if np.cos((phi_new_i-full_solution[i+1])/4) < 0:
                violations += 1
            if abs(r0_i) < r_lim:
                violations += 1
            if l_i < 0:
                violations += 1
            if l_i > np.pi*Distance[i][0]/2:
                violations += 1
        
        print(f"  Current violations: {violations}")
        
        # Early stopping if we find a feasible solution
        if violations == 0:
            print(f"FEASIBLE SOLUTION FOUND at generation {generation}!")
            print("Stopping early...")
            return "stop"

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       mutation_probability=mutation_probability,
                       crossover_probability=crossover_probability,
                       sol_per_pop=sol_per_pop,
                       num_genes=Nw-1,  # Optimize only Nw-1 variables
                       initial_population=initial_population,
                       gene_space=gene_space,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_elitism=keep_elitism,
                       on_generation=on_generation)

# # Running the GA to optimize the parameters of the function.
# Running the GA to optimize the parameters of the function.
# Record the start time
start_time = time.time()

# Run the GA
ga_instance.run()

# Record the end time
end_time = time.time()

ga_instance.plot_fitness()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Parameters of the best solution (Nw-1 variables): {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# Reconstruct full solution with fixed phi0 for output
full_solution = np.zeros(Nw)
full_solution[0] = phi0
full_solution[1:] = solution
print(f"Full solution with fixed phi0: {full_solution}")

Result_file = config.get_full_path(config.Result_file, use_data_path=True)
figure_file = config.get_full_path(config.figure_file, use_data_path=True)
# solution=best_x
violations=constraint_check(solution)
print('violations:',violations)
fig, ax = plt.subplots()

# Load environment to get boundary information
environment = Environment.load_from_file(config.get_full_path(environment_file, use_data_path=True))

# Draw environment boundary from coord_bounds
coord_bounds = environment.coord_bounds
x_min, x_max, y_min, y_max = coord_bounds

# Draw boundary walls as thick black lines
boundary_width = 3
ax.plot([x_min, x_max], [y_min, y_min], 'k-', linewidth=boundary_width, label='Environment Boundary')  # Bottom wall
ax.plot([x_min, x_max], [y_max, y_max], 'k-', linewidth=boundary_width)  # Top wall
ax.plot([x_min, x_min], [y_min, y_max], 'k-', linewidth=boundary_width)  # Left wall
ax.plot([x_max, x_max], [y_min, y_max], 'k-', linewidth=boundary_width)  # Right wall

# Draw environment obstacles
environment.draw("black")

# Draw safe corridor boundaries
for i, corridor_vertex in enumerate(vertex):
    if corridor_vertex is not None and len(corridor_vertex) > 0:
        corridor_x = corridor_vertex[:, 0]
        corridor_y = corridor_vertex[:, 1]
        ax.plot(corridor_x, corridor_y, 'g--', linewidth=2, alpha=0.7, 
               label='Safe Corridor' if i == 0 else "")

l_Opt=np.zeros(Nw-1)
r_Opt=np.zeros(Nw-1)
for i in range(Nw-1):
    Ps=reeb_graph.nodes[Waypoints[i]].configuration
    if Flagb[i]!=0:
        ax.plot(Ps[0], Ps[1], 'ro', label='relay point')
    else:
        plt.plot(Ps[0], Ps[1], 'go',label='waypoint')
    phi_new_opt=full_solution[i]+Flagb[i]*np.pi/2 #A
    thetaC_opt=full_solution[i+1]/2 + phi_new_opt/2
    sigma_opt=-1+np.cos(phi_new_opt-full_solution[i+1])
    if sigma_opt==0:
        r0_opt=100000000
        l_opt=Distance[i][0]
        x_line=np.linspace(0,l_opt*np.cos(full_solution[i+1]),100)
        y_line=np.linspace(0,l_opt*np.sin(full_solution[i+1]),100)
        
        x_line=x_line+Ps[0]
        y_line=y_line+Ps[1]
        ax.plot(x_line, y_line,'b')
    else:
        l_opt=Distance[i]*(np.cos(Angle[i]-phi_new_opt)-np.cos(Angle[i]-full_solution[i+1]))/sigma_opt
        r0_opt=-Distance[i]*np.sin(Angle[i]-full_solution[i+1])/sigma_opt
#     # Plot the arc

        theta_start =phi_new_opt+np.pi/2
        theta_end = full_solution[i+1]+np.pi/2 # Assuming the arc spans the angle phi_A
        center=(r0_opt*np.cos(theta_start),r0_opt*np.sin(theta_start))
        theta = np.linspace(theta_start, theta_end, 100)
        
        x = r0_opt * np.cos(theta)-center[0]
        y = r0_opt * np.sin(theta)-center[1]
        
        x=x+Ps[0]
        y=y+Ps[1]
        ax.plot(x, y,'b')
    # plot line 
        x_line=np.linspace(r0_opt * np.cos(theta_end)-center[0],r0_opt * np.cos(theta_end)-center[0]+l_opt*np.cos(full_solution[i+1]),100)
        y_line=np.linspace(r0_opt * np.sin(theta_end)-center[1],r0_opt * np.sin(theta_end)-center[1]+l_opt*np.sin(full_solution[i+1]),100)
        
        x_line=x_line+Ps[0]
        y_line=y_line+Ps[1]
        l_Opt[i]=l_opt
        r_Opt[i]=r0_opt
        ax.plot(x_line, y_line,'b')

# Get coordinate frame information
frame_info = get_frame_info()
world_pixel_frame = frame_info['world_pixel_frame']

data = {
        'Initial_guess_phi': full_solution.tolist(),  # Save full solution with fixed phi0
        'Optimization_l': l_Opt.tolist(),
        'Optimization_r': r_Opt.tolist(),
        'Optimizationf': solution_fitness,
        'coordinate_frame': world_pixel_frame['name'],
        'units': world_pixel_frame['units']
        }
with open(Result_file, 'w') as file:
            json.dump(data, file)  
        # Get the solution     
loaded_environment = Environment.load_from_file(config.get_full_path(environment_file, use_data_path=True))
loaded_environment.draw("black")

# Plot safe corridor boundaries
for i in range(len(vertex)):
    if len(vertex[i]) > 0:
        # Plot corridor boundaries
        corridor_vertices = vertex[i]
        ax.plot(corridor_vertices[:, 0], corridor_vertices[:, 1], 'g--', alpha=0.5, linewidth=1, label='Safe Corridor' if i == 0 else "")
ax.set_aspect('equal')
# plt.autoscale()
plt.legend()
plt.savefig(figure_file)  
plt.show()

