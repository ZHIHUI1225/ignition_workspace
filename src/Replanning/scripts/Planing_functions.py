# path planning functions
## initial guess and planning
import numpy as np
import matplotlib.pyplot as plt
import json
from GenerateMatrix import load_reeb_graph_from_file
import casadi as ca
import casadi.tools as ca_tools
from Environment import Environment
from BarriersOriginal import generate_barriers_test
from IntegerProgramming import get_NormalizationMatrix
from shapely.geometry import LineString, Polygon

def check_collision(edge, polygons, coord_bounds=None):
    """
    Check if an edge collides with any polygon in the environment or goes outside workspace boundaries.
    
    Parameters:
    edge (tuple): A tuple of two nodes representing the edge.
    polygons (list): A list of polygons in the environment.
    coord_bounds (list): [x_min, x_max, y_min, y_max] workspace boundaries.
    
    Returns:
    bool: True if the edge collides with any polygon or boundary, False otherwise.
    """
    node1, node2 = edge
    edge_line = LineString([node1, node2])
    
    # Check boundary collision first if coord_bounds is provided
    if coord_bounds is not None:
        x_min, x_max, y_min, y_max = coord_bounds
        
        # Check if any point of the edge is outside workspace boundaries
        for point in [node1, node2]:
            if (point[0] < x_min or point[0] > x_max or 
                point[1] < y_min or point[1] > y_max):
                return True
        
        # Check if edge crosses boundary lines
        boundary_lines = [
            LineString([(x_min, y_min), (x_max, y_min)]),  # Bottom boundary
            LineString([(x_max, y_min), (x_max, y_max)]),  # Right boundary  
            LineString([(x_max, y_max), (x_min, y_max)]),  # Top boundary
            LineString([(x_min, y_max), (x_min, y_min)])   # Left boundary
        ]
        
        for boundary_line in boundary_lines:
            if edge_line.intersects(boundary_line):
                return True
    
    # Check polygon collisions
    for polygon in polygons:
        polygon_array = [polygon.vertices[i] for i in range(len(polygon.vertices))]
        polygon_shape = Polygon(polygon_array)
        if edge_line.intersects(polygon_shape):
            return True
    return False


def load_WayPointFlag_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Waypoints'], data['Flags'],data["FlagB"]

def load_matrices_from_file(file_path):
    data = np.load(file_path)
    Ec = data['Ec']
    El = data['El']
    Ad = data['Ad']
    return Ec, El, Ad

def load_initial_guess_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Initial_guess_phi']


# generate the safe corridor between two points
def get_safe_corridor(reeb_graph, waypoints_file_path, environment_file, step_size=0.5, max_distance=100):
    """
    Generate safe corridor with improved collision detection and robustness
    
    Parameters:
    - reeb_graph: The loaded graph
    - waypoints_file_path: Path to waypoints file
    - environment_file: Path to environment file
    - step_size: Step size for collision checking (default: 0.5)
    - max_distance: Maximum distance to check for corridor bounds (default: 100)
    
    Returns:
    - safe_corridor: List of corridor bounds [slope, y_min, y_max]
    - Distance: Distance between waypoints
    - Angle: Angle between waypoints  
    - vertex: Corridor boundary vertices for visualization
    """
    
    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
    loaded_environment = Environment.load_from_file(environment_file)
    
    # Get workspace boundaries from environment
    coord_bounds = loaded_environment.coord_bounds  # [x_min, x_max, y_min, y_max]
    
    # Add safety margin to boundaries to keep corridors well within workspace
    safety_margin = 5.0  # pixels
    if coord_bounds is not None:
        x_min, x_max, y_min, y_max = coord_bounds
        coord_bounds = [x_min + safety_margin, x_max - safety_margin, 
                       y_min + safety_margin, y_max - safety_margin]
    
    N = len(Waypoints)
    safe_corridor = []
    vertex = []
    
    # Calculate the distance and angle between waypoints
    Distance = np.zeros((N-1, 1))  # r_B in polar coordinate
    Angle = np.zeros((N-1, 1))    # theta_B in polar coordinate
    
    for i in range(N - 1):
        # Get waypoint positions
        start_pos = reeb_graph.nodes[Waypoints[i]].configuration
        end_pos = reeb_graph.nodes[Waypoints[i+1]].configuration
        
        # Calculate distance and angle
        Distance[i] = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        Angle[i] = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        
        # Handle vertical lines (improved tolerance)
        if abs(end_pos[0] - start_pos[0]) < 1e-6:
            # Vertical line case
            slope = 100000000  # Large number to represent infinity
            
            # Find lower bound (left side)
            db_min = 0
            collision_found = False
            
            while not collision_found and db_min < max_distance:
                db_min += step_size
                p_start_low = [start_pos[0] - db_min, start_pos[1]]
                p_end_low = [end_pos[0] - db_min, end_pos[1]]
                edge = [p_start_low, p_end_low]
                
                if check_collision(edge, loaded_environment.polygons, coord_bounds):
                    collision_found = True
                    break
            
            # Find upper bound (right side)
            db_max = 0
            collision_found = False
            
            while not collision_found and db_max < max_distance:
                db_max += step_size
                p_start_up = [start_pos[0] + db_max, start_pos[1]]
                p_end_up = [end_pos[0] + db_max, end_pos[1]]
                edge = [p_start_up, p_end_up]
                
                if check_collision(edge, loaded_environment.polygons, coord_bounds):
                    collision_found = True
                    break
            
            # For vertical lines, y_min and y_max are in x-direction relative to the line
            y_min = -db_min
            y_max = db_max
            
            # Create corridor boundary vertices
            p_A_low = [start_pos[0] - db_min, start_pos[1]]
            P_B_low = [end_pos[0] - db_min, end_pos[1]]
            P_B_up = [end_pos[0] + db_max, end_pos[1]]
            p_A_up = [start_pos[0] + db_max, start_pos[1]]
            
        else:
            # Non-vertical line case
            slope = (end_pos[1] - start_pos[1]) / (end_pos[0] - start_pos[0])
            
            # Find lower bound (offset in negative normal direction)
            db_min = 0
            collision_found = False
            
            while not collision_found and db_min < max_distance:
                db_min += step_size
                # Offset perpendicular to the line
                p_start_low = [start_pos[0] + db_min * slope, start_pos[1] - db_min]
                p_end_low = [end_pos[0] + db_min * slope, end_pos[1] - db_min]
                edge = [p_start_low, p_end_low]
                
                if check_collision(edge, loaded_environment.polygons, coord_bounds):
                    collision_found = True
                    break
            
            # Find upper bound (offset in positive normal direction)
            db_max = 0
            collision_found = False
            
            while not collision_found and db_max < max_distance:
                db_max += step_size
                # Offset perpendicular to the line
                p_start_up = [start_pos[0] - db_max * slope, start_pos[1] + db_max]
                p_end_up = [end_pos[0] - db_max * slope, end_pos[1] + db_max]
                edge = [p_start_up, p_end_up]
                
                if check_collision(edge, loaded_environment.polygons, coord_bounds):
                    collision_found = True
                    break
            
            # Calculate y_min and y_max in the local coordinate system
            P = [start_pos[0], start_pos[1]]
            p_A_low = [start_pos[0] + db_min * slope, start_pos[1] - db_min]
            p_A_up = [start_pos[0] - db_max * slope, start_pos[1] + db_max]
            P_B_low = [end_pos[0] + db_min * slope, end_pos[1] - db_min]
            P_B_up = [end_pos[0] - db_max * slope, end_pos[1] + db_max]
            
            y_min = -np.linalg.norm(np.array(P) - np.array(p_A_low))
            y_max = np.linalg.norm(np.array(P) - np.array(p_A_up))
        
        # Store corridor information
        safe_corridor.append([slope, min(y_min, y_max), max(y_min, y_max)])
        vertex.append(np.array([p_A_low, P_B_low, P_B_up, p_A_up, p_A_low]))
    
    return safe_corridor, Distance, Angle, vertex

def Initial_Guess(reeb_graph,phi0,waypoints_file_path,environment_file,safe_corridor,Normalization_path,Result_file,figure_file):

# load the waypoints and flags from a file
    # Load config for radius limit
    import sys
    sys.path.append('/root/workspace/config')
    from config_loader import config
    
    # Convert radius limit from meters to pixels
    r_min_pixels = config.meters_to_pixels(config.r_lim)
    
    # Convert minimum length from meters to pixels
    l_min_pixels = config.meters_to_pixels(config.min_length)
    
    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
    
    with open(Normalization_path, 'r') as file:
        data = json.load(file)
    al=data['al']
    ac=data['ac']
    length_min=data['length_min']
    curvature_min=data['curvature_min']
    # Number of variables
    N = len(Waypoints)
    sumf=0
    # calculate the distance and angle between waypoints
    Distance=np.zeros((N-1,1)) # r_B in polar coordinate
    Angle=np.zeros((N-1,1)) # theta_B in polar coordinate
    for i in range(N-1):
        Distance[i]=np.linalg.norm(reeb_graph.nodes[Waypoints[i+1]].configuration-reeb_graph.nodes[Waypoints[i]].configuration)
        Angle[i] = np.arctan2(reeb_graph.nodes[Waypoints[i+1]].configuration[1] - reeb_graph.nodes[Waypoints[i]].configuration[1],
                            reeb_graph.nodes[Waypoints[i+1]].configuration[0] - reeb_graph.nodes[Waypoints[i]].configuration[0])

    # delta=ca.MX.sym('delta',N-1) # angle of each straight line
    # 
    # ac=100
    phi_A=phi0
    Initial_guess_phi=np.zeros(N)
    Initial_guess_phi[0]=phi_A


    # plot the path
    fig, ax = plt.subplots()
    for i in range(N-1):
        # start point A and end point B
        A=reeb_graph.nodes[Waypoints[i]].configuration
        B=reeb_graph.nodes[Waypoints[i+1]].configuration
        # Create symbolic variables
        phi=ca.SX.sym('phi', 1) # the angel at each waypoints
        l=ca.SX.sym('l', 1) # the disatnce of AC
        r0=ca.SX.sym('a',1) # the radius h of each arc 

        phi_new=phi_A+Flagb[i]*np.pi/2 # A
        sigma=ca.if_else(ca.cos(phi_new-phi)!=1,-1+ca.cos(phi_new-phi),0.0001)
        l=ca.if_else(ca.cos(phi_new-phi)!=1,Distance[i]*(ca.cos(Angle[i]-phi_new)-ca.cos(Angle[i]-phi))/sigma,Distance[i])
        r0=-Distance[i]*ca.sin(Angle[i]-phi)/sigma
        # r0[i] = ca.if_else(ca.fabs(phi_new - phi[i+1]) ==0,
        #                    0,
        #                    ca.if_else(phi_new > phi[i+1],
        #                             rC[i]*ca.sin(thetaC[i]-phi_new+np.pi/2)/ ca.sin(phi_new - phi[i+1]),
        #                             rC[i]*ca.sin(thetaC[i]-phi_new-np.pi/2)/ ca.sin( phi[i+1]-phi_new)))


    # f = ca.Function('f', [phi], [thetaC,rC,r0], ['direction'], ['thetaC','rC','radius'])
        g = [] # constrains
        lbg = []
        ubg = []
        # if i<N-2:
        #     bound1=max(Angle[i][0]-np.pi/2-Flagb[i]*np.pi/2,Angle[i-1][0]-np.pi/2)
        #     bound2=min(Angle[i][0]+np.pi/2-Flagb[i]*np.pi/2,Angle[i-1][0]+np.pi/2)
        #     lb_phi=min(bound1,bound2)
        #     ub_phi=max(bound1,bound2)
        # else:
        #     lb_phi=Angle[N-2][0]-np.pi/2
        #     ub_phi=Angle[N-2][0]+np.pi/2
    # Define the bounds for the variables
        if phi_new>Angle[i]:
            lb_phi = 2*Angle[i]-phi_new
            ub_phi = Angle[i]
            g.append(phi_new-phi)
            lbg.append(0)
            ubg.append(2*np.pi)
        else:
            lb_phi = Angle[i]
            ub_phi = 2*Angle[i]-phi_new
            g.append(phi-phi_new)
            lbg.append(0)
            ubg.append(2*np.pi)
        center=(r0*ca.cos(phi_new+np.pi/2),r0*ca.sin(phi_new+np.pi/2))
        for k in range(2):
            theta=phi_new+ca.pi/2+(phi-phi_new)/2*(k+1)    
            x_k = r0 * ca.cos(theta)-center[0]
            y_k = r0 * ca.sin(theta)-center[1]
            g.append(-x_k*ca.sin(Angle[i][0])+y_k*ca.cos(Angle[i][0]))
            lbg.append(safe_corridor[i][1]+2)
            ubg.append(safe_corridor[i][2]-2)
    # # lb_phi[0]=0
    # ub_phi[0]=0


    # Define an example objective function (e.g., minimize the sum of squares)
       
        objective = al[i]*(ca.fabs((phi- phi_new)*r0)+l-length_min[i])+ac/ca.fabs(r0)
        # objective = objective + al[i]*(ca.fabs((phi[i+1]- phi[i]-Flagb[i]*np.pi/2)*r0[i])+l[i]-length_min[i])+ac*(ca.if_else(phi[i+1]!=phi_new[i],1/ca.fabs(r0[i]),0)-curvature_min)

        # Define an example constraint (e.g., sum of variables equals zero)
        g.append(ca.fabs(r0))
        lbg.append(r_min_pixels)  # Use config radius limit
        ubg.append(100000)  # Much larger radius maximum
        g.append(l)
        lbg.append(l_min_pixels)  # Use config minimum length
        ubg.append(Distance[i][0])  # Maximum length is the distance between waypoints

        # Create an NLP problem
        nlp = {'x': phi, 'f': objective, 'g': ca.vertcat(*g)}

        opts_setting = {'ipopt.max_iter':10000, 'ipopt.print_level':2, 'print_time':0, 'ipopt.acceptable_tol':1e-5, 'ipopt.acceptable_obj_change_tol':1e-5}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)
        # # Create an NLP solver
        # opts = {'ipopt.print_level': 0, 'print_time': 0}
        # solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        # Initial guess
        x0=np.random.uniform(lb_phi,ub_phi)
       
        # Solve the NLP problem
        sol = solver(x0=x0, lbx=lb_phi, ubx=ub_phi, lbg=lbg, ubg=ubg)
        x_opt = sol['x'].full().flatten()
        phi_opt = np.array(x_opt)

        print("Optimal result", sol['f'])
        if solver.stats()['success']:
            print("Solver succeeded!")
            sumf=sumf+sol['f']
            Ps=reeb_graph.nodes[Waypoints[i]].configuration
            if Flagb[i]!=0:
                ax.plot(Ps[0], Ps[1], 'ro', label='relay point')
            else:
                plt.plot(Ps[0], Ps[1], 'go',label='waypoint')
            phi_new=phi_A+Flagb[i]*np.pi/2 #A
            thetaC_opt=phi_opt/2 + phi_new/2
            sigma_opt=-1+ca.cos(phi_new-phi_opt)
            l_opt=Distance[i]*(ca.cos(Angle[i]-phi_new)-ca.cos(Angle[i]-phi_opt))/sigma_opt
            r0_opt=-Distance[i]*ca.sin(Angle[i]-phi_opt)/sigma_opt
        #     # Plot the arc
            theta_start =phi_new+np.pi/2
            theta_end = phi_opt+np.pi/2 # Assuming the arc spans the angle phi_A
            center=(r0_opt*np.cos(theta_start),r0_opt*np.sin(theta_start))
            theta = np.linspace(theta_start, theta_end, 100)
            x = r0_opt * np.cos(theta)-center[0]
            y = r0_opt * np.sin(theta)-center[1]
            # plot line 
            x_line=np.linspace(r0_opt * np.cos(theta_end)-center[0],r0_opt * np.cos(theta_end)-center[0]+l_opt*np.cos(phi_opt),100)
            y_line=np.linspace(r0_opt * np.sin(theta_end)-center[1],r0_opt * np.sin(theta_end)-center[1]+l_opt*np.sin(phi_opt),100)
            x=x+Ps[0]
            y=y+Ps[1]
            x_line=x_line+Ps[0]
            y_line=y_line+Ps[1]
            ax.plot(x, y,'b')
            ax.plot(x_line, y_line,'b')
            phi_A=phi_opt
            Initial_guess_phi[i+1]=phi_A

            # Get the solution     
        else:
            Ps=reeb_graph.nodes[Waypoints[i]].configuration
            if Flagb[i]!=0:
                ax.plot(Ps[0], Ps[1], 'ro', label='relay point')
            else:
                plt.plot(Ps[0], Ps[1], 'go',label='waypoint')
            print("Solver failed!")
            phi_A=0
    # environment.draw('black')

    # save the result phi_opt as Json file
    data = {
        'Initial_guess_phi': Initial_guess_phi.tolist(),
        'Intial_f': float(sumf)
    }
    with open(Result_file, 'w') as file:
        json.dump(data, file)
    print("the sum of the objective function:",sumf)
    loaded_environment = Environment.load_from_file(environment_file)
    loaded_environment.draw("black")
    ax.set_aspect('equal')  # Keep x and y with same scale
    plt.tight_layout()
    plt.savefig(figure_file)

def Planning_normalization(waypoints_file_path,Normalization_path,environment_file,safe_corridor,reeb_graph,phi0,Initial_Guess_file_path,Result_file,figure_file):

    # Load config for radius limit
    import sys
    sys.path.append('/root/workspace/config')
    from config_loader import config
    
    # Convert radius limit from meters to pixels
    r_min_pixels = config.meters_to_pixels(config.r_lim)

    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
    Initial_Guess=np.array(load_initial_guess_from_file(Initial_Guess_file_path))
    with open(Normalization_path, 'r') as file:
        data = json.load(file)
    al=data['al']
    ac=data['ac']
    length_min=data['length_min']
    curvature_min=data['curvature_min']
    # Number of variables
    N = len(Waypoints)
    # calculate the distance and angle between waypoints
    Distance=np.zeros((N-1,1)) # r_B in polar coordinate
    Angle=np.zeros((N-1,1)) # theta_B in polar coordinate
    for i in range(N-1):
        Distance[i]=np.linalg.norm(reeb_graph.nodes[Waypoints[i+1]].configuration-reeb_graph.nodes[Waypoints[i]].configuration)
        Angle[i] = np.arctan2(reeb_graph.nodes[Waypoints[i+1]].configuration[1] - reeb_graph.nodes[Waypoints[i]].configuration[1],
                            reeb_graph.nodes[Waypoints[i+1]].configuration[0] - reeb_graph.nodes[Waypoints[i]].configuration[0])

    # Create symbolic variables
    phi=ca.SX.sym('phi', N) # the angel at each waypoints
    sigma = ca.SX.sym('sigma', N-1) # the angle of each arc
    r0=ca.SX.sym('r', N-1) # the radius of each arc
    l=ca.SX.sym('l', N-1) # the length of each straight line
    phi_new=ca.SX.sym('phi_new', N-1) # the angle of each straight line
    for i in range(N-1):
        phi_new[i]=phi[i]+Flagb[i]*np.pi/2 # A
        sigma[i]=ca.if_else(ca.cos(phi_new[i]-phi[i+1])!=1,-1+ca.cos(phi_new[i]-phi[i+1]),0.0001)
        l[i]=ca.if_else(ca.cos(phi_new[i]-phi[i+1])!=1,Distance[i]*(ca.cos(Angle[i]-phi_new[i])-ca.cos(Angle[i]-phi[i+1]))/sigma[i],Distance[i])
        r0[i]=-Distance[i]*ca.sin(Angle[i]-phi[i+1])/sigma[i]
    # delta=ca.MX.sym('delta',N-1) # angle of each straight line
    # al=1
    # # ac=0.05438074639134307
    # ac=100
    # Define the bounds for the variables
    g = [] # constrains
    lbg = []
    ubg = []
    g.append(phi[0])
    lbg.append(phi0)
    ubg.append(phi0)
    for i in range(N-1):
        g.append(ca.cos((phi_new[i]-phi[i+1])/4))
        lbg.append(0)
        ubg.append(1)
        g.append((phi_new[i]-Angle[i])*(phi_new[i]+phi[i+1]-2*Angle[i]))
        lbg.append(0)
        ubg.append(10000000)
        g.append((phi_new[i]-Angle[i])*(phi[i+1]-Angle[i]))
        lbg.append(-10000000)
        ubg.append(0)

        g.append(ca.fabs(r0[i]))
        lbg.append(r_min_pixels)  # Use config radius limit
        ubg.append(100000)  # Much larger radius maximum
        g.append(l[i])
        lbg.append(0.01)
        ubg.append(Distance[i][0])  # Maximum length is the distance between waypoints
        center=(r0[i]*ca.cos(phi_new[i]+np.pi/2),r0[i]*ca.sin(phi_new[i]+np.pi/2))
        for k in range(3):
            theta=phi_new[i]+ca.pi/2+(phi[i+1]-phi_new[i])/3*(k+1)    
            x_k = r0[i] * ca.cos(theta)-center[0]
            y_k = r0[i] * ca.sin(theta)-center[1]
            g.append(-x_k*ca.sin(Angle[i][0])+y_k*ca.cos(Angle[i][0]))
            lbg.append(safe_corridor[i][1]+10)
            ubg.append(safe_corridor[i][2]-10)

    lb_phi = -np.pi * np.ones(N)
    ub_phi = np.pi * np.ones(N)
    lb_phi[0]=phi0
    ub_phi[0]=phi0
    for i in range(1,N-2):
        bound1=max(Angle[i][0]-np.pi/2-Flagb[i]*np.pi/2,Angle[i-1][0]-np.pi/2)
        bound2=min(Angle[i][0]+np.pi/2-Flagb[i]*np.pi/2,Angle[i-1][0]+np.pi/2)
        lb_phi[i]=min(bound1,bound2)
        ub_phi[i]=max(bound1,bound2)
    lb_phi[N-1]=Angle[N-2][0]-np.pi/2
    ub_phi[N-1]=Angle[N-2][0]+np.pi/2
    # for i in range(1,N-2):
    #     lb_phi[i]=Angle[i+1]-np.pi/2-Flagb[i+1]*np.pi/2
    #     ub_phi[i]=Angle[i+1]+np.pi/2-Flagb[i+1]*np.pi/2
    # Define an example objective function (e.g., minimize the sum of squares)
    # theta_l,theta_c,length_min,length_max,curvature_min,curvature_max=get_NormalizationMatrix(file_name=Normalization_file)
    # al=1
    # ac=1
    objective=0
    # to test the different between considering minimazing the curvature or not
    # al=np.ones(N-1)
    # length_min=np.zeros(N-1)
    for i in range(N-1):
        # objective = objective + al*theta_l*(ca.fabs((phi[i+1]- phi[i]-Flagb[i]*np.pi/2)*r0[i])+l[i]-length_min)+ac*theta_c*(1/ca.fabs(r0[i])-curvature_min)
        objective = objective + al[i]*(ca.fabs((phi[i+1]- phi[i]-Flagb[i]*np.pi/2)*r0[i])+l[i]-length_min[i])+ac*(ca.if_else(ca.cos(phi_new[i]-phi[i+1])!=1,1/ca.fabs(r0[i]),0))



    # Create an NLP problem
    nlp = {'x': ca.vertcat(phi), 'f': objective, 'g': ca.vertcat(*g)}

    opts_setting = {'ipopt.max_iter':10000, 'ipopt.print_level':3, 'print_time':1, 'ipopt.acceptable_tol':1e-5, 'ipopt.acceptable_obj_change_tol':1e-5}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)


    # Initial guess
    x0 = Initial_Guess[:N]
    # x0=np.zeros(N)
    # Solve the NLP problem
    sol = solver(x0=x0, lbx=lb_phi, ubx=ub_phi, lbg= lbg, ubg=ubg)

    # Get the solution
    phi_opt = sol['x'].full().flatten()

    print("the phi of each waypoint:",phi_opt)
    # plot the path
    fig, ax = plt.subplots()
    # Print the solution
    print("Optimal result", sol['f'])
    l_Opt=np.zeros(N-1)
    r_Opt=np.zeros(N-1)
    if solver.stats()['success']:
        print("Solver succeeded!")
        for i in range(N-1):
            Ps=reeb_graph.nodes[Waypoints[i]].configuration
            if Flagb[i]!=0:
                ax.plot(Ps[0], Ps[1], 'ro', label='relay point')
            else:
                plt.plot(Ps[0], Ps[1], 'go',label='waypoint')
            phi_new_opt=phi_opt[i]+Flagb[i]*np.pi/2 #A
            thetaC_opt=phi_opt[i+1]/2 + phi_new_opt/2
            sigma_opt=-1+ca.cos(phi_new_opt-phi_opt[i+1])
            l_opt=Distance[i]*(ca.cos(Angle[i]-phi_new_opt)-ca.cos(Angle[i]-phi_opt[i+1]))/sigma_opt
            r0_opt=-Distance[i]*ca.sin(Angle[i]-phi_opt[i+1])/sigma_opt
        #     # Plot the arc
            theta_start =phi_new_opt+np.pi/2
            theta_end = phi_opt[i+1]+np.pi/2 # Assuming the arc spans the angle phi_A
            center=(r0_opt*np.cos(theta_start),r0_opt*np.sin(theta_start))
            theta = np.linspace(theta_start, theta_end, 100)
            x = r0_opt * np.cos(theta)-center[0]
            y = r0_opt * np.sin(theta)-center[1]
            # plot line 
            x_line=np.linspace(r0_opt * np.cos(theta_end)-center[0],r0_opt * np.cos(theta_end)-center[0]+l_opt*np.cos(phi_opt[i+1]),100)
            y_line=np.linspace(r0_opt * np.sin(theta_end)-center[1],r0_opt * np.sin(theta_end)-center[1]+l_opt*np.sin(phi_opt[i+1]),100)
            x=x+Ps[0]
            y=y+Ps[1]
            x_line=x_line+Ps[0]
            y_line=y_line+Ps[1]
            ax.plot(x, y,'b')
            ax.plot(x_line, y_line,'b')
            l_Opt[i]=l_opt
            r_Opt[i]=r0_opt
        data = {
        'Optimization_phi': phi_opt.tolist(),
        'Optimization_l': l_Opt.tolist(),
        'Optimization_r': r_Opt.tolist(),
        'Optimizationf': float(sol['f'])
        }
        with open(Result_file, 'w') as file:
            json.dump(data, file)  
        # Get the solution     
        loaded_environment = Environment.load_from_file(environment_file)
        loaded_environment.draw("black")
        
        # Plot safe corridor boundaries
        for i in range(N-1):
            # Get waypoint positions
            start_pos = reeb_graph.nodes[Waypoints[i]].configuration
            end_pos = reeb_graph.nodes[Waypoints[i+1]].configuration
            
            # Calculate corridor bounds
            slope = safe_corridor[i][0]
            y_min = safe_corridor[i][1]
            y_max = safe_corridor[i][2]
            
            # Generate corridor boundary points
            if abs(slope) > 100000:  # Vertical line case
                # For vertical lines, y_min and y_max are in x-direction
                x_coords = [start_pos[0] + y_min, start_pos[0] + y_max, end_pos[0] + y_max, end_pos[0] + y_min, start_pos[0] + y_min]
                y_coords = [start_pos[1], start_pos[1], end_pos[1], end_pos[1], start_pos[1]]
            else:
                # Non-vertical line case - calculate perpendicular offsets
                length = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                dx = (end_pos[0] - start_pos[0]) / length
                dy = (end_pos[1] - start_pos[1]) / length
                
                # Perpendicular direction
                perp_dx = -dy
                perp_dy = dx
                
                # Corridor boundary points
                p1 = [start_pos[0] + y_min * perp_dx, start_pos[1] + y_min * perp_dy]
                p2 = [start_pos[0] + y_max * perp_dx, start_pos[1] + y_max * perp_dy]
                p3 = [end_pos[0] + y_max * perp_dx, end_pos[1] + y_max * perp_dy]
                p4 = [end_pos[0] + y_min * perp_dx, end_pos[1] + y_min * perp_dy]
                
                x_coords = [p1[0], p2[0], p3[0], p4[0], p1[0]]
                y_coords = [p1[1], p2[1], p3[1], p4[1], p1[1]]
            
            ax.plot(x_coords, y_coords, 'g--', alpha=0.5, linewidth=1, label='Safe Corridor' if i == 0 else "")
        
        plt.autoscale()
        ax.set_aspect('equal')  # Keep x and y with same scale
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(figure_file)
    else:    
        print("Solver failed!")

    # loaded_environment = Environment.load_from_file(environment_file)
    # loaded_environment.draw("black")
    # plt.savefig(figure_file)
def Planning_error_withinSC(waypoints_file_path,Normalization_path,environment_file,safe_corridor,reeb_graph,phi0,Initial_Guess_file_path,Result_file,figure_file):

    # Load config for radius limit
    import sys
    sys.path.append('/root/workspace/config')
    from config_loader import config
    
    # Convert radius limit from meters to pixels
    r_min_pixels = config.meters_to_pixels(config.r_lim)

    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
    Initial_Guess=np.array(load_initial_guess_from_file(Initial_Guess_file_path))
    with open(Normalization_path, 'r') as file:
        data = json.load(file)
    al=data['al']
    ac=data['ac']
    length_min=data['length_min']
    curvature_min=data['curvature_min']
    # Number of variables
    N = len(Waypoints)
    # calculate the distance and angle between waypoints
    Distance=np.zeros((N-1,1)) # r_B in polar coordinate
    Angle=np.zeros((N-1,1)) # theta_B in polar coordinate
    for i in range(N-1):
        Distance[i]=np.linalg.norm(reeb_graph.nodes[Waypoints[i+1]].configuration-reeb_graph.nodes[Waypoints[i]].configuration)
        Angle[i] = np.arctan2(reeb_graph.nodes[Waypoints[i+1]].configuration[1] - reeb_graph.nodes[Waypoints[i]].configuration[1],
                            reeb_graph.nodes[Waypoints[i+1]].configuration[0] - reeb_graph.nodes[Waypoints[i]].configuration[0])

    # Create symbolic variables
    phi=ca.SX.sym('phi', N) # the angel at each waypoints
    sigma = ca.SX.sym('sigma', N-1) # the angle of each arc
    r0=ca.SX.sym('r', N-1) # the radius of each arc
    l=ca.SX.sym('l', N-1) # the length of each straight line
    phi_new=ca.SX.sym('phi_new', N-1) # the angle of each straight line
    for i in range(N-1):
        phi_new[i]=phi[i]+Flagb[i]*np.pi/2 # A
        sigma[i]=ca.if_else(ca.cos(phi_new[i]-phi[i+1])!=1,-1+ca.cos(phi_new[i]-phi[i+1]),0.0001)
        l[i]=ca.if_else(ca.cos(phi_new[i]-phi[i+1])!=1,Distance[i]*(ca.cos(Angle[i]-phi_new[i])-ca.cos(Angle[i]-phi[i+1]))/sigma[i],Distance[i])
        r0[i]=-Distance[i]*ca.sin(Angle[i]-phi[i+1])/sigma[i]
    # delta=ca.MX.sym('delta',N-1) # angle of each straight line
    # al=1
    # # ac=0.05438074639134307
    # ac=100
    # Define the bounds for the variables
    g = [] # constrains
    lbg = []
    ubg = []
    g.append(phi[0])
    lbg.append(phi0)
    ubg.append(phi0)
    for i in range(N-1):
        g.append(ca.cos((phi_new[i]-phi[i+1])/4))
        lbg.append(0)
        ubg.append(1)
        g.append((phi_new[i]-Angle[i])*(phi_new[i]+phi[i+1]-2*Angle[i]))
        lbg.append(0)
        ubg.append(1000)  # Reduced from 10000000
        g.append((phi_new[i]-Angle[i])*(phi[i+1]-Angle[i]))
        lbg.append(-1000)  # Reduced from -10000000
        ubg.append(0)

        g.append(ca.fabs(r0[i]))
        lbg.append(r_min_pixels)  # Slightly more relaxed radius constraint
        ubg.append(10000)  # Much more reasonable radius maximum
        g.append(l[i])
        lbg.append(0)  # Minimum length matches GA_planning constraint
        ubg.append(np.pi*Distance[i][0]/2)  # Maximum length matches GA_planning constraint
        center=(r0[i]*ca.cos(phi_new[i]+np.pi/2),r0[i]*ca.sin(phi_new[i]+np.pi/2))
        
        # Simplified safe corridor constraints - check fewer key points to avoid over-constraining
        # Check start point, middle point, and end point of the arc
        for k in [0.25, 0.5, 0.75]:  # Check 3 strategic points along the arc
            theta=phi_new[i]+ca.pi/2+(phi[i+1]-phi_new[i])*k    
            x_k = r0[i] * ca.cos(theta)-center[0]
            y_k = r0[i] * ca.sin(theta)-center[1]
            # Transform to corridor coordinate system and check bounds with larger safety margin
            g.append(-x_k*ca.sin(Angle[i][0])+y_k*ca.cos(Angle[i][0]))
            lbg.append(safe_corridor[i][1] + 5)  # Larger safety margin for better convergence
            ubg.append(safe_corridor[i][2] - 5)  # Larger safety margin for better convergence

    lb_phi = -np.pi * np.ones(N)
    ub_phi = np.pi * np.ones(N)
    lb_phi[0]=phi0
    ub_phi[0]=phi0
    for i in range(1,N-2):
        bound1=max(Angle[i][0]-np.pi/2-Flagb[i]*np.pi/2,Angle[i-1][0]-np.pi/2)
        bound2=min(Angle[i][0]+np.pi/2-Flagb[i]*np.pi/2,Angle[i-1][0]+np.pi/2)
        lb_phi[i]=min(bound1,bound2)
        ub_phi[i]=max(bound1,bound2)
    lb_phi[N-1]=Angle[N-2][0]-np.pi/2
    ub_phi[N-1]=Angle[N-2][0]+np.pi/2
    # for i in range(1,N-2):
    #     lb_phi[i]=Angle[i+1]-np.pi/2-Flagb[i+1]*np.pi/2
    #     ub_phi[i]=Angle[i+1]+np.pi/2-Flagb[i+1]*np.pi/2
    # Define an example objective function (e.g., minimize the sum of squares)
    # theta_l,theta_c,length_min,length_max,curvature_min,curvature_max=get_NormalizationMatrix(file_name=Normalization_file)
    # al=1
    # ac=1
    objective=0
    # to test the different between considering minimazing the curvature or not
    # al=np.ones(N-1)
    # length_min=np.zeros(N-1)
    
    # Objective function similar to GA_planning for easier comparison
    for i in range(N-1):
        arc_length = ca.fabs((phi[i+1] - phi[i] - Flagb[i]*np.pi/2) * r0[i])
        straight_length = l[i]
        curvature = ca.fabs(Distance[i][0] * ca.sin(Angle[i][0] - phi[i+1]))
        objective = objective + al[i] * (arc_length + straight_length - length_min[i]) + ac * curvature
        #     lbg.append(safe_corridor[i][1])
        #     ubg.append(safe_corridor[i][2])
        # objective = objective + al*theta_l*(ca.fabs((phi[i+1]- phi[i]-Flagb[i]*np.pi/2)*r0[i])+l[i]-length_min)+ac*theta_c*(1/ca.fabs(r0[i])-curvature_min)
        # objective = objective + al[i]*(ca.fabs((phi[i+1]- phi[i]-Flagb[i]*np.pi/2)*r0[i])+l[i]-length_min[i])+ac*(ca.if_else(ca.cos(phi_new[i]-phi[i+1])!=1,1/ca.fabs(r0[i]),0))



    # Create an NLP problem
    nlp = {'x': ca.vertcat(phi), 'f': objective, 'g': ca.vertcat(*g)}

    # Improved solver settings for better convergence with safe corridor constraints
    opts_setting = {
        'ipopt.max_iter': 3000,           # Increase iterations for complex constraints
        'ipopt.print_level': 2,           # Reduce print level to avoid clutter
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-1,     # Much more relaxed tolerance for feasibility
        'ipopt.acceptable_obj_change_tol': 1e-2,
        'ipopt.tol': 1e-2,                # More relaxed tolerance
        'ipopt.constr_viol_tol': 1e-1,    # Allow larger constraint violations
        'ipopt.acceptable_iter': 10,      # Accept solution after 10 iterations at acceptable tolerance
        'ipopt.mu_init': 1e-1,           # Start with larger barrier parameter
        'ipopt.barrier_tol_factor': 10,  # More relaxed barrier tolerance
        'ipopt.bound_relax_factor': 1e-6  # Relax bounds slightly
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)


    # Initial guess
    x0 = Initial_Guess[:N]
    # x0=np.zeros(N)
    # Solve the NLP problem
    sol = solver(x0=x0, lbx=lb_phi, ubx=ub_phi, lbg= lbg, ubg=ubg)

    # Get the solution
    phi_opt = sol['x'].full().flatten()

    print("the phi of each waypoint:",phi_opt)
    # plot the path
    fig, ax = plt.subplots()
    
    # Load environment to get boundary information
    loaded_environment = Environment.load_from_file(environment_file)
    
    # Draw environment boundary from coord_bounds
    coord_bounds = loaded_environment.coord_bounds
    x_min, x_max, y_min, y_max = coord_bounds
    
    # Draw boundary walls as thick black lines
    boundary_width = 3
    ax.plot([x_min, x_max], [y_min, y_min], 'k-', linewidth=boundary_width, label='Environment Boundary')  # Bottom wall
    ax.plot([x_min, x_max], [y_max, y_max], 'k-', linewidth=boundary_width)  # Top wall
    ax.plot([x_min, x_min], [y_min, y_max], 'k-', linewidth=boundary_width)  # Left wall
    ax.plot([x_max, x_max], [y_min, y_max], 'k-', linewidth=boundary_width)  # Right wall
    
    # Draw environment obstacles
    loaded_environment.draw("gray")
    
    # Print the solution
    print("Optimal result", sol['f'])
    l_Opt=np.zeros(N-1)
    r_Opt=np.zeros(N-1)
    if solver.stats()['success']:
        print("Solver succeeded!")
        for i in range(N-1):
            Ps=reeb_graph.nodes[Waypoints[i]].configuration
            if Flagb[i]!=0:
                ax.plot(Ps[0], Ps[1], 'ro', label='relay point')
            else:
                plt.plot(Ps[0], Ps[1], 'go',label='waypoint')
            phi_new_opt=phi_opt[i]+Flagb[i]*np.pi/2 #A
            thetaC_opt=phi_opt[i+1]/2 + phi_new_opt/2
            sigma_opt=-1+ca.cos(phi_new_opt-phi_opt[i+1])
            l_opt=Distance[i]*(ca.cos(Angle[i]-phi_new_opt)-ca.cos(Angle[i]-phi_opt[i+1]))/sigma_opt
            r0_opt=-Distance[i]*ca.sin(Angle[i]-phi_opt[i+1])/sigma_opt
        #     # Plot the arc
            theta_start =phi_new_opt+np.pi/2
            theta_end = phi_opt[i+1]+np.pi/2 # Assuming the arc spans the angle phi_A
            center=(r0_opt*np.cos(theta_start),r0_opt*np.sin(theta_start))
            theta = np.linspace(theta_start, theta_end, 100)
            x = r0_opt * np.cos(theta)-center[0]
            y = r0_opt * np.sin(theta)-center[1]
            # plot line 
            x_line=np.linspace(r0_opt * np.cos(theta_end)-center[0],r0_opt * np.cos(theta_end)-center[0]+l_opt*np.cos(phi_opt[i+1]),100)
            y_line=np.linspace(r0_opt * np.sin(theta_end)-center[1],r0_opt * np.sin(theta_end)-center[1]+l_opt*np.sin(phi_opt[i+1]),100)
            x=x+Ps[0]
            y=y+Ps[1]
            x_line=x_line+Ps[0]
            y_line=y_line+Ps[1]
            ax.plot(x, y,'b')
            ax.plot(x_line, y_line,'b')
            l_Opt[i]=l_opt
            r_Opt[i]=r0_opt
        data = {
        'Optimization_phi': phi_opt.tolist(),
        'Optimization_l': l_Opt.tolist(),
        'Optimization_r': r_Opt.tolist(),
        'Optimizationf': float(sol['f'])
        }
        with open(Result_file, 'w') as file:
            json.dump(data, file)  
        # Get the solution     
        loaded_environment = Environment.load_from_file(environment_file)
        loaded_environment.draw("black")
        
        # Plot safe corridor boundaries
        for i in range(N-1):
            # Get waypoint positions
            start_pos = reeb_graph.nodes[Waypoints[i]].configuration
            end_pos = reeb_graph.nodes[Waypoints[i+1]].configuration
            
            # Calculate corridor bounds
            slope = safe_corridor[i][0]
            y_min = safe_corridor[i][1]
            y_max = safe_corridor[i][2]
            
            # Generate corridor boundary points
            if abs(slope) > 100000:  # Vertical line case
                # For vertical lines, y_min and y_max are in x-direction
                x_coords = [start_pos[0] + y_min, start_pos[0] + y_max, end_pos[0] + y_max, end_pos[0] + y_min, start_pos[0] + y_min]
                y_coords = [start_pos[1], start_pos[1], end_pos[1], end_pos[1], start_pos[1]]
            else:
                # Non-vertical line case - calculate perpendicular offsets
                length = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                dx = (end_pos[0] - start_pos[0]) / length
                dy = (end_pos[1] - start_pos[1]) / length
                
                # Perpendicular direction
                perp_dx = -dy
                perp_dy = dx
                
                # Corridor boundary points
                p1 = [start_pos[0] + y_min * perp_dx, start_pos[1] + y_min * perp_dy]
                p2 = [start_pos[0] + y_max * perp_dx, start_pos[1] + y_max * perp_dy]
                p3 = [end_pos[0] + y_max * perp_dx, end_pos[1] + y_max * perp_dy]
                p4 = [end_pos[0] + y_min * perp_dx, end_pos[1] + y_min * perp_dy]
                
                x_coords = [p1[0], p2[0], p3[0], p4[0], p1[0]]
                y_coords = [p1[1], p2[1], p3[1], p4[1], p1[1]]
            
            ax.plot(x_coords, y_coords, 'g--', alpha=0.5, linewidth=1, label='Safe Corridor' if i == 0 else "")
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('IPOPT Planning Result with Environment Boundary')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')  # Keep x and y with same scale
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_file)
    else:    
        print("Solver failed!")
        # Print detailed solver statistics for debugging
        stats = solver.stats()
        print(f"Solver statistics:")
        print(f"  Return status: {stats.get('return_status', 'unknown')}")
        print(f"  Success: {stats.get('success', False)}")
        print(f"  Iterations: {stats.get('iter_count', 'unknown')}")
        print(f"  Objective value: {sol['f']}")
        print(f"  Constraint violation: {stats.get('constr_viol', 'unknown')}")
        
        # Still save the solution data even if solver "failed" but found a solution
        phi_opt = sol['x'].full().flatten()
        print(f"Final phi values: {phi_opt}")
        
        # Save the result anyway - sometimes Ipopt reports "failure" but still finds good solutions
        data = {
            'phi': phi_opt.tolist(),
            'objective_value': float(sol['f']),
            'solver_success': False,
            'solver_stats': {k: str(v) for k, v in stats.items()},
            'coordinate_frame': 'world_pixel'
        }
        with open(Result_file, 'w') as file:
            json.dump(data, file, indent=4)


# normally unsolved!
def Planning(waypoints_file_path,al,ac,Max0Min,reeb_graph,Initial_Guess_file_path):

    # Load config for radius limit
    import sys
    sys.path.append('/root/workspace/config')
    from config_loader import config
    
    # Convert radius limit from meters to pixels
    r_min_pixels = config.meters_to_pixels(config.r_lim)

    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)

    Initial_Guess=np.array(load_initial_guess_from_file(Initial_Guess_file_path))

    # Number of variables
    N = len(Waypoints)
    # calculate the distance and angle between waypoints
    Distance=np.zeros((N-1,1)) # r_B in polar coordinate
    Angle=np.zeros((N-1,1)) # theta_B in polar coordinate
    for i in range(N-1):
        Distance[i]=np.linalg.norm(reeb_graph.nodes[Waypoints[i+1]].configuration-reeb_graph.nodes[Waypoints[i]].configuration)
        Angle[i] = np.arctan2(reeb_graph.nodes[Waypoints[i+1]].configuration[1] - reeb_graph.nodes[Waypoints[i]].configuration[1],
                            reeb_graph.nodes[Waypoints[i+1]].configuration[0] - reeb_graph.nodes[Waypoints[i]].configuration[0])

    # Create symbolic variables
    phi=ca.SX.sym('phi', N) # the angel at each waypoints
    sigma = ca.SX.sym('sigma', N-1) # the angle of each arc
    r0=ca.SX.sym('r', N-1) # the radius of each arc
    l=ca.SX.sym('l', N-1) # the length of each straight line
    phi_new=ca.SX.sym('phi_new', N-1) # the angle of each straight line
    for i in range(N-1):
        phi_new[i]=phi[i]+Flagb[i]*np.pi/2 # A
        sigma[i]=-1+ca.cos(phi_new[i]-phi[i+1])
        l[i]=ca.if_else(phi[i+1]!=phi_new[i],Distance[i]*(ca.cos(Angle[i]-phi_new[i])-ca.cos(Angle[i]-phi[i+1]))/sigma[i],Distance[i])
        r0[i]=ca.if_else(phi[i+1]!=phi_new[i],-Distance[i]*ca.sin(Angle[i]-phi[i+1])/sigma[i],100000)
    # delta=ca.MX.sym('delta',N-1) # angle of each straight line
    # al=1
    # # ac=0.05438074639134307
    # ac=100
    # Define the bounds for the variables
    g = [] # constrains
    lbg = []
    ubg = []
    for i in range(N-1):

        g.append(ca.cos((phi_new[i]-phi[i+1])/4))
        lbg.append(0)
        ubg.append(1)
        g.append((phi_new[i]-Angle[i])*(phi_new[i]+phi[i+1]-2*Angle[i]))
        lbg.append(0)
        ubg.append(10000000)
        g.append((phi_new[i]-Angle[i])*(phi[i+1]-Angle[i]))
        lbg.append(-10000000)
        ubg.append(0)

        g.append(ca.fabs(r0[i]))
        lbg.append(r_min_pixels)  # Use config radius limit
        ubg.append(100000)  # Much larger radius maximum
        g.append(l[i])
        lbg.append(0)
        ubg.append(Distance[i][0])  # Maximum length is the distance between waypoints

    lb_phi = -np.pi * np.ones(N)
    ub_phi = np.pi * np.ones(N)

    # Define an example objective function (e.g., minimize the sum of squares)
    # theta_l,theta_c,length_min,length_max,curvature_min,curvature_max=get_NormalizationMatrix(file_name=Normalization_file)
    # al=1
    # ac=1
    objective=0
    for i in range(N-1):
        # objective = objective + al*theta_l*(ca.fabs((phi[i+1]- phi[i]-Flagb[i]*np.pi/2)*r0[i])+l[i]-length_min)+ac*theta_c*(1/ca.fabs(r0[i])-curvature_min)
        objective = objective + al*ca.fabs((phi[i+1]- phi[i]-Flagb[i]*np.pi/2)*r0[i])+al*l[i]+ac*ca.if_else(phi[i+1]!=phi_new[i],1/ca.fabs(r0[i]),0)
    if Max0Min=="Max":
        objective=-objective


    # Create an NLP problem
    nlp = {'x': ca.vertcat(phi), 'f': objective, 'g': ca.vertcat(*g)}

    opts_setting = {'ipopt.max_iter':10000, 'ipopt.print_level':5, 'print_time':1, 'ipopt.acceptable_tol':1e-5, 'ipopt.acceptable_obj_change_tol':1e-5}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)


    # Initial guess
    x0 = Initial_Guess[:N]
    # Solve the NLP problem
    sol = solver(x0=x0, lbx=lb_phi, ubx=ub_phi, lbg= lbg, ubg=ubg)

    # Get the solution
    phi_opt = sol['x'].full().flatten()

    print("the phi of each waypoint:",phi_opt)
    # plot the path
    fig, ax = plt.subplots()
    # Print the solution
    print("Optimal result", sol['f'])
    l_Opt=np.zeros(N-1)
    r_Opt=np.zeros(N-1)
    if solver.stats()['success']:
        print("Solver succeeded!")
        for i in range(N-1):
            Ps=reeb_graph.nodes[Waypoints[i]].configuration
            if Flagb[i]!=0:
                ax.plot(Ps[0], Ps[1], 'ro', label='relay point')
            else:
                plt.plot(Ps[0], Ps[1], 'go',label='waypoint')
            phi_new_opt=phi_opt[i]+Flagb[i]*np.pi/2 #A
            thetaC_opt=phi_opt[i+1]/2 + phi_new_opt/2
            sigma_opt=-1+ca.cos(phi_new_opt-phi_opt[i+1])
            l_opt=Distance[i]*(ca.cos(Angle[i]-phi_new_opt)-ca.cos(Angle[i]-phi_opt[i+1]))/sigma_opt
            r0_opt=-Distance[i]*ca.sin(Angle[i]-phi_opt[i+1])/sigma_opt
        #     # Plot the arc
            theta_start =phi_new_opt+np.pi/2
            theta_end = phi_opt[i+1]+np.pi/2 # Assuming the arc spans the angle phi_A
            center=(r0_opt*np.cos(theta_start),r0_opt*np.sin(theta_start))
            theta = np.linspace(theta_start, theta_end, 100)
            x = r0_opt * np.cos(theta)-center[0]
            y = r0_opt * np.sin(theta)-center[1]
            # plot line 
            x_line=np.linspace(r0_opt * np.cos(theta_end)-center[0],r0_opt * np.cos(theta_end)-center[0]+l_opt*np.cos(phi_opt[i+1]),100)
            y_line=np.linspace(r0_opt * np.sin(theta_end)-center[1],r0_opt * np.sin(theta_end)-center[1]+l_opt*np.sin(phi_opt[i+1]),100)
            x=x+Ps[0]
            y=y+Ps[1]
            x_line=x_line+Ps[0]
            y_line=y_line+Ps[1]
            ax.plot(x, y,'b')
            ax.plot(x_line, y_line,'b')
            l_Opt[i]=l_opt
            r_Opt[i]=r0_opt 
        # Get the solution     
        return float(sol['f'])
    else:    
        print("Solver failed!")

    # loaded_environment = Environment.load_from_file(environment_file)
    # loaded_environment.draw("black")
    # plt.savefig(figure_file)

def get_normalization_prams(waypoints_file_path,Normalization_path,reeb_graph,Initial_Guess_file_path):
    # Load config for radius limit
    import sys
    sys.path.append('/root/workspace/config')
    from config_loader import config
    
    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
    N=len(Waypoints)
    length_min=[]
    length_max=[]
    
    # Convert radius limit from meters to pixels
    r_min = config.meters_to_pixels(config.r_lim)  # Convert from meters to pixels
    print(f"Using radius limit: {r_min:.2f} pixels (converted from {config.r_lim} meters)")
    
    al=[]
    for i in range(N-1):
        d=np.linalg.norm(reeb_graph.nodes[Waypoints[i+1]].configuration-reeb_graph.nodes[Waypoints[i]].configuration)
        length_min.append(d)
        length_max.append(d*np.pi/2)
        al.append(1/(d*np.pi/2-d))
    curvature_min=0
    curvature_max=1/r_min
    ac=1/(curvature_max-curvature_min)
    data={
        'length_max':length_max,
        'length_min':length_min,
        'curvature_max':curvature_max,
        'curvature_min':curvature_min,
        'al':al,
        'ac':ac
    }
    with open(Normalization_path, 'w') as file:
                json.dump(data, file)
    return True

    