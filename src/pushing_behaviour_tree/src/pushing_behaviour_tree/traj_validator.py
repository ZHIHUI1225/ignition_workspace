import dubins
from math import pi, sqrt, sin, cos
from random import random
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def dist(a,b):
    return sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def check_traj(start,traj,turning_radius,obstacles=[]):
    step_size = 0.1
    safe_distance = 0.4
    is_feasible = []
    for t in traj:
        path = dubins.shortest_path(start, t, turning_radius)
        configurations, _ = path.sample_many(step_size)
        is_safe = True
        for conf in configurations:
            for obstacle in obstacles:
                if dist(conf,obstacle)<safe_distance:
                    is_safe = False
                    break
            else:
                continue
            break
        is_feasible.append(is_safe)
    return any(is_feasible)
    
def transform_trajectory(t,b_,):
    traj = []
    for p in t:
        p_x = p.pose.position.x
        p_y = p.pose.position.y
        q = [p.pose.orientation.x, p.pose.orientation.y,
            p.pose.orientation.z, p.pose.orientation.w]
        thetao = euler_from_quaternion(
             q)[2]
        p_r = (p_x-b_*cos(thetao),p_y-b_*sin(thetao),thetao)
        traj.append(p_r)
    return traj

if __name__=="__main__":
    start = (0.0, 0.0, 0.0)
    t_start = (0.0,-0.75,0.0)
    t_end = (5.0, 0.75, 0.0)
    turning_radius = 0.4
    path = dubins.shortest_path(t_start, t_end, turning_radius)
    traj, _ = path.sample_many(0.01)
    obstacles = [(5.0*random(),2.0*random()-1.0) for i in range(10)]
    plt.plot([t[0] for t in obstacles],[t[1] for t in obstacles],'ro')
    plt.plot([start[0]],[start[1]],'go')
    plt.plot([t[0] for t in traj],[t[1] for t in traj],'go')
    print(f'Result %s' % check_traj(start,traj,turning_radius,obstacles))
    plt.show()