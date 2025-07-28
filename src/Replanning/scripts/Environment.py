import numpy as np
import json
import copy
from Polygon import Polygon, rectangle
class Environment:
    """ProblemConfigurations description of the environment, given by a list of polygons."""

    def __init__(self, polygons):
        self.polygons = polygons

        # Compute bounds of environment
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
        for polygon in self.polygons:
            for vertex in polygon.vertices:
                min_x = min(min_x, vertex[0])
                max_x = max(max_x, vertex[0])
                min_y = min(min_y, vertex[1])
                max_y = max(max_y, vertex[1])
        self.coord_bounds = (min_x, max_x, min_y, max_y)
        self.width = max_x - min_x
        self.height = max_y - min_y
        self.num_clear_calls = 0

    def clear_coords(self, x, y):
        """Returns whether or not a point is in the environment's free space."""
        return all(not poly.contains_coords(x, y) for poly in self.polygons)

    def clear(self, robot):
        """Returns whether or not the robot is in the environment's free space."""
        self.num_clear_calls += 1
        return all(not robot.intersects_polygon(poly) for poly in self.polygons)

    def link_coords(self, position_start, position_end, step_size):
        """Returns whether or not the interpolated point sequence from start to end lies entirely in the
        environment's free space."""
        dist = np.linalg.norm(position_start - position_end)
        num_steps = max(np.ceil(dist / step_size), 3)
        for x, y in zip(np.linspace(position_start[0], position_end[0], int(num_steps)),
                        np.linspace(position_start[1], position_end[1], int(num_steps))):
            if not self.clear_coords(x, y):
                return False
        return True

    def link(self, robot_start, robot_end, step_size):
        """Returns whether or not the interpolated configuration sequence from start to end lies entirely in the
        environment's free space."""
        workspace_pt1 = robot_start.configuration()
        workspace_pt2 = robot_end.configuration()

        dist = np.linalg.norm(workspace_pt1 - workspace_pt2)
        num_steps = max(np.ceil(dist / step_size), 3)
        # print("LINK: using {} steps".format(num_steps))

        robot = robot_start
        for x, y, angle in zip(np.linspace(workspace_pt1[0], workspace_pt2[0], int(num_steps)),
                               np.linspace(workspace_pt1[1], workspace_pt2[1],int( num_steps)),
                               np.linspace(workspace_pt1[2], workspace_pt2[2], int(num_steps))):
            robot.move(x, y, angle)
            if not self.clear(robot):
                return False

        return True

    def draw(self, color):
        for polygon in self.polygons:
            polygon.draw(color)

    def bounds(self):
        """Returns (min_x, max_x, min_y, max_y) bounds of environment."""
        return self.coord_bounds
    def to_dict(self):
        """Convert the environment to a dictionary for serialization."""
        return {
            'polygons': [{'vertices': polygon.vertices} for polygon in self.polygons],
            'coord_bounds': self.coord_bounds,
            'width': self.width,
            'height': self.height
        }

    @classmethod
    def from_dict(cls, data):
        """Create an environment from a dictionary."""
        polygons = [Polygon(vertices=poly['vertices']) for poly in data['polygons']]
        env = cls(polygons)
        
        # If coord_bounds, width, and height are provided in the data, use them
        # This preserves the original workspace bounds instead of recalculating from polygons
        if 'coord_bounds' in data:
            env.coord_bounds = tuple(data['coord_bounds'])
        if 'width' in data:
            env.width = data['width']
        if 'height' in data:
            env.height = data['height']
            
        return env

    def save_to_file(self, file_path):
        """Save the environment to a JSON file."""
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load_from_file(cls, file_path):
        """Load the environment from a JSON file."""
        with open(file_path, 'r') as file:
            data = json.load(file)
        return cls.from_dict(data)
    




