#!/usr/bin/env python3
"""
Configuration loader for GA Planning System
This module provides easy access to the centralized YAML configuration
"""

import yaml
import os

class PlanningConfig:
    """Configuration loader for GA planning system"""
    
    def __init__(self, config_path="/root/workspace/config/config.yaml"):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Basic configuration
        self.case = config['case_name']
        self.N = config['robots']['count']
        self.r_lim = config['robots']['radius_limit']
        self.min_length = config['robots']['min_length']
        
        # Planning parameters
        self.phi0 = config['planning']['phi0']
        self.deltal = config['planning']['deltal']  # Small segment length for discretization
        
        # Robot physical parameters
        robot_phys = config['robot_physical']
        self.aw_max = robot_phys['angular_acceleration_max']
        self.w_max = robot_phys['angular_velocity_max']
        self.r_w = robot_phys['wheel_radius']
        self.l_r = robot_phys['wheelbase']
        self.r_limit = robot_phys['turning_radius_limit']
        self.v_max = robot_phys['linear_velocity_max']
        self.a_max = robot_phys['linear_acceleration_max']
        self.mu = robot_phys['friction_coefficient']
        self.mu_f = robot_phys['safety_factor']
        self.g = robot_phys['gravity']
        self.mu_mu_f = robot_phys['friction_limit']
        
        # Coordinate conversion parameters
        coord_conv = config['coordinate_conversion']
        self.pixel_to_meter_scale = coord_conv['pixel_to_meter_scale']
        self.meter_to_pixel_scale = coord_conv['meter_to_pixel_scale']
        
        # Arc range configuration
        self.arc_range = config['arc_range']
        
        # Paths
        self.base_path = config['paths']['base_path']
        self.data_path = config['paths']['data_path']
        
        # Poses
        self.start_pose = config['poses']['start_pose']
        self.goal_pose = config['poses']['goal_pose']
        
        # For backward compatibility, also create start and goal attributes
        self.start = config['poses']['start_pose']
        self.goal = config['poses']['goal_pose']
        
        # GA parameters
        ga_config = config['genetic_algorithm']
        self.num_generations = ga_config['num_generations']
        self.sol_per_pop = ga_config['population_size']
        self.num_parents_mating = ga_config['num_parents_mating']
        self.mutation_probability = ga_config['mutation_probability']
        self.crossover_probability = ga_config['crossover_probability']
        self.keep_elitism = ga_config['keep_elitism']
        self.parent_selection_type = ga_config['parent_selection_type']
        self.crossover_type = ga_config['crossover_type']
        self.mutation_type = ga_config['mutation_type']
        
        # Constraints
        constraints = config['constraints']
        self.angle_bounds = constraints['angle_bounds']
        self.first_waypoint = constraints['first_waypoint']
        self.penalties = constraints['penalties']
        
        # Visualization
        self.visualization = config['visualization']
        
        # Monitoring
        self.monitoring = config['monitoring']
        
        # Generate file paths
        self._generate_file_paths()
    
    def _generate_file_paths(self):
        """Generate file paths based on case and robot count"""
        # Generate full paths to data directory for most files
        self.file_path = os.path.join(self.data_path, f"Graph_new_{self.case}.json")
        self.environment_file = os.path.join(self.data_path, f"environment_{self.case}.json")
        self.assignment_result_file = os.path.join(self.data_path, f"AssignmentResult{self.N}{self.case}.json")
        self.waypoints_file_path = os.path.join(self.data_path, f"WayPointFlag{self.N}{self.case}.json")
        # Two different normalization files
        self.Normalization_path = os.path.join(self.data_path, f"Normalization{self.N}_{self.case}.json")  # For assignment
        self.Normalization_planning_path = os.path.join(self.data_path, f"Normalization_planning{self.N}_{self.case}.json")  # For GA planning
        self.Result_file = os.path.join(self.data_path, f"Optimization_GA_{self.N}_IG_norma{self.case}.json")
        self.figure_file = os.path.join(self.data_path, f"Optimization_GA_{self.N}_IG_norma{self.case}.png")
        
        # Planning deltaT specific file paths
        self.ga_initial_guess_file = os.path.join(self.data_path, f"Optimization_GA_{self.N}_IG_norma{self.case}.json")  # True initial guess from GA
        self.ga_initial_guess_figure = os.path.join(self.data_path, f"InitialGuess{self.N}{self.case}.png")
        self.planning_path_result_file = os.path.join(self.data_path, f"Optimization_withSC_path{self.N}{self.case}.json")  # Result from Planning_path with safe corridors
        self.deltaT_result_file = os.path.join(self.data_path, f"Optimization_deltaT_{self.N}{self.case}.json")
        self.deltaT_figure_file = os.path.join(self.data_path, f"Optimization_deltaT_{self.N}{self.case}.png")
    
    def update_case(self, new_case):
        """Update case name and regenerate file paths"""
        self.case = new_case
        self._generate_file_paths()
    
    def update_robot_count(self, new_count):
        """Update robot count and regenerate file paths"""
        self.N = new_count
        self._generate_file_paths()
    
    def get_full_path(self, filename, use_data_path=False):
        """Get full path for a file"""
        base = self.data_path if use_data_path else self.base_path
        return os.path.join(base, filename)
    
    def pixels_to_meters(self, pixel_value):
        """Convert pixel values to meters"""
        if isinstance(pixel_value, (list, tuple)):
            return [p * self.pixel_to_meter_scale for p in pixel_value]
        else:
            return pixel_value * self.pixel_to_meter_scale
    
    def meters_to_pixels(self, meter_value):
        """Convert meter values to pixels"""
        if isinstance(meter_value, (list, tuple)):
            return [m * self.meter_to_pixel_scale for m in meter_value]
        else:
            return meter_value * self.meter_to_pixel_scale
    
    def get_robot_physical_params(self):
        """Get all robot physical parameters as a dictionary"""
        return {
            'aw_max': self.aw_max,
            'w_max': self.w_max,
            'r_w': self.r_w,
            'l_r': self.l_r,
            'r_limit': self.r_limit,
            'v_max': self.v_max,
            'a_max': self.a_max,
            'mu': self.mu,
            'mu_f': self.mu_f,
            'g': self.g,
            'mu_mu_f': self.mu_mu_f
        }
    
    def get_case_data_dir(self, case_name=None):
        """Get the full path to the data directory for a specific case"""
        case = case_name if case_name else self.case
        return os.path.join(self.data_path, case)

# Global config instance
config = PlanningConfig()
