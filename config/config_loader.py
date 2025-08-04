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
        
        # Minimum turning radius (used by Planning_deltaT and other modules)
        self.r_limit = self.r_lim  # Use the same radius_limit from robots section
        
        # Timing configuration
        self.discrete_dt = config['discrete_dt']  # Time step for trajectory discretization and MPC frequency
        
        # Planning parameters
        self.phi0 = config['planning']['phi0']
        self.deltal = config['planning']['deltal']  # Small segment length for discretization
        
        # Robot physical parameters
        robot_phys = config['robot_physical']
        
        # Hardware limits (reference values)
        hardware_limits = robot_phys['hardware_limits']
        self.r_w = hardware_limits['wheel_radius']
        self.l_r = hardware_limits['wheelbase']
        
        # Operational limits (actual values used in planning)
        operational_limits = robot_phys['operational_limits']
        self.aw_max = operational_limits['angular_acceleration_max']
        self.w_max = operational_limits['angular_velocity_max']
        self.w_min = operational_limits.get('angular_velocity_min', -self.w_max)  # Default to negative max if not specified
        self.v_max = operational_limits['linear_velocity_max']
        self.v_min = operational_limits.get('linear_velocity_min', 0.0)  # Default to 0 if not specified
        self.a_max = operational_limits['linear_acceleration_max']
        self.mu = operational_limits['friction_coefficient']
        self.mu_f = operational_limits['safety_factor']
        self.g = operational_limits['gravity']
        self.mu_mu_f = operational_limits['friction_limit']
        
        # Additional operational parameters
        self.wheel_w_max = operational_limits['wheel_angular_velocity_max']
        self.wheel_aw_max = operational_limits['wheel_angular_acceleration_max']
        
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
            # Operational limits (used in planning)
            'aw_max': self.aw_max,
            'w_max': self.w_max,
            'w_min': self.w_min,
            'v_max': self.v_max,
            'v_min': self.v_min,
            'a_max': self.a_max,
            'wheel_w_max': self.wheel_w_max,
            'wheel_aw_max': self.wheel_aw_max,
            'r_limit': self.r_limit,
            
            # Physical dimensions
            'r_w': self.r_w,
            'l_r': self.l_r,
            
            # Environmental parameters
            'mu': self.mu,
            'mu_f': self.mu_f,
            'g': self.g,
            'mu_mu_f': self.mu_mu_f
        }
    
    def get_hardware_limits(self):
        """Get hardware limitation parameters"""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        hardware_limits = config['robot_physical']['hardware_limits']
        return {
            'linear_velocity_max_hw': hardware_limits['linear_velocity_max_hw'],
            'angular_velocity_max_hw': hardware_limits['angular_velocity_max_hw'],
            'wheel_angular_velocity_max': hardware_limits['wheel_angular_velocity_max'],
            'wheel_angular_acceleration_max': hardware_limits['wheel_angular_acceleration_max'],
            'linear_acceleration_max_hw': hardware_limits['linear_acceleration_max_hw'],
            'angular_acceleration_max_hw': hardware_limits['angular_acceleration_max_hw'],
            'max_steps_per_second': hardware_limits['max_steps_per_second'],
            'gear_reduction_ratio': hardware_limits['gear_reduction_ratio'],
            'steps_per_revolution': hardware_limits['steps_per_revolution']
        }
    
    def get_case_data_dir(self, case_name=None):
        """Get the full path to the data directory for a specific case"""
        case = case_name if case_name else self.case
        return os.path.join(self.data_path, case)

# Global config instance
config = PlanningConfig()
