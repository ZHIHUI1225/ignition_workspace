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
        self.file_path = f"Graph_new_{self.case}.json"
        self.environment_file = f"environment_{self.case}.json"
        self.assignment_result_file = f"AssignmentResult{self.N}{self.case}.json"
        self.waypoints_file_path = f"WayPointFlag{self.N}{self.case}.json"
        # Two different normalization files
        self.Normalization_path = f"Normalization{self.N}_{self.case}.json"  # For assignment
        self.Normalization_planning_path = f"Normalization_planning{self.N}_{self.case}.json"  # For GA planning
        self.Result_file = f"Optimization_GA_{self.N}_IG_norma{self.case}.json"
        self.figure_file = f"Optimization_GA_{self.N}_IG_norma{self.case}.png"
    
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

# Global config instance
config = PlanningConfig()
