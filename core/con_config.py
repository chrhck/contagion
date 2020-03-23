"""
Name: con_config.py
Authors: Stephan Meighen-Berger
Config file for the contagion package.
It is recommended, that only advanced users change
the settings here.
"""

"Imports"
import numpy as np
import logging

config = {
    ###################################################
    # General
    ###################################################
    # Output level
    'debug level': logging.INFO,
    # Switch to store steps or not
    # This requires a bit more memory
    "save population": True,
    # Simulation duration
    "simulation length": 100,
    # The probability distribution to use for the movement pdf
    # Currently supported:
    #   - 'gauss':
    #       A gaussian distribution
    'pdf move': 'gauss',
    # Simulation type:
    #   -'random walk':
    #       A simple random walk simulation,
    #       useful for visualizations
    #   -'realistic':
    #       Realistic social and infection modelling
    'simulation type': 'realistic',
    ###################################################
    # 'realistic' options
    ###################################################
    # The average size of a person's social circle
    'average social circle': 20,
    # The variance of one's social circle
    'variance social circle': 5,
    # Social circle pdf:
    # Available: 'gauss'
    'social circle pdf': 'gauss',
    # Average number of interactions per time step in sc
    'mean social circle interactions': 1,
    # Variance of sc interactions
    'variance social circle interactions': 2,
    # Distribution of the interaction rates
    # Available: 'gauss'
    'social circle interactions pdf': 'gauss',
    # Infection probability pdf
    # available: 'intensity'
    'infection probability pdf': 'intensity',
    # Infection duration mean
    'infection duration mean': 20,
    # Infection duration variance
    'infection duration variance': 5,
    # Infection duration pdf
    'infection duration pdf': 'gauss',
    # Interaction intensity distribution:
    # Available: uniform
    'interaction intensity': 'uniform',
    ###################################################
    # 'random walk' options
    ###################################################
    # Number of dimensions for the simulation
    # Current options:
    #   - 2, 3
    "dimensions": 2,
    # The geometry of the problem.
    #   -'box':
    #       Creates a uniform box of 1m x 1m x 1m evenly filled.
    #   -'sphere':
    #       Creates a uniform sphere
    #   -'custom':
    #       Use a custom geometry defined in a pkl file.
    #       Place file in data/detector/geometry which needs to be a
    #       dumped library with:
    #           {'dimensions': d,  # dimensions as int
    #            'bounding box': a,  # bounding box as float
    #            'volume': v,  # The volume
    #            'points': np.array  # point cloud as 2d array with e.g. [x,y,z]
    'geometry': 'box',
    'box size': 1e2,  # Side length in mm of box
    'sphere diameter': 1e2,  # Radius of the sphere
    'custom geometry': 'example_tetrahedron.pkl',  # File for custom geometry
    # Size of bounding box
    # This box needs to surround the volume of interest
    # It is used to create a population sample
    'bounding box': 1.1e2,
    ###################################################
    # More advanced
    ###################################################
    # Time step to use
    # TODO: Define time steps
    # This should be at maximum 1
    'time step': 1.,
    # Freedom of movement
    # How large the angle change between steps can be
    # (for the organisms)
    # Org. won't move with angles between the values
    "angle change": [90, 270],
    # Number of points to use when constructing a spherical
    # geometry. Increasing the number increases the precision,
    # while reducing efficiency
    'sphere samples': int(5e1),  # Number of points to construct the sphere
}