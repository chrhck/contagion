# -*- coding: utf-8 -*-

"""
Name: config.py
Authors: Christian Haack, Stephan Meighen-Berger
Config file for the contagion package.
It is recommended, that only advanced users change
the settings here.
"""

import numpy as np
import logging
from typing import Dict, Any
import yaml


_baseconfig = {
    ###################################################
    # General
    ###################################################
    # Output level
    'debug level': logging.WARNING,
    # Location of logging file handler
    'log file handler': '../run/contagion.log',
    # Dump experiment config to this location
    'config location': '../run/config.txt',
    # The size of the population
    "population size": 1000,
    # Simulation duration
    "simulation length": 200,
    # The number of starting infections
    "infected": 10,
    # random state to use
    'random state': np.random.RandomState(1337),
    # re-simulate population
    "re-use population": False,
    # Population storage lcoation
    "population storage": "../populations/generic.pkl",
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
    'mean social circle interactions': 0.2,
    # sd of sc interactions
    'variance social circle interactions': 2,
    # Distribution of the interaction rates
    # Available: 'gauss'
    'social circle interactions pdf': 'gauss',
    # Infection probability pdf
    # available: 'intensity'
    'infection probability pdf': 'intensity',
    # Infection duration mean
    'infection duration mean': 14,
    # Infection duration sd
    'infection duration variance': 5,
    # Infection duration pdf
    # Available: 'gauss', 'gamma'
    'infection duration pdf': 'gauss',
    # Infectious duration mean
    'infectious duration mean': 3,
    # Infectious duration sd
    'infectious duration variance': 5,
    # Infectious duration pdf
    # Available: 'gauss', 'gamma'
    'infectious duration pdf': 'gauss',

    # TODO: Rename to latent period
    # Incubation duration mean
    'incubation duration mean': 2.5,
    # Incubation duration sd
    'incubation duration variance': 2,
    # Incubation duration pdf
    # Available: 'gauss', 'gamma'
    'incubation duration pdf': 'gauss',

    # Hospitalization / death and recovery pdfs
    'hospitalization probability pdf': 'beta',
    'hospitalization probability mean': 0.2,
    'hospitalization probability sd': 0.1,

    # Available: 'gauss', 'gamma'
    'hospitalization duration pdf': 'gauss',
    'hospitalization duration mean': 28,
    'hospitalization duration sd': 5,

    # Available: 'gauss', 'gamma'
    'time until hospitalization pdf': 'gauss',
    'time until hospitalization mean': 5,
    'time until hospitalization sd': 2,

    # Available: 'gauss', 'gamma'
    'time incubation death pdf': 'gauss',
    'time incubation death mean': 32,
    'time incubation death sd': 5,

    # Available: 'gauss', 'gamma'
    'recovery time pdf': 'gauss',
    'recovery time mean': 11,
    'recovery time sd': 5,

    # Mortalitiy rate relative to hospitalization prob
    'mortality prob pdf': 'beta',
    'mortality rate mean': 0.1,
    'mortality rate sd': 0.1,

    # Possible measures to take
    # -'None'
    # -'contact tracing'
    'measures': 'none',
    # fraction of population tracked
    'tracked': 0.2,
    # Interaction intensity distribution:
    # Available: uniform
    'interaction intensity': 'uniform',
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


# TODO: Explain this to idiot Stephan
# Why is a class required?
class ConfigClass(dict):
    """
    class: ConfigClass
    The configuration class. This is used
    by the package for all parameter settings
    Parameters:
        -dic config:
            The config dictionary
    Returns:
        -None
    """
    def __init__(self, *args, **kwargs):
        """
        function: __init__
        initializes the configuration class. This is used
        by the package for all parameter settings
        Parameters:
            -dic config:
                The config dictionary
        Returns:
            -None
        """
        super().__init__(*args, **kwargs)

    # TODO: Update this
    def from_yaml(self, yaml_file: str) -> None:
        """
        function: from_yaml
        Update config with yaml file
        Parameters:
            -str yaml_file:
                path to yaml file
        Returns:
            -None
        """
        yaml_config = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        self.update(yaml_config)

    # TODO: Update this
    def from_dict(self, user_dict: Dict[Any, Any]) -> None:
        """
        function: from_yaml
        Creates a config from dictionary
        Parameters:
            -dic user_dict:
                The user dictionary
        Returns:
            -None
        """
        self.update(user_dict)


config = ConfigClass(_baseconfig)
