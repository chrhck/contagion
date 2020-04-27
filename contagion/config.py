# -*- coding: utf-8 -*-

"""
Name: config.py
Authors: Christian Haack, Stephan Meighen-Berger
Config file for the contagion package.
It is recommended, that only advanced users change
the settings here.
"""

import logging
from typing import Dict, Any
import yaml

_baseconfig: Dict[str, Any]

_baseconfig = {
    "general": {
        "debug level": logging.WARNING,
        # Location of logging file handler
        "log file handler": "../run/contagion.log",
        # Dump experiment config to this location
        "config location": "../run/config.txt",
        "random state seed": 1337,
        # Trace the infection spread
        "trace spread": False,
        "track graph history": True
    },
    "population": {
        "population size": 1000,
        "re-use population": False,
        # store population
        "store population": False,
        "population storage": "../populations/generic.pkl",
        "population class": "HomogeneousPopulation",
        # Social circle pdf:
        "social circle pdf": {
            "class": "Gamma",
            "mean": 40,
            "sd": 5},
        "social circle interactions pdf": {
            "class": "Gamma",
            "mean": 6,
            "sd": 0.2},


        "hierarchy": [{
            # Close contacts (family)
            "social circle pdf": {
                "class": "Gamma",
                "mean": 2,
                "sd": 2},
            "social circle interactions pdf": {
                "class": "Gamma",
                "mean": 3,
                "sd": 1},
            # A person can only be in one of these circles
            # -> Not interactions between circles of this type
            "interconnectivity": 0,
            "fully_connected": True,
            },
            {
            # Friends
            "social circle pdf": {
                "class": "Gamma",
                "mean": 10,
                "sd": 5},
            "social circle interactions pdf": {
                "class": "Gamma",
                "mean": 5,
                "sd": 2},
            # 30 % of people in this group can be in other groups
            "interconnectivity": 0.3,
            "fully_connected": True,
            },
            {
            # Loose contacts (work, ...)
            "social circle pdf": {
                "class": "Gamma",
                "mean": 20,
                "sd": 5},
            "social circle interactions pdf": {
                "class": "Gamma",
                "mean": 5,
                "sd": 2},
            # 10 % of people in this group can be in other groups
            "interconnectivity": 0.1,
            "fully_connected": True,
            }],

        "random interactions pdf": {
            "class": "Gamma",
            "mean": 1,
            "sd": 1},
        "random interactions intensity pdf": {
            "class": "Gamma",
            "mean": 0.1,
            "sd": 0.5},

        "nx": {
            "func": "lfr_benchmark",
            "kwargs": {
                "tau1": 3,
                "tau2": 1.5,
                "mu": 0.1,
                "average_degree": 7,
                "min_community": 8,
                "max_iters": 200,
            },
            "inter freq pdf": {
                "class": "Gamma",
                "mean": 10,
                "sd": 3
            },
            "intra freq pdf": {
                "class": "Gamma",
                "mean": 3,
                "sd": 3
            }

        }


    },
    "infection": {
        # The number of starting infections
        "infected": 10,
        # Symptom probability
        "will have symptoms prob pdf": {
            "class": "Beta",
            "mean": 0.6,
            "sd": 0.1
        },
        # Infection properties
        "infection probability pdf": {
            "class": "Gamma",
            "mean": 3.,
            "sd": 2.42,
            "max_val": 0.15},
        "infectious duration pdf": {
            "class": "Gamma",
            "mean": 8.,
            "sd": 2.42},
        "latency duration pdf": {
            "class": "Gamma",
            "mean": 6,
            "sd": 3,
            },
        "incubation duration pdf": {
            "class": "Gamma",
            "mean": 7.47522,
            "sd": 4.27014},
        # Hospitalization
        "hospitalization probability pdf": {
            "class": "Beta",
            "mean": 0.1,
            "sd": 0.01},
        "hospitalization duration pdf": {
            "class": "Gamma",
            "mean": 14.,
            "sd": 0.01},
        "time until hospitalization pdf": {
            "class": "Gamma",
            "mean": 2.52,
            "sd": 1.},
        # Mortality
        "time incubation death pdf": {
            "class": "Gamma",
            "mean": 32.,
            "sd": 5.},
        "mortality prob pdf": {
            "class": "Beta",
            "mean": 0.01,
            "sd": 0.01},
        # Recovery
        "recovery time pdf": {
            "class": "Gamma",
            "mean": 11.,
            "sd": 5.},
    },
    "measures": {
        # Measures implemented (None, contact_tracing, social_distancing, all)
        "type": None,
        # fraction of population tracked
        "tracked fraction": 0.2,
        # days of back tracking
        "backtrack length": 0.,
        # duration of the quarantine
        "quarantine duration": 14.0,
        # social distancing
        "distanced fraction": 0.0,
    },
    "scenario": {
        "class": "StandardScenario",
        "sim_length": 200,
    }
}


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
