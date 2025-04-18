# -*- coding: utf-8 -*-

"""
Name: config.py
Authors: Christian Haack, Stephan Meighen-Berger, Andrea Turcati
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
        "track graph history": True,
        "trace states": False
    },
    "population": {
        "population size": 10000,
        "re-use population": False,
        # store population
        "store population": False,
        "population storage": "../populations/generic.pkl",
        "population class": "HomogeneousPopulation",
        # Social circle pdf:
        "social circle pdf": {"class": "Gamma", "mean": 10, "sd": 1},
        "social circle interactions pdf": {
            "class": "Gamma",
            "mean": 10,
            "sd": 1,
        },
        "random interactions pdf": {
            "class": "Gamma",
            "mean": 0.0001,
            "sd": 0.001,
        },
        "random interactions intensity pdf": {
            "class": "Gamma",
            "mean": 0.1,
            "sd": 0.5,
        },
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
            "inter freq pdf": {"class": "Gamma", "mean": 10, "sd": 3},
            "intra freq pdf": {"class": "Gamma", "mean": 3, "sd": 3},
        },
    },
    "infection": {
        # The number of starting infections
        "infected": 1,
        # Symptom probability
        "will have symptoms prob pdf": {
            "class": "Beta",
            "mean": 0.5,
            "sd": 0.1,
        },
        # Infection properties
        "infection probability pdf": {
            "class": "Gamma",
            "mean": 3.0,
            "sd": 2.42,
            "max_val": 0.25,
        },
        "infectious duration pdf": {"class": "Gamma", "mean": 8.0, "sd": 2.42},
        "latency duration pdf": {"class": "Gamma", "mean": 6, "sd": 3},
        "incubation duration pdf": {
            "class": "Gamma",
            "mean": 7.47522,
            "sd": 4.27014,
        },
        # Hospitalization
        "hospitalization probability pdf": {
            "class": "Beta",
            "mean": 0.1,
            "sd": 0.01,
        },
        "hospitalization duration pdf": {
            "class": "Gamma",
            "mean": 14.0,
            "sd": 0.01,
        },
        "time until hospitalization pdf": {
            "class": "Gamma",
            "mean": 2.52,
            "sd": 1.0,
        },
        # Mortality
        "time incubation death pdf": {
            "class": "Gamma",
            "mean": 32.0,
            "sd": 5.0,
        },
        "mortality prob pdf": {"class": "Beta", "mean": 0.01, "sd": 0.01},
        # Recovery
        "recovery time pdf": {"class": "Gamma", "mean": 11.0, "sd": 5.0},
    },
    "measures": {
        # Measures implemented (True, False)
        "contact tracing": False,
        "population tracking": False,
        # fraction of population tracked
        "backward tracing": True,
        "tracked fraction": 1.0,
        # Second order Tracing (True, False)
        "second order": False,
        # days of back tracking
        "backtrack length": 0.0,
        # track uninfected (True, False)
        "track uninfected": False,
        "tracing efficiency": 1.0,
        "fill backtrace random": True,
        # quarantine (True, False)
        "quarantine": False,
        # report symptomatic (True, False)
        "report symptomatic": True,
        # duration of the quarantine
        "quarantine duration": 14.0,
        # testing (True, False)
        "testing": False,
        # Time until testing
        "time until test": 1.0,
        # Time until test results
        "time until result": 1.0,
        "test true positive rate": 0.9,
        "time until second test": 5,
        "time until second test result": 0,
        "test false positive rate": 0.01,
        "test threshold": 0.01,
        "app fraction": 1,
        "random test num": 0,
        "rnd testing": False,
        "random testing mode": "lin weight",
        "test capacity": None
    },
    "scenario": {"class": "StandardScenario", "sim_length": 200},
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
