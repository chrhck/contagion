# -*- coding: utf-8 -*-

"""
Name: measures.py
Authors: Stephan Meighen-Berger
The different measures one can take
to suppress the spread.
"""

# imports
from sys import exit
import numpy as np
import logging

from .config import config

_log = logging.getLogger(__name__)


class Measures(object):
    """
    class: Measures
    Class to implement different possible
    containment measures.
    Parameters:
        -None
    Returns:
        -None
    """

    def __init__(self):
        """
        function: __init__
        Initializes the class
        Parameters:
            -None
        Returns:
            -None
        """
        if config["measures"] == "none":
            _log.info("No measure taken")
            self.__tracked = None
        elif config["measures"] == "contact tracing":
            _log.info("Using contact tracing")
            self.__contact_tracing()
            self.__distanced = None
        elif config["measures"] == "social sidtancing":
            _log.info("Using social distancing")
            self.__social_distancing()
            self.__tracked = None
        elif config["measures"] == "all":
            _log.info("Using social distancing")
            _log.info("Using contact tracing")
            self.__contact_tracing()
            self.__social_distancing()

        else:
            _log.error("measure not implemented! Set to " + config["measures"])
            exit("Please check the config file what measures are allowed")

    @property
    def tracked(self):
        """
        function: tracked
        Getter function for the tracked population
        Parameters:
            -None
        Returns:
            -np.array tracked:
                The ids of the tracked population
        """
        return self.__tracked

<<<<<<< Updated upstream
=======
    @property
    def distanced(self):
        """
        function: distanced
        Getter function for the distanced population
        Parameters:
            -None
        Returns:
            -np.array distanced:
                The ids of the distanced population
        """
        return self.__distanced

>>>>>>> Stashed changes
    # TODO: Not 100% of participants will report correctly
    def __contact_tracing(self):
        """
        function: __contact_tracing
        Implements the measure contact tracing
        Parameters:
            -None
        Returns:
            -None
        """
        tracked_pop = int(config["population size"] * config["tracked"])
        _log.debug("Number of people tracked is %d" % tracked_pop)
        self.__tracked = np.random.choice(
            range(config["population size"]), size=tracked_pop, replace=False
        ).flatten()

    def __social_distancing(self):
        """
        function: __social_distancing
        Implements the measure social distancing
        Parameters:
            -None
        Returns:
            -None
        """
        distanced_pop = int(config["population size"] * config["distanced"])
        _log.debug("Number of people social distancing is %d" % distanced_pop)
        self.__distanced = np.random.choice(
            range(self.__config["population size"]), size=distanced_pop, replace=False
        ).flatten()
