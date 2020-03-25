"""
Name: con_measures.py
Authors: Stephan Meighen-Berger
The different measures one can take
to suppress the spread.
"""
# imports
from sys import exit
import numpy as np


class CON_measures(object):
    """
    class: CON_measures
    Class to implement different possible
    containment measures.
    Parameters:
        -obj log:
            The logger
        -dic config:
            The configuration of the module
    Returns:
        -None
    """

    def __init__(
        self, log, config,
    ):
        """
        function: __init__
        Initializes the class
        Parameters:
            -obj log:
                The logger
            -dic config:
                The configuration of the module
        Returns:
            -None
        """
        self.__log = log
        self.__config = config
        if self.__config["measures"] == "none":
            self.__log.info("No measure taken")
            self.__tracked = None
            self.__distanced = None
        elif "contact tracing" in self.__config["measures"]:
            self.__log.info("Using contact tracing")
            self.__contact_tracing()
            self.__distanced = None
        elif "social distancing" in self.__config["measures"]:
            self.__log.info("Using social distancing")
            self.__social_distancing()
            self.__tracked = None
        elif "all" in self.__config["measures"]:
            self.__log.info("Using social distancing")
            self.__log.info("Using contact tracing")
            self.__contact_tracing()
            self.__social_distancing()
        else:
            self.__log.error(
                "measure not implemented! Set to " + self.__config["measures"]
            )
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
        tracked_pop = int(self.__config["population size"] * self.__config["tracked"])
        self.__log.debug("Number of people tracked is %d" % tracked_pop)
        self.__tracked = np.random.choice(
            range(self.__config["population size"]), size=tracked_pop, replace=False
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
        distanced_pop = int(
            self.__config["population size"] * self.__config["distanced"]
        )
        self.__log.debug("Number of people social distancing is %d" % distanced_pop)
        self.__distanced = np.random.choice(
            range(self.__config["population size"]), size=distanced_pop, replace=False
        ).flatten()
