# -*- coding: utf-8 -*-

"""
Name: contagion.py
Authors: Christian Haack, Martina Karl, Stephan Meighen-Berger, Dominik Scholz
Main interface to the contagion module.
This package calculates the spread of an infection in
a given population. It allows for the introduction
of safety measures, such as social distancing and tracking.
"""


# Native modules
import sys
import logging
import pickle
import yaml

from copy import deepcopy

import numpy as np  # type: ignore

# -----------------------------------------
# Package modules
from .config import config
from .infection import Infection
from .mc_sim import MC_Sim
from .measures import Measures
from . import population

# unless we put this class in __init__, __name__ will be contagion.contagion
_log = logging.getLogger("contagion")


class Contagion(object):
    """
    Interace to the contagion package. This class
    stores all methods required to run the simulation
    of the infection spread
    Parameters
    ----------
    userconfig: dic, optional
        -User config dictionary

    Returns
    -------
    None: -
        -

    Raises
    ------
    ImportError
        Population file wasn't found
    """

    def __init__(self, userconfig=None):
        """
        Initializes the class Contagion.
        Here all run parameters are set.
        Parameters
        ----------
        userconfig: dic, optional
            -User config dictionary

        Returns
        -------
        None: -
            -

        Raises
        ------
        ImportError
            Population file wasn't found
        """
        # Inputs
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(deepcopy(userconfig))
            else:
                config.from_yaml(userconfig)

        # Create RandomState
        if config["general"]["random state seed"] is None:
            _log.warning("No random state seed given, constructing new state")
            rstate = np.random.RandomState()
        else:
            rstate = np.random.RandomState(
                config["general"]["random state seed"]
            )
        config["runtime"] = {"random state": rstate}

        self.__infected = config["infection"]["infected"]

        # Logger
        # creating file handler with debug messages
        """
        fh = logging.FileHandler(
            config["general"]["log file handler"], mode="w"
        )
        fh.setLevel(logging.WARN)
        """
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(config["general"]["debug level"])

        # Logging formatter
        fmt = "%(levelname)s: %(message)s"
        fmt_with_name = "[%(name)s] " + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        # fh.setFormatter(formatter_with_name)
        # add class name to ch only when debugging
        if config["general"]["debug level"] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)

        # _log.addHandler(fh)
        _log.addHandler(ch)
        _log.setLevel(logging.WARN)
        _log.info("Welcome to contagion!")
        _log.info("This package will help you model the spread of infections")

        def is_same_config(pop_conf_a, pop_conf_b):
            pop_conf_a = dict(pop_conf_a)
            pop_conf_b = dict(pop_conf_b)

            del pop_conf_a["re-use population"]
            del pop_conf_b["re-use population"]
            del pop_conf_a["store population"]
            del pop_conf_b["store population"]

            return pop_conf_a == pop_conf_b

        pop = None
        if config["population"]["re-use population"]:
            try:
                pop, pop_config = pickle.load(
                    open(config["population"]["population storage"], "rb")
                )
                if not is_same_config(pop_config, config["population"]):
                    _log.warn(
                        "Attempting to reuse population with a "
                        "different config. Continue at own risk."
                    )
                _log.debug("Population loaded")
            except (ImportError, FileNotFoundError):
                _log.error("Population file not found!")
        if pop is None:
            _log.info("Starting population construction")
            population_class = getattr(
                population, config["population"]["population class"]
            )
            pop = population_class()
            if config["population"]["store population"]:
                # Storing for later
                _log.debug("Storing for later use")

                pickle.dump(
                    (pop, config["population"]),
                    open(config["population"]["population storage"], "wb"),
                )
        self.pop = pop

        if config["population"]["population class"] == "NetworkXPopulation":
            config["population"]["population size"] = len(self.pop._graph)
        _log.info("Finished the population")

        _log.info("Starting the infection construction")
        self.infection = Infection()
        _log.info("Finished the infection construction")

        _log.info("Starting the measure construction")
        self.measures = Measures()
        _log.info("Finished the measure construction")

        _log.info("Setting the simulation framework")
        self.sim = self.__sim_realistic
        _log.info("Simulation framework set. Please type:")
        _log.info("self.sim(parameters) to run the simulation")

    @property
    def statistics(self):
        """
        Getter functions for the simulation results
        Parameters
        ----------
        None: -
            -

        Returns
        -------
        statistics: dic
            - The simulation results

        Raises
        ------
        -
        """
        return self.__mc_run.statistics

    @property
    def trace_contacts(self):
        """
        function: trace_contacts
        Getter functions for the simulation results
        from the simulation
        Parameters:
            -None
        Returns:
            -list trace_contacts:
                Stores the results from the simulation
        """
        return self.__mc_run.trace_contacts

    @property
    def traced_states(self):
        """
        function: trace_contacts
        Getter functions for the simulation results
        from the simulation
        Parameters:
            -None
        Returns:
            -list trace_contacts:
                Stores the results from the simulation
        """
        return self.__mc_run.traced_states

    @property
    def trace_infection(self):
        """
        function: trace_infection
        Getter functions for the simulation results
        from the simulation
        Parameters:
            -None
        Returns:
            -list trace_infection:
                Stores the results from the simulation
        """
        return self.__mc_run.trace_infection

    @property
    def t(self):
        """
        function: t
        Getter functions for the used time array
        from the simulation
        Parameters:
            -None
        Returns:
            -np.array:
                The used time array
        """
        return self.__mc_run.time_array

    @property
    def R0(self):
        """
        function: R0
        Getter functions for the R0 value
        from the simulation
        Parameters:
            -None
        Returns:
            -float:
                The calculated R0
        """
        return self.__mc_run.R0

    def __sim_realistic(self):
        """
        function: __sim_realistic
        Realistic infection spread simulation
        Parameters:
            -None
        Returns:
            -np.array infected:
                The current population
        """
        _log.debug("Starting MC run")
        self.__mc_run = MC_Sim(self.pop, self.infection, self.measures)
        _log.info("Finished calculation")
        _log.info("The results are stored in a dictionary self.statistics")
        _log.info("Structure of dictionray:")
        _log.info(self.statistics.keys())
        _log.debug(
            "Dumping run settings into %s",
            config["general"]["config location"],
        )
        with open(config["general"]["config location"], "w") as f:
            yaml.dump(config, f)
        _log.debug("Finished dump")
        # Closing log
        logging.shutdown()
