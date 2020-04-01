# -*- coding: utf-8 -*-

"""
Name: mc_sim.py
Authors: Christian Haack, Martina Karl, Stephan Meighen-Berger, Andrea Turcati
Runs a monte-carlo (random walk) simulation
for the social interactions.
"""
from collections import defaultdict
from sys import exit
from time import time
import logging
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .config import config
from .pdfs import Uniform
from .state_machine import ContagionStateMachine, StatCollector


_log = logging.getLogger(__name__)


class MC_Sim(object):
    """
    class: MC_Sim
    Monte-carlo simulation for the infection spread.
    Parameters:
        -scipy.sparse population:
            The population
        -obj infection:
            The infection object
        -np.array tracked:
            The tracked population
        -dic config:
            The config file
    Returns:
        -None
    "God does not roll dice!"
    """

    def __init__(self, population, infection, tracked):
        """
        function: __init__
        Initializes the class.
        Parameters:
            -scipy.sparse population:
                The population
            -obj infection:
                The infection object
            -np.array tracked:
                The tracked population
        Returns:
            -None
        """
        # Inputs
        self.__infected = config["infected"]
        self.__infect = infection
        self.__dt = config["time step"]
        self.__pop_matrix = population
        self.__t = np.arange(0.0, config["simulation length"], step=self.__dt)

        _log.debug("The interaction intensity pdf")
        if config["interaction intensity"] == "uniform":
            self.__intense_pdf = Uniform(0, 1)
            # The Reproductive Number
            self.__R0 = (
                config["mean social circle interactions"]
                * (config["infectious duration mean"] +
                   config["incubation duration mean"])
                * 0.5
            )
        else:
            _log.error(
                "Unrecognized intensity pdf! Set to " +
                config["interaction intensity"]
            )
            exit("Check the interaction intensity in the config file!")

        # Checking random state
        if config["random state"] is None:
            _log.warning("No random state given, constructing new state")
            self.__rstate = np.random.RandomState()
        else:
            self.__rstate = config["random state"]

        _log.debug("Constructing simulation population")
        _log.debug("The infected ids and durations...")

        self.__pop_size = population.shape[0]

        _log.debug("Constructing the population array")

        self.__population = pd.DataFrame(
            {
                "is_infected": False,
                "is_new_infected": False,
                "is_incubation": False,
                "is_new_incubation": False,
                "is_latent": False,
                "is_new_latent": False,
                "is_infectious": False,
                "is_new_infectious": False,
                "can_infect": False,
                "is_removed": False,
                "is_critical": False,
                "is_hospitalized": False,
                "is_new_hospitalized": False,
                "is_recovering": False,
                "is_new_recovering": False,
                "is_recovered": False,
                "will_die": False,
                "will_die_new": False,
                "will_be_hospitalized": False,
                "will_be_hospitalized_new": False,
                "is_dead": False,
                "is_new_dead": False,
                "incubation_duration": 0,
                "infectious_duration": 0,
                "latent_duration": 0,
                "time_until_hospitalization": 0,
                "hospitalization_duration": 0,
                "recovery_time": 0,
                "time_until_death": 0,

            },
            index=np.arange(self.__pop_size),
        )

        # Choosing the infected
        infect_id = self.__rstate.choice(
            range(self.__pop_size), size=self.__infected, replace=False
        )

        # Their infection duration
        infect_dur = (
            np.around(self.__infect.infectious_duration.rvs(self.__infected))
        )

        # Filling the array
        self.__population.loc[infect_id, "is_infected"] = True
        self.__population.loc[infect_id, "is_infectious"] = True
        # TODO: Add a switch if these people have symptoms or not
        self.__population.loc[infect_id, "can_infect"] = True
        self.__population.loc[infect_id, "infectious_duration"] = infect_dur

        _log.info("There will be %d simulation steps" % len(self.__t))

        # Set tracking
        if tracked is not None:
            _log.debug("Constructiong tracked people ids")
            self.__tracked = True
            tracked_df = pd.DataFrame(
                {"is_tracked": False}, index=np.arange(self.__pop_size)
            )
            tracked_df.loc[tracked, "is_tracked"] = True
            self.__population = pd.concat([self.__population, tracked_df],
                                          axis=1)
        else:
            _log.debug("Population is not tracked")
            self.__tracked = False

        # The storage dictionary
        self.__statistics = defaultdict(list)

        # The statistics of interes
        stat_collector = StatCollector(
            ["is_removed", "is_incubation", "is_latent", "is_infectious",
             "is_infected", "can_infect",
             "is_hospitalized", "is_recovered", "is_dead"])
        # The state machine
        _log.debug("Setting up the state machine")
        self._sm = ContagionStateMachine(
            self.__population,
            stat_collector,
            self.__pop_matrix,
            self.__infect,
            self.__intense_pdf,
            self.__rstate)
        _log.debug("Finished the state machine")
        # Running the simulation
        _log.debug("Launching the simulation")
        start = time()
        self.__simulation()
        end = time()
        _log.debug("Finished the simulation")
        self.__statistics = self._sm.statistics
        _log.info("MC simulation took %f seconds" % (end - start))

    @property
    def statistics(self):
        """
        function: statistics
        Getter functions for the simulation results
        from the simulation
        Parameters:
            -None
        Returns:
            -dic statistics:
                Stores the results from the simulation
        """
        return self.__statistics

    @property
    def time_array(self):
        """
        function: time_array
        Returns the time array used
        Parameters:
            -None
        Returns:
            np.array __t:
                The time array
        """
        return self.__t

    @property
    def R0(self):
        """
        function: reproductive number
        Average number of infections due to
        one patient (not assuming measures were taken)
        Parameters:
            -None
        Returns:
            -float R:
                The reproductive number
        """
        return self.__R0

    @property
    def population(self):
        return self.__population

    def __simulation(self):
        """
        function: __simulation
        Runs the simulation
        Parameters:
            -None
        Returns:
            -None
        """

        start = time()

        for step, _ in enumerate(self.__t):
            self._sm.tick()
            if step % (int(len(self.__t) / 10)) == 0:
                end = time()
                _log.debug("In step %d" % step)
                _log.debug(
                    "Last round of simulations took %f seconds" % (end - start)
                )
                start = time()
