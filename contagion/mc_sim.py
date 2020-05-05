# -*- coding: utf-8 -*-

"""
Name: mc_sim.py
Authors: Christian Haack, Martina Karl, Stephan Meighen-Berger, Andrea Turcati
Runs a monte-carlo (random walk) simulation
for the social interactions.
"""
from collections import defaultdict

from time import time
import logging
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .config import config
from .state_machine import ContagionStateMachine, StatCollector
from . import scenario


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

    def __init__(self, population, infection, measures):
        """
        function: __init__
        Initializes the class.
        Parameters:
            -Population population:
                The population
            -obj infection:
                The infection object
            -obj measures:
                The measures object
        Returns:
            -None
        """
        # Inputs
        self.__infected = config["infection"]["infected"]
        self.__infect = infection
        self.__pop = population
        self.__measures = measures

        self.__rstate = config["runtime"]["random state"]

        self.__pop_size = config["population"]["population size"]

        _log.debug("Constructing the population array")

        # TODO: doesn't actually have to be a DataFrame anymore
        self.__population = pd.DataFrame(
            {
                "is_infected": False,
                "is_new_infected": False,

                "is_latent": False,
                "is_new_latent": False,
                "is_infectious": False,
                "is_new_infectious": False,

                "will_have_symptoms": False,
                "will_have_symptoms_new": False,
                "is_symptomatic": False,
                "is_symptomatic_new": False,

                "is_removed": False,

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
                "is_quarantined": False,
                "is_new_quarantined": False,

                "is_tracked": False,

                "infectious_duration": 0,
                "latent_duration": 0,
                "time_until_hospitalization": 0,
                "time_since_infectious": -np.inf,
                "hospitalization_duration": 0,
                "recovery_time": 0,
                "time_until_death": 0,
                "time_until_symptoms": 0,
                "quarantine_duration": 0,

            },
            index=np.arange(self.__pop_size),
        )

        # Choosing the infected
        infect_id = self.__rstate.choice(
            range(self.__pop_size), size=self.__infected, replace=False
        )

        if hasattr(self.__pop, "_graph"):
            g = self.__pop._graph
            for inf_id in infect_id:
                g.nodes[inf_id]["initial_infected"] = True

        # Their infection duration
        infec_dur = np.around(
            self.__infect.infectious_duration.rvs(self.__infected)
        )

        # Filling the array
        self.__population.loc[infect_id, "is_infected"] = True
        self.__population.loc[infect_id, "is_infectious"] = True
        self.__population.loc[infect_id, "is_new_infectious"] = True
        self.__population.loc[infect_id, "infectious_duration"] = infec_dur
        # TODO: Add a switch if these people have symptoms or not

        # Set Contact Tracing
        tracked = self.__measures.tracked
        if tracked is not None:
            _log.debug("Constructing tracked people ids")
            self.__population.loc[tracked, "is_tracked"] = True
        else:
            _log.debug("Population is not tracked")

        # The storage dictionary
        self.__statistics = defaultdict(list)

        # The statistics of interest
        stat_collector = StatCollector(
            [
                "is_removed",
                "is_latent",
                "is_infectious",
                "is_infected",
                "is_hospitalized",
                "is_recovered",
                "is_recovering",
                "is_dead",
                "is_quarantined",
                "is_symptomatic",
            ]
        )
        # The state machine
        _log.debug("Setting up the state machine")
        self._sm = ContagionStateMachine(
            self.__population,
            stat_collector,
            self.__pop,
            self.__infect,
            self.__measures,
        )

        _log.debug("Setting simulation scenario")

        scenario_conf = dict(config["scenario"])
        scenario_class = scenario_conf.pop("class")

        self._scenario = getattr(scenario, scenario_class)(
            state_machine=self._sm, **scenario_conf)

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
    def trace_contacts(self):
        """
        function: trace_contacts
        Getter functions for the simulation results
        from the simulation
        Parameters:
            -None
        Returns:
            -list trace_contacts:
                Stores the spread
        """
        return self._sm.trace_contacts

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
                Stores the spread
        """
        return self._sm.trace_infection

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
        return np.arange(self._scenario._sim_length)

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

        self._scenario.run()
