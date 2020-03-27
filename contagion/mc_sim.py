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
import numpy as np
import pandas as pd
from scipy import sparse

from .config import config
from .pdfs import Uniform

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

    def __init__(
            self,
            population,
            infection,
            tracked
            ):
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
        self.__infected = config['infected']
        self.__infect = infection
        self.__dt = config['time step']
        self.__pop_matrix = population
        self.__t = np.arange(
            0., config['simulation length'],
            step=self.__dt
        )

        _log.debug('The interaction intensity pdf')
        if config['interaction intensity'] == 'uniform':
            self.__intense_pdf = Uniform(0, 1).rvs
            # The Reproductive Number
            self.__R0 = (
                config['mean social circle interactions'] *
                config['infection duration mean'] * 0.5
            )
        else:
            _log.error('Unrecognized intensity pdf! Set to ' +
                             config['interaction intensity'])
            exit('Check the interaction intensity in the config file!')

        # Checking random state
        if config['random state'] is None:
            _log.warning("No random state given, constructing new state")
            self.__rstate = np.random.RandomState()
        else:
            self.__rstate = config['random state']

        _log.debug('Constructing simulation population')
        _log.debug('The infected ids and durations...')

        self.__pop_size = population.shape[0]

        _log.debug('Constructing the population array')

        self.__population = pd.DataFrame(
            {"is_infected": False,
             "in_incubation": False,
             "is_infectious": False,
             "incubation_duration": 0,
             "infectious_duration": 0,
             "is_removed": False,
             "is_critical": False,
             "is_hospitalized": False,
             "is_recovering": False,
             "time_until_hospitalization": 0,
             "hospitalization_duration": 0,
             "recovery_time": 0,
             "has_recovered": 0,
             "time_until_death": 0,
             "will_die": False,
             "will_be_hospitalized": False,
             "has_died": False},
            index=np.arange(self.__pop_size))

        # Choosing the infected
        infect_id = self.__rstate.choice(
            range(self.__pop_size),
            size=self.__infected,
            replace=False)

        # Their infection duration
        infect_dur = np.around(
            self.__infect.infectious_duration(self.__infected)
        )

        # Filling the array
        self.__population.loc[infect_id, "is_infected"] = True
        self.__population.loc[infect_id, "is_infectious"] = True
        self.__population.loc[infect_id, "infectious_duration"] = infect_dur

        _log.info('There will be %d simulation steps' %len(self.__t))
        # Removing social mobility of tracked people
        if tracked is not None:
            # TODO make this configurable
            # The current implementation disables all contacts
            # of tracked persons

            self.__pop_matrix = self.__pop_matrix.tolil()
            self.__pop_matrix[tracked] = 0

        # Some additional storage
        self.__distribution = []
        self.__total_infections = []
        self.__new_infections = []
        self.__immune = []

        # The storage dictionary
        self.__statistics = defaultdict(list)
        # Running the simulation
        start = time()
        self.__simulation()
        end = time()
        _log.info('MC simulation took %f seconds' % (end-start))

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

    def __simulation(self):
        """
        function: __simulation
        Runs the simulation
        Parameters:
            -None
        Returns:
            -None
        """
        pop_csr = self.__pop_matrix.tocsr()

        start = time()
        for step, _ in enumerate(self.__t):


            """
            New infections
            """

            infected_mask = self.__population.loc[:, "is_infectious"]
            infected_indices = self.__population.index[infected_mask]

            # Find all non-zero connections of the infected
            # rows are the ids / indices of the infected
            # columns are the people they have contact with

            _, contact_cols, contact_strengths =\
                sparse.find(pop_csr[infected_indices])

            # Based on the contact rate, sample a poisson rvs
            # for the number of interactions per timestep.
            # A contact is sucessful if the rv is > 1, ie.
            # more than one contact per timestep
            successful_contacts_mask = self.__rstate.poisson(
                contact_strengths) >= 1

            # we are just interested in the columns, ie. only the
            # ids of the people contacted by the infected.
            # Note, that contacted ids can appear multiple times
            # if a person is successfully contacted by multiple people.
            successful_contacts_indices = contact_cols[successful_contacts_mask]
            num_succesful_contacts = len(successful_contacts_indices)

            self.__statistics["contacts"].append(
                num_succesful_contacts)

            # Calculate infection probability for all contacts
            contact_strength = self.__intense_pdf(num_succesful_contacts)
            infection_prob = self.__infect.pdf_infection_prob(contact_strength)

            # An infection is successful if the bernoulli outcome
            # based on the infection probability is 1

            newly_infected_mask = self.__rstate.binomial(1, infection_prob)
            newly_infected_mask = np.asarray(newly_infected_mask, bool)

            # Get the indices for the newly infected
            newly_infected_indices = successful_contacts_indices[
                newly_infected_mask]

            # There might be multiple successfull infections per person 
            # from different infected people
            newly_infected_indices = np.unique(newly_infected_indices)

            # check if people are already infected or aleady immune
            already_infected = (
                (self.__population.loc[
                    newly_infected_indices, "in_incubation"] == True) |
                (self.__population.loc[
                    newly_infected_indices, "is_infectious"] == True)
                )

            already_immune = self.__population.loc[
                    newly_infected_indices, "is_removed"] == True

            newly_infected_indices = self.__population.index[
                newly_infected_indices[~(already_infected | already_immune)]]

            num_newly_infected = len(newly_infected_indices)

            """
            For newly infected determine whether they will be hospitalized
            and die
            """

            will_be_hospitalized_prob = self.__infect.hospitalization_prob(
                num_newly_infected)

            # roll the dice
            will_be_hospitalized = self.__rstate.binomial(
                1, will_be_hospitalized_prob, size=num_newly_infected) == 1

            num_hospitalized = np.sum(will_be_hospitalized)

            will_be_hospitalized_indices = newly_infected_indices[
                will_be_hospitalized]

            time_until_hospit = self.__infect.time_until_hospitalization(
                num_hospitalized)

            # Same for mortality

            will_die_prob = self.__infect.death_prob(
                num_hospitalized)

            will_die = self.__rstate.binomial(
                1, will_die_prob, size=num_hospitalized) == 1

            will_die_indices = will_be_hospitalized_indices[will_die]
            num_will_die = np.sum(will_die)

            # Time until death is relative to end of incubation perdiod
            # Thus add after calculating incubation time
            time_until_death = self.__infect.time_incubation_death(
                num_will_die)

            # Add info to dataframe

            self.__population.loc[
                will_be_hospitalized_indices,
                "will_be_hospitalized"] = True

            self.__population.loc[
                will_be_hospitalized_indices,
                "time_until_hospitalization"] = time_until_hospit

            self.__population.loc[
                will_die_indices,
                "will_die"] = True


            """
            Status updates
            """

            """
            Incubation

            First update incubation duration of old cases. Then add new cases.
            Finally check cases that passes incubation peroid
            """
            # Old cases
            in_incubation_mask = self.__population.loc[:, "in_incubation"]

            # adjusting incubation duration
            self.__population.loc[in_incubation_mask, "incubation_duration"] -= 1

            # add new incubations
            tmp_dur = np.around(
                self.__infect.incubation_duration(num_newly_infected))

            self.__population.loc[newly_infected_indices, "in_incubation"] = True
            self.__population.loc[newly_infected_indices, "is_infected"] = True
            self.__population.loc[newly_infected_indices, "incubation_duration"] = tmp_dur

            # For those who will day, calculate time until death

            self.__population.loc[will_die_indices, "time_until_death"] = (
                self.__population.loc[will_die_indices, "incubation_duration"] 
                + time_until_death)

            # Change state to infected if passed incubration duration
            # Check only old cases
            passed_incubation_index = self.__population.loc[in_incubation_mask].index
            passed_incubation = passed_incubation_index[
                self.__population.loc[
                    in_incubation_mask, "incubation_duration"] <= 0
                ]

            num_newly_infectious = len(passed_incubation)
            if num_newly_infectious > 0:
                self.__population.loc[passed_incubation, "in_incubation"] = False



            """
            Death

            TODO: Remove on death / hospitalization
            """

            # print(self.__population.index[(self.__population["is_infected"] == False) & self.__population["is_hospitalized"] == True])

            will_die = self.__population.loc[:, "will_die"] == True
            self.__population.loc[will_die, "time_until_death"] -= 1

            will_die_indices = self.__population.loc[will_die].index

            has_died = self.__population.loc[will_die_indices, "time_until_death"] <= 0
            has_died_indices = will_die_indices[has_died]
            num_has_died = len(has_died_indices)

            if num_has_died > 0:
                self.__population.loc[has_died_indices, "has_died"] = True
                self.__population.loc[has_died_indices, "is_hospitalized"] = False
                self.__population.loc[has_died_indices, "will_die"] = False
                self.__population.loc[has_died_indices, "is_infectious"] = False
                self.__population.loc[has_died_indices, "is_infected"] = False

            """
            Hospitalization recovery
            """

            def where_col(df, col, cond, other):
                df[col].where(cond, other, axis=0, inplace=True)
                return df

            (
                self.__population["hospitalization_duration"]
                .where(~self.__population["is_hospitalized"]==True,
                       self.__population["hospitalization_duration"]-1,
                       inplace=True)
            )

            cond = ~(
                (self.__population["hospitalization_duration"] <= 0) &
                self.__population["is_hospitalized"]==True)

            self.__population = (
                self.__population                
                .pipe(where_col, "is_infected", cond, False)
                .pipe(where_col, "has_recovered", cond, True)
                .pipe(where_col, "is_hospitalized", cond, False)
            )

            """
            Hospitalization
            """

            will_be_hospitalized = self.__population.loc[:, "will_be_hospitalized"] == True
            self.__population.loc[will_be_hospitalized, "time_until_hospitalization"] -= 1

            will_be_hospitalized_indices = self.__population.loc[will_be_hospitalized].index

            is_hospitalized = self.__population.loc[will_be_hospitalized_indices, "time_until_hospitalization"] <= 0
            is_hospitalized_indices = will_be_hospitalized_indices[is_hospitalized]

            num_is_hospitalized = len(is_hospitalized_indices)

            if num_is_hospitalized > 0:
                self.__population.loc[is_hospitalized_indices, "is_hospitalized"] = True
                self.__population.loc[is_hospitalized_indices, "will_be_hospitalized"] = False
                # TODO: could reduce contact rate instead??
                self.__population.loc[is_hospitalized_indices, "is_infectious"] = False

                hostpit_dur = self.__infect.hospitalization_duration(num_is_hospitalized)
                #print(hostpit_dur)
                self.__population.loc[is_hospitalized_indices, "hospitalization_duration"] = hostpit_dur


            """
            Recovery
            """

            is_recovering_mask = self.__population.loc[:, "is_recovering"] == True
            is_recovering_indices = self.__population.index[is_recovering_mask]

            self.__population.loc[is_recovering_indices, "recovery_time"] -= 1

            has_recovered_mask = self.__population.loc[is_recovering_indices, "recovery_time"] <= 0
            has_recovered_indices = is_recovering_indices[has_recovered_mask]

            num_newly_recovered = len(has_recovered_indices)


            if num_newly_recovered > 0:
                self.__population.loc[has_recovered_indices, "is_recovering"] = False
                self.__population.loc[has_recovered_indices, "has_recovered"] = True
                self.__population.loc[has_recovered_indices, "is_infected"] = False
           


            """
            Infectious

            First update infectious duration of old cases. Then add new cases.
            Finally check cases that passes infectious peroid
            """

            # Number of people who became infectious this timestep

            # Old cases
            is_infectious_mask = self.__population.loc[:, "is_infectious"]

            # adjusting infectious duration
            self.__population.loc[is_infectious_mask, "infectious_duration"] -= 1

            # add new infectious
            tmp_dur = np.around(
                self.__infect.infectious_duration(num_newly_infectious))

            if len(passed_incubation) > 0:
                self.__population.loc[passed_incubation, "is_infectious"] = True
                self.__population.loc[passed_incubation, "infectious_duration"] = tmp_dur

            # Change state to removed if passed infectious duration
            # Check only old cases
            passed_infectious_index = self.__population.loc[is_infectious_mask].index
            passed_infectious = passed_infectious_index[
                self.__population.loc[
                    is_infectious_mask, "infectious_duration"] <= 0
                ]

            num_newly_removed = len(passed_infectious)
            if num_newly_removed > 0:
                self.__population.loc[passed_infectious, "is_infectious"] = False
                self.__population.loc[passed_infectious, "is_removed"] = True
                recovery_time = self.__infect.recovery_time(
                    num_newly_removed)

                self.__population.loc[passed_infectious, "recovery_time"] = recovery_time
                self.__population.loc[passed_infectious, "is_recovering"] = True

            # Storing statistics
            is_removed = self.__population.loc[:, "is_removed"]
            self.__statistics["removed"].append(is_removed.sum(axis=0))
            in_incubation = self.__population.loc[:, "in_incubation"]
            self.__statistics["incubation"].append(in_incubation.sum(axis=0))
            is_infectious = self.__population.loc[:, "is_infectious"]
            self.__statistics["infectious"].append(is_infectious.sum(axis=0))
            has_recovered = self.__population.loc[:, "has_recovered"]
            self.__statistics["recovered"].append(has_recovered.sum(axis=0))
            is_infected = self.__population.loc[:, "is_infected"]
            self.__statistics["infected"].append(is_infected.sum(axis=0))

            is_hospitalized = self.__population.loc[:, "is_hospitalized"]
            self.__statistics["hospitalized"].append(is_hospitalized.sum(axis=0))

            self.__statistics["new infections"].append(num_newly_infected)
            self.__statistics["newly infectious"].append(num_newly_infectious)
            self.__statistics["newly recovered"].append(num_newly_recovered)
            self.__statistics["newly removed"].append(num_newly_removed)
            self.__statistics["will be hospitalized"].append(num_hospitalized)
            self.__statistics["will die"].append(num_will_die)
            self.__statistics["new deaths"].append(num_has_died)
            is_dead = self.__population.loc[:, "has_died"]
            self.__statistics["total_deaths"].append(is_dead.sum(axis=0))
            if step % (int(len(self.__t)/10)) == 0:
                end = time()
                _log.debug('In step %d' %step)
                _log.debug(
                    'Last round of simulations took %f seconds' %(end-start)
                )
                start = time()
