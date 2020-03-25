"""
Name: fd_roll_dice_social.py
Authors: Christian Haack, Martina Karl, Stephan Meighen-Berger
Runs a monte-carlo (random walk) simulation
for the social interactions.
"""

from sys import exit
import numpy as np
from time import time
from numpy.random import choice
from scipy import sparse
from collections import defaultdict


class CON_mc_sim(object):
    """
    class: CON_mc_sim
    Monte-carlo simulation for the infection spread.
    Parameters:
        -scipy.sparse population:
            The population
        -obj infection:
            The infection object
        -np.array tracked:
            The tracked population
        -obj log:
            The logger
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
            tracked,
            log,
            config
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
            -obj log:
                The logger
            -dic config:
                The config file
        Returns:
            -None
        """
        # Inputs
        self.__log = log.getChild(self.__class__.__name__)
        self.__config = config
        self.__infected = self.__config['infected']
        self.__infect = infection
        self.__dt = config['time step']
        self.__pop_matrix = population
        self.__t = np.arange(
            0., self.__config['simulation length'],
            step=self.__dt
        )

        self.__log.debug('The interaction intensity pdf')
        if self.__config['interaction intensity'] == 'uniform':
            self.__intense_pdf = self.__intens_pdf_uniform
            # The Reproductive Number
            self.__R0 = (
                self.__config['mean social circle interactions'] *
                self.__config['infection duration mean'] * 0.5
            )
        else:
            self.__log.error('Unrecognized intensity pdf! Set to ' +
                             self.__config['interaction intensity'])
            exit('Check the interaction intensity in the config file!')

        # Checking random state
        if self.__config['random state'] is None:
            self.__log.warning("No random state given, constructing new state")
            self.__rstate = np.random.RandomState()
        else:
            self.__rstate = self.__config['random state']

        self.__log.debug('Constructing simulation population')
        self.__log.debug('The infected ids and durations...')

        self.__pop_size = population.shape[0]

        self.__log.debug('Constructing the population array')
        # 4 components
        #   - The population
        #   - Infected?
        #   - The infection duration
        self.__population = np.empty((self.__pop_size, 4))
        self.__population[:, 0] = np.arange(self.__pop_size)
        self.__population[:, 1] = 0
        self.__population[:, 2] = 0
        self.__population[:, 3] = 0

        # Choosing the infected
        infect_id = self.__rstate.choice(
            range(self.__pop_size),
            size=self.__infected,
            replace=False)

        # Their infection duration
        infect_dur = np.around(
            self.__infect.pdf_duration(self.__infected)
        )

        # Filling the array
        self.__population[infect_id, 1] = 1
        self.__population[infect_id, 2] = infect_dur

        self.__log.info('There will be %d simulation steps' %len(self.__t))
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
        self.__log.info('MC simulation took %f seconds' % (end-start))

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

            infected_mask = self.__population[:, 1] == 1
            infected_indices = np.nonzero(infected_mask)[0]

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
            infection_prob = self.__infect.pdf(contact_strength)

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
            already_infected = self.__population[newly_infected_indices, 1] == 1
            already_immune = self.__population[newly_infected_indices, 3] == 1

            newly_infected_indices = newly_infected_indices[
                ~(already_infected | already_immune)]

            num_newly_infected = len(newly_infected_indices)
            # Store new infections
            self.__statistics["infections"].append(
                len(newly_infected_indices)
            )

            # adjusting infection duration
            self.__population[infected_indices, 2] -= 1

            recovered_indices = infected_indices[self.__population[infected_indices, 2] <= 0]
            # Set recovered
            self.__population[recovered_indices, 1] = 0
            # Set immune
            self.__population[recovered_indices, 3] = 1

            # Storing recovered
            self.__statistics["recovered"].append(len(recovered_indices))
            print(len(recovered_indices))
            # add new infections
            tmp_dur = np.around(
                self.__infect.pdf_duration(num_newly_infected))
            self.__population[newly_infected_indices, 1] = 1
            self.__population[newly_infected_indices, 2] = tmp_dur

            # Storing immune
            self.__statistics["immune"].append(
                np.sum(self.__population[:, 3] == 1))

            self.__statistics["infectious"].append(
                np.sum(self.__population[:, 1] == 1))

            self.__statistics["susceptible"].append(
                np.sum(
                    (self.__population[:, 1] == 0) &
                    (self.__population[:, 3] == 0) )
                )
            if step % (int(len(self.__t)/10)) == 0:
                end = time()
                self.__log.debug('In step %d' %step)
                self.__log.debug(
                    'Last round of simulations took %f seconds' %(end-start)
                )
                start = time()

    def __intens_pdf_uniform(self, contacts):
        """
        function: __intens_pdf_uniform
        The social interaction intensity
        drawn from a uniform distribution
        Parameters:
            -int contacts:
                Number of contacts
        Returns:
            -np.array res:
                The contact intensities
        """
        return self.__rstate.uniform(low=0., high=1., size=contacts)
