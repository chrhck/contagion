"""
Name: fd_roll_dice_social.py
Authors: Stephan Meighen-Berger, Martina Karl
Runs a monte-carlo (random walk) simulation
for the social interactions.
"""

from sys import exit
import numpy as np
from time import time
from numpy.random import choice
import logging
from scipy import sparse
from .con_config import config
from tqdm.autonotebook import tqdm

from collections import defaultdict
logger = logging.getLogger(__name__)


class CON_mc_sim(object):
    """
    class: CON_mc_sim
    Monte-carlo simulation for the infection spread.
    Parameters:
        -int infected:
            The starting infected population
        -np.array population:
            The population
        -obj infection:
            The infection object
        -np.array tracked:
            The tracked population
        -dic config:
            Dictionary from the config file
    Returns:
        -None
    "God does not roll dice!"
    """

    def __init__(
            self,
            infected,
            population,
            infection,
            tracked,
            rstate=None):
        """
        function: __init__
        Initializes the class.
        Parameters:
            -int infected:
                The starting infected population
            -np.array population:
                The population matrix
            -obj infection:
                The infection object
            -np.array tracked:
                The tracked population
            -obj log:
                The logger
            -dic config:
                Dictionary from the config file

        Returns:
            -None
        """

        self.__infect = infection
        self.__dt = config['time step']
        self.__pop_matrix = population
        self.__t = np.arange(
            0., config['simulation length'],
            step=self.__dt
        )

        logger.debug('The interaction intensity pdf')

        if config['interaction intensity'] == 'uniform':
            self.__intense_pdf = self.__intens_pdf_uniform
            # The Reproductive Number
            self.__R = (
                config['mean social circle interactions'] *
                config['infection duration mean'] * 0.5
            )
        else:
            logger.error('Unrecognized intensity pdf! Set to ' +
                         config['interaction intensity'])
            exit('Check the interaction intensity in the config file!')

        if rstate is None:
            logger.warning("No random state given, constructing new state")
            rstate = np.random.RandomState()
        self.__rstate = rstate

        logger.debug('Constructing simulation population')
        logger.debug('The infected ids and durations...')

        self.pop_size = population.shape[0]

        infect_id = self.__rstate.choice(
            range(self.pop_size),
            size=infected,
            replace=False)
        infect_dur = np.around(
            self.__infect.pdf_duration(infected))
        # Constructing population array
        # Every individual has 5 components
        #   -individual's id
        #   -infected
        #   -remaining duration of infection
        #   -immune
        logger.debug('Filling the population array')

        self.__population = np.empty((self.pop_size, 4))
        self.__population[:, 0] = np.arange(self.pop_size)
        self.__population[:, 1] = 0
        self.__population[:, 2] = 0
        self.__population[:, 3] = 0

        # Adding the infected
        self.__population[infect_id, 1] = 1
        self.__population[infect_id, 2] = infect_dur

        logger.info('There will be %d simulation steps' %len(self.__t))
        # Removing social mobility of tracked people
        if tracked is not None:
            # TODO make this configurable

            self.__pop_matrix = self.__pop_matrix.tolil()
            self.__pop_matrix[tracked] = 0

        if config['save population']:
            logger.debug("Saving the distribution of infected")
            self.__distribution = []
            self.__total_infections = []
            self.__new_infections = []
            self.__immune = []

        self.statistics = defaultdict(list)
        # Running the simulation
        start = time()
        self.__simulation()
        end = time()
        logger.info('MC simulation took %f seconds' % (end-start))

    @property
    def population(self):
        """
        function: population
        Returns the population
        Parameters:
            -None
        Returns:
            -np.array population:
                The current population
        """
        return self.__population

    @property
    def distribution(self):
        """
        function: distribution
        Returns the distribution
        Parameters:
            -None
        Returns:
            -np.array distribution:
                The distribution
        """
        return self.__distribution

    @property
    def infections(self):
        """
        function: infections
        Returns the infections
        Parameters:
            -None
        Returns:
            -np.array infections:
                The total infections
        """
        return np.array(self.__total_infections)

    @property
    def new_infections(self):
        """
        function: new_infections
        Returns the new_infections
        Parameters:
            -None
        Returns:
            -np.array new_infections:
                The total new_infections
        """
        return np.array(self.__new_infections)

    @property
    def immune(self):
        """
        function: immune
        Returns the immune
        Parameters:
            -None
        Returns:
            -np.array immune:
                The total immune
        """
        return np.array(self.__immune)

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
    def R(self):
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
        return self.__R

    def __simulation(self):
        """
        function: __simulation
        Runs the simulation
        Parameters:
            -None
        Returns:
            -None
        """
        self.__infections = []

        self.statistics = defaultdict(list)

        pop_csr = self.__pop_matrix.tocsr()

        for _ in tqdm(self.__t, total=len(self.__t)):

            infected_mask = self.__population[:, 1] == 1
            infected_indices = np.nonzero(infected_mask)[0]

            contact_rows, contact_cols, contact_strengths =\
                sparse.find(pop_csr[infected_indices])

            successful_contacts_mask = self.__rstate.poisson(
                contact_strengths) >= 1

            # we are just interested in the columns, ie. only the id's 
            # of the contacted people

            successful_contacts_indices = contact_cols[successful_contacts_mask]

            # successful_contacts_indices = np.unique(successful_contacts_indices)

            num_succesful_contacts = len(successful_contacts_indices)

            self.statistics["succesful_contacts"].append(
                num_succesful_contacts)

            contact_strength = self.__intense_pdf(num_succesful_contacts)
            infection_prob = self.__infect.pdf(contact_strength)

            newly_infected_mask = self.__rstate.binomial(1, infection_prob)
            newly_infected_mask = np.asarray(newly_infected_mask, bool)

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

            self.statistics["succesful_infections"].append(
                len(newly_infected_indices))

            # adjusting infection duration

            self.__population[infected_indices, 2] -= 1

            recovered_indices = infected_indices[self.__population[infected_indices, 2] <= 0]
            # Set recovered
            self.__population[recovered_indices, 1] = 0
            # Set immune
            self.__population[recovered_indices, 3] = 1

            self.statistics["recovered"].append(len(recovered_indices))

            # add new infections

            tmp_dur = np.around(
                self.__infect.pdf_duration(num_newly_infected))

            self.__population[newly_infected_indices, 1] = 1
            self.__population[newly_infected_indices, 2] = tmp_dur

            self.statistics["immune"].append(
                np.sum(self.__population[:, 3] == 1))

            self.statistics["infected"].append(
                np.sum(self.__population[:, 1] == 1))

            self.statistics["healthy"].append(
                np.sum(
                    (self.__population[:, 1] == 0) &
                    (self.__population[:, 3] == 0) )
                )

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
