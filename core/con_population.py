"""
Name: con_population.py
Authors: Stephan Meighen-Berger
Constructs the population.
"""


import sys
import traceback
import numpy as np
import scipy.stats
from scipy.stats import truncnorm
import scipy.sparse as sparse
from tqdm.autonotebook import tqdm
from .con_config import config

import logging
import warnings

"""
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
"""

logger = logging.getLogger(__name__)


class CON_population(object):
    """
    Class to help with the construction of a realistic population

    Paremeters:
        pop: int
            The size of the populations
        rstate: Optional[RandomState]
            A numpy RandomState object

    """

    def __init__(self, pop, rstate=None):

        if rstate is None:
            logger.warning("No random state given, constructing new state")
            rstate = np.random.RandomState()
        self.__rstate = rstate

        logger.info('Constructing social circles for the population')
        logger.debug('Number of people in social circles')

        if config['social circle pdf'] == 'gauss':
            self.__social_circles = self.__social_pdf_norm(pop)
        else:
            logger.error('Unrecognized social pdf! Set to ' + config['social circle pdf'])
            raise RuntimeError('Check the social circle distribution in the config file!')

        logger.debug('The social circle interactions for each person')
        if config['social circle interactions pdf'] == 'gauss':
            self.__sc_interactions = self.__sc_interact_norm(pop)
        else:
            logger.error('Unrecognized sc interactions pdf! Set to ' +
                         config['social circle interactions pdf'])
            raise RuntimeError('Check the social circle interactions distribution in the config file!')

        logger.debug('Constructing population')

        # LIL sparse matrices are efficient for row-wise construction
        interaction_matrix = sparse.lil_matrix((pop, pop), dtype=np.bool)
        indices = np.arange(pop)

        # Here, the interaction matrix stores the connections of every
        # person, aka their social circle.
        for i, circle_size in tqdm(enumerate(self.__social_circles), total=pop):
            # Get the social circle size for this person and randomly
            # select indices of people they are connected to

            # This code below ensures that the social circle size
            # is what we simulated before
            # self.__rstate.shuffle(indices)
            # sel_indices = indices[:circle_size]

            # This code will be biased to smaller social circle sizes
            # use with care if average_social_circle_size / population
            # > 0.1%
            sel_indices = self.__rstate.randint(0, pop, size=circle_size)
            # Set the connection
            interaction_matrix[i, sel_indices] = True

        # Symmetrize the matrix, such that when A is connected to B,
        # B is connected to A. This is equivalent of a logical or
        # of ther upper and lower(tranposed) triangle of the matrix

        # Logical or for the upper triangle
        interaction_matrix = sparse.triu(interaction_matrix, 1) +\
            sparse.triu(interaction_matrix.transpose(), 1)

        # Mirror the upper triangle to the lower triangle
        interaction_matrix = interaction_matrix +\
            interaction_matrix.transpose()
        interaction_matrix = interaction_matrix.tolil()

        # No self-interaction
        interaction_matrix.setdiag(0)

        # Next, instead of boolean connections we want to store the
        # contact rate. The contact rate is given by the number
        # of connections per person divided by the interaction number

        interaction_matrix = interaction_matrix.asfptype()

        num_contacts = self.__sc_interactions
        num_connections = (interaction_matrix.sum(axis=1)-1)
        num_connections = np.asarray(num_connections).squeeze()

        with np.errstate(all='ignore'):
            contact_rate = num_contacts / (num_connections)
        contact_rate[num_connections <= 0] = 0

        # Set the contact rate for each connection by scaling the matrix
        # with each persons contact rate
        d = sparse.spdiags(contact_rate, 0, pop, pop, format="csr")

        interaction_matrix = interaction_matrix.tocsr()
        interaction_matrix = d * interaction_matrix

        # For each two person encounter (A <-> B) there are now two rates,
        # one from person A and one from B. Pick the max for both.
        # This re-symmetrizes the matrix

        upper_triu = sparse.triu(interaction_matrix, 1)
        upper_triu_transp = sparse.triu(
            interaction_matrix.transpose(), 1)

        max_inter = upper_triu.maximum(upper_triu_transp)

        interaction_matrix = max_inter + max_inter.transpose()

        self.__interaction_matrix = interaction_matrix

        print("Done constructing population")


    @property
    def population(self):
        """
        function: population
        Returns the population
        Parameters:
            -None
        Returns:
            -np.array population:
                The constructed population
        """
        # return self.__pop
        return self.__interaction_matrix

    def __social_pdf_norm(self, pop):
        """
        function: __social_pdf_norm
        Constructs the number of people in each person's circle
        Parameters:
            -int pop:
                The population size
            -np.array circles:
                The number of people in each circle
        Returns:
            -np.array circles:
                The size of the social circles for every individual
        """

        mean = config['average social circle']
        scale = config['variance social circle']

        # Minimum social circle size is 0
        a, b = (0 - mean) / scale, (pop - mean) / scale

        # could also use binomial here
        circles = truncnorm.rvs(
            a, b, loc=mean, scale=scale, size=pop, random_state=self.__rstate)
        circles = np.asarray(circles, dtype=int)

        """
        for _ in range(pop):
            res = -1
            while res < 0:
                res = int(norm.rvs(
                    size=1,
                    loc=config['average social circle'],
                    scale=config['variance social circle']))
            circles.append(res)
        circles = np.array(circles)
        """
        return circles

    def __sc_interact_norm(self, pop):
        """
        function: __sc_interact_norm
        The number of interactions within each person's circle
        Parameters:
            -int pop:
                The size of the population
        Returns:
            -np.array interact:
                The number of interactions within the social circle
        """

        mean = config['mean social circle interactions']
        scale = config['variance social circle interactions']

        a, b = (0 - mean) / scale, (self.__social_circles - mean) / scale

        zero_friends = b <= a
        b[zero_friends] = (1 - mean) / scale
        # could also use binomial here

        interactions = truncnorm.rvs(
            a, b, loc=mean, scale=scale, size=pop, random_state=self.__rstate)
        interactions = np.asarray(interactions, dtype=int)

        interactions[zero_friends] = 0
        return interactions
