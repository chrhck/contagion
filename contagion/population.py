# -*- coding: utf-8 -*-

"""
Name: population.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the population.
"""


import numpy as np
import scipy.stats
# A truncated normal continuous random variable
from scipy.stats import truncnorm
import scipy.sparse as sparse


from .pdfs import TruncatedNormal
from .config import config


class Population(object):
    """
    Class to help with the construction of a realistic population

    Paremeters:
        -obj log:
            The logger

    """

    def __init__(self, log):

        # Inputs
        self.__log = log.getChild(self.__class__.__name__)
        self.__pop = config['population size']
        # Checking random state
        if config['random state'] is None:
            self.__log.warning("No random state given, constructing new state")
            self.__rstate = np.random.RandomState()
        else:
            self.__rstate = config['random state']

        self.__log.info('Constructing social circles for the population')
        self.__log.debug('Number of people in social circles')
        if config['social circle pdf'] == 'gauss':

            soc_circ_pdf = TruncatedNormal(
                0,
                config['population size'],
                config['average social circle'],
                config['infection duration variance']
                )

            self.__social_circles = soc_circ_pdf.rvs(self.__pop, dtype=np.int)
        else:
            self.__log.error('Unrecognized social pdf! Set to ' + config['social circle pdf'])
            raise RuntimeError('Check the social circle distribution in the config file!')

        self.__log.debug('The social circle interactions for each person')
        if config['social circle interactions pdf'] == 'gauss':

            upper = self.__social_circles
            # Check if there are people with zero contacts and set them to
            # 1 for the time being
            zero_contacts_mask = upper == 0
            upper[zero_contacts_mask] = 1

            soc_circ_interact_pdf = TruncatedNormal(
                0,
                upper,
                config['mean social circle interactions'],
                config['variance social circle interactions']
                )

            self.__sc_interactions = soc_circ_interact_pdf.rvs(self.__pop)

            # Set the interactions to zero for all people with zero contacts
            self.__sc_interactions[zero_contacts_mask] = 0
        else:
            self.__log.error('Unrecognized sc interactions pdf! Set to ' +
                             config['social circle interactions pdf'])
            raise RuntimeError('Check the social circle interactions distribution in the config file!')

        self.__log.debug('Constructing population')
        # LIL sparse matrices are efficient for row-wise construction
        interaction_matrix = sparse.lil_matrix((self.__pop, self.__pop), dtype=np.bool)
        indices = np.arange(self.__pop)

        # Here, the interaction matrix stores the connections of every
        # person, aka their social circle.
        for i, circle_size in enumerate(self.__social_circles):
            # Get the social circle size for this person and randomly
            # select indices of people they are connected to
            self.__rstate.shuffle(indices)
            sel_indices = indices[:circle_size]
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

        # Context manager
        with np.errstate(all='ignore'):
            contact_rate = num_contacts / (num_connections)
        contact_rate[num_connections <= 0] = 0

        # Set the contact rate for each connection by scaling the matrix
        # with each persons contact rate
        d = sparse.spdiags(contact_rate, 0, self.__pop, self.__pop, format="csr")

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

    @property
    def population(self):
        """
        function: population
        Returns the population
        Parameters:
            -None
        Returns:
            -scipy.sparse population:
                The constructed population
        """
        # return self.__pop
        return self.__interaction_matrix
