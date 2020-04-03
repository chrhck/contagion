# -*- coding: utf-8 -*-

"""
Name: population.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the population.
"""


import numpy as np
# A truncated normal continuous random variable
import scipy.sparse as sparse
import logging
from time import time

from .pdfs import TruncatedNormal, Gamma
from .config import config

_log = logging.getLogger(__name__)


class Population(object):
    """
    class: Population
    Class to help with the construction of a realistic population
    Paremeters:
        -obj log:
            The logger
    Returns:
        -None
    """

    def __init__(self):
        """
        function: __init__
        Initializes the class Population
        Paremeters:
            -None
        Returns:
            -None
        """
        # Inputs
        self.__pop = config['population size']
        self.__std_pdfs = {
            'gauss': TruncatedNormal,
            'gamma': Gamma
        }
        # Checking random state
        if config['random state'] is None:
            _log.warning("No random state given, constructing new state")
            self.__rstate = np.random.RandomState()
        else:
            self.__rstate = config['random state']
        start = time()
        _log.info('Constructing social circles for the population')
        _log.debug('Number of people in social circles')
        try:
            soc_circ_pdf = (
                self.__std_pdfs[config['social circle pdf']](
                    config['average social circle'],
                    config['variance social circle'],
                    max_val=config['population size']
                )
            )
            self.__social_circles = soc_circ_pdf.rvs(self.__pop, dtype=np.int)
        except ValueError:
            _log.error('Unrecognized social circle pdf! Set to ' +
                             config['social circle pdf'])
            exit('Check the social circle pdf in the config file!')

        _log.debug('The social circle interactions for each person')
        try:
            upper = self.__social_circles
            # Check if there are people with zero contacts and set them to
            # 1 for the time being
            zero_contacts_mask = upper == 0
            # upper[zero_contacts_mask] = 1
            soc_circ_interact_pdf = (
                self.__std_pdfs[config['social circle pdf']](
                    config['mean social circle interactions'],
                    config['variance social circle interactions']
                )
            )
            self.__sc_interactions = soc_circ_interact_pdf.rvs(self.__pop)
        except ValueError:
            _log.error('Unrecognized social circle pdf! Set to ' +
                             config['social circle pdf'])
            exit('Check the social circle pdf in the config file!')
        self.__sc_interactions = soc_circ_interact_pdf.rvs(self.__pop)
        # Set the interactions to zero for all people with zero contacts
        self.__sc_interactions[zero_contacts_mask] = 0
        _log.debug('Constructing population')
        # LIL sparse matrices are efficient for row-wise construction
        interaction_matrix = (
            sparse.lil_matrix((self.__pop, self.__pop), dtype=np.bool)
        )

        # Here, the interaction matrix stores the connections of every
        # person, aka their social circle.

        for i, circle_size in enumerate(self.__social_circles):
            # Get unique indices
            sel_indices = set()
            for _ in range(circle_size):
                while True:
                    ind = self.__rstate.randint(0, self.__pop)
                    if ind not in sel_indices:
                        sel_indices.add(ind)
                        break
            sel_indices = list(sel_indices)

            interaction_matrix[i, sel_indices] = True

        # Symmetrize the matrix, such that when A is connected to B,
        # B is connected to A. This is equivalent of a logical or
        # of ther upper and lower(tranposed) triangle of the matrix

        # Logical or for the upper triangle
        # interaction_matrix = interaction_matrix.tocsr()
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
        d = sparse.spdiags(contact_rate, 0,
                           self.__pop, self.__pop, format="csr")

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
        end = time()
        _log.debug('Population construction took: %.1f' % (end - start))

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
