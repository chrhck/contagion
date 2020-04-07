# -*- coding: utf-8 -*-

"""
Name: population.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the population.
"""
import abc
from typing import Union, Tuple
from time import time

import numpy as np
import scipy.sparse as sparse
import logging


from .pdfs import TruncatedNormal, Gamma
from .config import config

_log = logging.getLogger(__name__)

STD_PDFS = {
    'gauss': TruncatedNormal,
    'gamma': Gamma
}


class Population(object, metaclass=abc.ABCMeta):
    def __init__(self):
        # Checking random state
        if config['random state'] is None:
            _log.warning("No random state given, constructing new state")
            self._rstate = np.random.RandomState()
        else:
            self._rstate = config['random state']

        self._pop_size = config['population size']

    @abc.abstractmethod
    def get_contacts(
            self,
            rows: np.ndarray,
            return_rows=False)\
            -> Union[Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        pass


class PopulationWithSocialCircles(Population):

    def __init__(self):
        super().__init__()

        _log.info('Constructing social circles for the population')
        _log.debug('Number of people in social circles')
        try:
            soc_circ_pdf = (
                STD_PDFS[config['social circle pdf']](
                    config['average social circle'],
                    config['variance social circle'],
                    max_val=config['population size']
                )
            )
            self._social_circles = soc_circ_pdf.rvs(self._pop_size, dtype=np.int)
        except ValueError:
            _log.error('Unrecognized social circle pdf! Set to ' +
                             config['social circle pdf'])
            exit('Check the social circle pdf in the config file!')

    @property
    def social_circles(self):
        return self._social_circles


class HomogeneousPopulation(PopulationWithSocialCircles):

    def __init__(self):
        super().__init__()

        _log.debug('The social circle interactions for each person')
        try:
            soc_circ_interact_pdf = (
                STD_PDFS[config['social circle pdf']](
                    config['mean social circle interactions'],
                    config['variance social circle interactions']
                )
            )
            self.__soc_circ_interact_pdf = soc_circ_interact_pdf
        except ValueError:
            _log.error('Unrecognized social circle pdf! Set to ' +
                             config['social circle pdf'])
            exit('Check the social circle pdf in the config file!')

    def get_contacts(
            self,
            rows: np.ndarray,
            return_rows=False)\
            -> Union[Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the contacts for indices in `rows`

        Parameters:
            rows np.ndarray
            return_rows: Optional[bool]
                Return the rows with non-zero interactions

        Returns:
            contact_indices: np.ndarray
            contact_strengths: np.ndarray
        """

        if return_rows:
            raise RuntimeError("Not yet supported")

        sel_indices = []
        contact_rates = []
        n_contacts = self.__soc_circ_interact_pdf.rvs(rows.shape[0])
        contact_rate = n_contacts / self._social_circles[rows]

        contact_rate[self._social_circles[rows] == 0] = 0

        for i, row_ind in enumerate(rows):
            sel_indices.append(
                self._rstate.randint(
                    0, self._pop_size, size=int(n_contacts[i])))
            contact_rates.append(np.ones(int(n_contacts[i]))*contact_rate[i])

        if sel_indices:
            return (
                np.concatenate(sel_indices),
                np.concatenate(contact_rates))
        return np.empty(0, dtype=np.int), np.empty(0, dtype=np.int)


class AccuratePopulation(PopulationWithSocialCircles):
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

        start = time()
        super().__init__()

        _log.debug('The social circle interactions for each person')
        try:
            upper = self._social_circles
            # Check if there are people with zero contacts and set them to
            # 1 for the time being
            zero_contacts_mask = upper == 0
            # upper[zero_contacts_mask] = 1
            soc_circ_interact_pdf = (
                STD_PDFS[config['social circle pdf']](
                    config['mean social circle interactions'],
                    config['variance social circle interactions']
                )
            )
            self.__sc_interactions = soc_circ_interact_pdf.rvs(self._pop_size)
        except ValueError:
            _log.error('Unrecognized social circle pdf! Set to ' +
                             config['social circle pdf'])
            exit('Check the social circle pdf in the config file!')
        self.__sc_interactions = soc_circ_interact_pdf.rvs(self._pop_size)
        # Set the interactions to zero for all people with zero contacts
        self.__sc_interactions[zero_contacts_mask] = 0
        _log.debug('Constructing population')
        # LIL sparse matrices are efficient for row-wise construction
        interaction_matrix = (
            sparse.lil_matrix((self._pop_size, self._pop_size), dtype=np.bool)
        )

        # Here, the interaction matrix stores the connections of every
        # person, aka their social circle.

        for i, circle_size in enumerate(self.social_circles):
            # Get unique indices
            sel_indices = set()
            for _ in range(circle_size):
                while True:
                    ind = self._rstate.randint(0, self._pop_size)
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
                           self._pop_size, self._pop_size, format="csr")

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

        self.__interaction_matrix = interaction_matrix.tocsr()
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
        # return self._pop_size
        return self.__interaction_matrix

    def get_contacts(
            self,
            rows: np.ndarray,
            return_rows=False)\
            -> Union[Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the contacts for indices in `rows`

        Parameters:
            rows np.ndarray
            return_rows: Optional[bool]
                Return the rows with non-zero interactions

        Returns:
            contact_indices: np.ndarray
            contact_strengths: np.ndarray
        """

        infected_sub_mtx = self.__interaction_matrix[rows]
        if return_rows:
            # here we need the rows
            # NOTE: This is ~2times slower

            contact_rows, contact_cols, contact_strengths =\
                sparse.find(infected_sub_mtx)
        else:
            contact_cols = infected_sub_mtx.indices  # nonzero column indices
            contact_strengths = infected_sub_mtx.data  # nonzero data

        if return_rows:
            return contact_cols, contact_strengths, contact_rows
        else:
            return contact_cols, contact_strengths
