# -*- coding: utf-8 -*-

"""
Name: population.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the population.
"""
import abc
from typing import Union, Tuple
from time import time

import numpy as np  # type: ignore
import scipy.sparse as sparse  # type: ignore
import logging

from .config import config
from .pdfs import construct_pdf

_log = logging.getLogger(__name__)


class Population(object, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        # Checking random state

        self._pop_size = config["population"]["population size"]
        self._rstate = config["runtime"]["random state"]

        _log.debug("The interaction intensity pdf")

    @abc.abstractmethod
    def get_contacts(
            self,
            rows: np.ndarray,
            cols: np.ndarray,
            return_rows=False)\
            -> Union[Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        pass


class PopulationWithSocialCircles(Population):

    def __init__(self, interaction_rate_scaling=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interaction_rate_scaling = interaction_rate_scaling

        _log.info("Constructing social circles for the population")

        soc_circ_pdf = construct_pdf(
            config["population"]["social circle pdf"])

        self._social_circles = soc_circ_pdf.rvs(
            self._pop_size, dtype=np.int)

        soc_circ_interact_pdf = construct_pdf(
                config["population"]["social circle interactions pdf"])

        self._soc_circ_interact_pdf = soc_circ_interact_pdf

    @property
    def social_circles(self):
        return self._social_circles

    @property
    def interaction_rate_scaling(self):
        return self._interaction_rate_scaling

    @interaction_rate_scaling.setter
    def interaction_rate_scaling(self, val):
        self._interaction_rate_scaling = val


class HomogeneousPopulation(PopulationWithSocialCircles):

    def get_contacts(
            self,
            rows: np.ndarray,
            cols: np.ndarray,
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

        n_contacts = self._soc_circ_interact_pdf.rvs(rows.shape[0])
        with np.errstate(all="ignore"):
            contact_rate = n_contacts / self._social_circles[rows]
        contact_rate[self._social_circles[rows] == 0] = 0

        # n_contacts is the number of contacts for each infected
        # we also want the number of times the infected person
        # is contacted by others. For the ad-hoc calculation,
        # we use the contact rate of each infected person
        # Number of people who can contact each infected:
        # n_candidates = pop_size - n_infected
        # contact_rate = 1 / pop_size * n_candidates * contact_rate

        n_candidates = self._pop_size - len(rows)
        contact_rate_others = 1 / self._pop_size * n_candidates * contact_rate
        n_contacts_others = self._rstate.poisson(
            contact_rate_others, size=len(rows))

        succesful_rows = []
        sel_indices = []
        contact_rates = []
        # NOTE: This calculation will producde duplicate contacts

        all_sel_indices = []
        idx_ptrs = [0]
        succesful_rows = []
        for i, row_index in enumerate(rows):
            n_contact = min(int(n_contacts[i]+n_contacts_others[i]), len(cols))
            if n_contact == 0:
                continue
            this_sel_indices = self._rstate.randint(
                0, self._pop_size, size=n_contact, dtype=np.int)

            all_sel_indices.append(this_sel_indices)
            idx_ptrs.append(idx_ptrs[-1]+n_contact)
            succesful_rows.append(np.ones(
                    len(this_sel_indices), dtype=np.int)*row_index)
            contact_rates.append(
                np.ones(
                    len(this_sel_indices),
                    dtype=np.float)
                *contact_rate[i])

        if all_sel_indices:
            all_sel_indices = np.concatenate(all_sel_indices)
            contact_rates = np.concatenate(contact_rates)
            succesful_rows = np.concatenate(succesful_rows)

            unique_indices, ind, reverse_ind, counts = np.unique(
                all_sel_indices, return_index=True,
                return_inverse=True, return_counts=True)

            sel_indices, pos, _ = np.intersect1d(
                unique_indices, cols, assume_unique=True, return_indices=True)

            counts = counts[pos]
            contact_rates = contact_rates[ind][pos] * counts
            succesful_rows = succesful_rows[ind][pos]
        else:
            sel_indices = np.empty(0, dtype=int)
            contact_rates = np.empty(0, dtype=int)
            succesful_rows = np.empty(0, dtype=int)

        """
        unique_indices, counts = np.unique(
            this_sel_indices, return_counts=True)
        this_sel_indices, pos, _ = np.intersect1d(
            unique_indices, cols, assume_unique=True, return_indices=True)

        counts = counts[pos]

        sel_indices.append(this_sel_indices)
        contact_rates.append(
            np.ones(
                len(this_sel_indices),
                dtype=np.float)
            *contact_rate[i]*counts)
        succesful_rows.append(
                np.ones(len(this_sel_indices), dtype=int) * row_index)

        for i, row_index in enumerate(rows):
            n_contact = min(int(n_contacts[i]+n_contacts_others[i]), len(cols))

            this_sel_indices = set()
            for _ in range(n_contact):
                while True:
                    ind = self._rstate.randint(0, len(cols), dtype=np.int)
                    if ind not in this_sel_indices:
                        this_sel_indices.add(ind)
                        break

            sel_indices.append(list(this_sel_indices))
            contact_rates.append(
                np.ones(n_contact, dtype=np.float)*contact_rate[i])
            if np.any(contact_rate[i] > 0):
                succesful_rows.append(
                    np.ones(int(n_contact), dtype=int) * row_index)
            else:
                succesful_rows.append(np.empty(0, dtype=int))


        if sel_indices:
            sel_indices = np.concatenate(sel_indices)
            contact_rates = np.concatenate(contact_rates)
            succesful_rows = np.concatenate(succesful_rows)
        else:
            sel_indices = np.empty(0, dtype=int)
            contact_rates = np.empty(0, dtype=int)
            succesful_rows = np.empty(0, dtype=int)
        """
        contact_rates = contact_rates * self.interaction_rate_scaling
        if return_rows:
            return sel_indices, contact_rates, succesful_rows
        return sel_indices, contact_rates


class PureSocialCirclePopulation(PopulationWithSocialCircles):
    """
    class: Population
    Class to help with the construction of a realistic population
    Paremeters:
        -obj log:
            The logger
    Returns:
        -None
    """

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)

        self.__sc_interactions = self._soc_circ_interact_pdf.rvs(
            self._pop_size)

        _log.debug("Constructing population")
        # LIL sparse matrices are efficient for row-wise construction
        interaction_matrix = (
            sparse.lil_matrix((self._pop_size, self._pop_size), dtype=np.bool)
        )

        # Here, the interaction matrix stores the connections of every
        # person, aka their social circle.

        for i, circle_size in enumerate(self._social_circles):
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
        with np.errstate(all="ignore"):
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
        _log.debug("Population construction took: %.1f" % (end - start))

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
            cols: np.ndarray,
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
        infected_sub_mtx = infected_sub_mtx[:, cols]

        if return_rows:
            # here we need the rows
            # NOTE: This is ~2times slower

            contact_rows, contact_cols, contact_strengths =\
                sparse.find(infected_sub_mtx)
        else:
            contact_cols = infected_sub_mtx.indices  # nonzero column indices
            contact_strengths = infected_sub_mtx.data  # nonzero data

        contact_strengths = contact_strengths * self.interaction_rate_scaling
        contact_cols = cols[contact_cols]

        if return_rows:
            contact_rows = rows[contact_rows]
            return contact_cols, contact_strengths, contact_rows
        else:
            return contact_cols, contact_strengths


class AccuratePopulation(PureSocialCirclePopulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._random_interact_pdf = construct_pdf(
                config["population"]["random interactions pdf"])

        self._random_interact_intensity_pdf = construct_pdf(
                config["population"]["random interactions intensity pdf"])

    def get_contacts(
            self,
            rows: np.ndarray,
            cols: np.ndarray,
            return_rows=False)\
            -> Union[Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        res_soc_cir = super().get_contacts(
            rows, cols, return_rows)

        n_rnd_contacts = self._random_interact_pdf.rvs(rows.shape[0])
        # TODO: Random contacts are not yet symmetric
        sel_indices = []
        succesful_rows = []
        for i, row_index in enumerate(rows):

            n_contact = min(int(n_rnd_contacts[i]), len(cols))
            if n_contact == 0:
                continue
            """
            for _ in range(n_contact):
                while True:
                    ind = self._rstate.randint(0, len(cols))
                    if ind not in this_sel_indices:
                        this_sel_indices.add(ind)
                        break
            """
            # NOTE: This *will* produce duplicates
            this_sel_indices = self._rstate.randint(
                0, len(cols), size=n_contact, dtype=np.int)
            sel_indices.append(cols[this_sel_indices])

            succesful_rows.append(
                np.ones(int(n_contact), dtype=int) * row_index)

        if sel_indices:
            sel_indices = np.concatenate(sel_indices)
            succesful_rows = np.concatenate(succesful_rows)
        else:
            sel_indices = np.empty(0, dtype=int)
            succesful_rows = np.empty(0, dtype=int)

        contact_rates = self._random_interact_intensity_pdf.rvs(
            len(sel_indices))
        sel_indices = np.concatenate([sel_indices, res_soc_cir[0]])

        # Only the social circle rates have been scaled!
        contact_rates = np.concatenate([contact_rates, res_soc_cir[1]])
        if return_rows:
            succesful_rows = np.concatenate([succesful_rows, res_soc_cir[2]])
            return sel_indices, contact_rates, succesful_rows
        return sel_indices, contact_rates
