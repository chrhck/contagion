# -*- coding: utf-8 -*-

"""
Name: population.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the population.
"""
import abc
from typing import Union, Tuple
from time import time
import random

import numpy as np  # type: ignore
import scipy.sparse as sparse  # type: ignore
import logging
import networkx as nx  # type: ignore

from networkx.utils import py_random_state
from networkx.generators.community import _zipf_rv_below

from .config import config
from .pdfs import construct_pdf

_log = logging.getLogger(__name__)


class Population(object, metaclass=abc.ABCMeta):
    def __init__(self, interaction_rate_scaling=1, *args, **kwargs):
        # Checking random state
        self._pop_size = config["population"]["population size"]
        self._rstate = config["runtime"]["random state"]
        random.seed(config["general"]["random state seed"])

        _log.debug("The interaction intensity pdf")
        self._interaction_rate_scaling = interaction_rate_scaling

    @abc.abstractmethod
    def get_contacts(
            self,
            rows: np.ndarray,
            cols: np.ndarray,
            return_rows=False)\
            -> Union[Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        pass

    @property
    def interaction_rate_scaling(self):
        return self._interaction_rate_scaling

    @interaction_rate_scaling.setter
    def interaction_rate_scaling(self, val):
        self._interaction_rate_scaling = val


class PopulationWithSocialCircles(Population):

    def __init__(self, interaction_rate_scaling=1, *args, **kwargs):
        super().__init__(
            interaction_rate_scaling=interaction_rate_scaling,
            *args, **kwargs)

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

        sel_indices = []
        contact_rates = []
        # NOTE: This calculation will producde duplicate contacts

        all_sel_indices = []

        succesful_rows = []

        n_contacts_sym = np.asarray(
            np.round(
                np.min(
                    np.vstack(
                        [n_contacts+n_contacts_others,
                         np.ones_like(n_contacts)*len(cols)]),
                    axis=0)),
            dtype=np.int)
        all_sel_indices = np.split(
            self._rstate.randint(
                0, self._pop_size, size=np.sum(n_contacts_sym), dtype=np.int),
            np.cumsum(n_contacts_sym))[:-1]

        for i, (row_index, sel_indices) in enumerate(
                zip(rows, all_sel_indices)):
            if len(all_sel_indices[i]) == 0:
                continue

            succesful_rows.append(np.ones(
                    len(all_sel_indices[i]), dtype=np.int)*row_index)
            contact_rates.append(
                np.ones(
                    len(all_sel_indices[i]),
                    dtype=np.float)
                * contact_rate[i])

        if all_sel_indices:
            all_sel_indices = np.concatenate(all_sel_indices)
            contact_rates = np.concatenate(contact_rates)
            succesful_rows = np.concatenate(succesful_rows)

            unique_indices, ind, counts = np.unique(
                all_sel_indices, return_index=True,
                return_counts=True)

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

    def __init__(self, interaction_rate_scaling=1, *args, **kwargs):
        """
        function: __init__
        Initializes the class Population
        Paremeters:
            -None
        Returns:
            -None
        """
        # Inputs

        super().__init__(
            interaction_rate_scaling=interaction_rate_scaling,
            *args, **kwargs)

        self.__sc_interactions = self._soc_circ_interact_pdf.rvs(
            self._pop_size)

        self.construct_population()

    def construct_population(self):
        _log.debug("Constructing population")
        start = time()
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

        self._interaction_matrix = interaction_matrix.tocsr()
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
        return self._interaction_matrix

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
        infected_sub_mtx = self._interaction_matrix[rows]
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

    def __init__(self, interaction_rate_scaling=1, *args, **kwargs):
        super().__init__(
            interaction_rate_scaling=interaction_rate_scaling,
            *args, **kwargs)

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

        # NOTE: This calculation will producde duplicate contacts

        all_sel_indices = []

        succesful_rows = []
        for i, row_index in enumerate(rows):
            n_contact = min(int(n_rnd_contacts[i]), len(cols))
            if n_contact == 0:
                continue
            this_sel_indices = self._rstate.randint(
                0, self._pop_size, size=n_contact, dtype=np.int)

            all_sel_indices.append(this_sel_indices)
            succesful_rows.append(np.ones(
                    len(this_sel_indices), dtype=np.int)*row_index)

        if all_sel_indices:
            all_sel_indices = np.concatenate(all_sel_indices)
            succesful_rows = np.concatenate(succesful_rows)

            unique_indices, ind, counts = np.unique(
                all_sel_indices, return_index=True,
                return_counts=True)

            sel_indices, pos, _ = np.intersect1d(
                unique_indices, cols, assume_unique=True, return_indices=True)

            counts = counts[pos]
            succesful_rows = succesful_rows[ind][pos]
        else:
            sel_indices = np.empty(0, dtype=int)
            contact_rates = np.empty(0, dtype=int)
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


class SocialCircle(object):
    def __init__(self, conf, pop_size, rstate):
        self._pop_size = pop_size
        self._conf = conf
        self._rstate = rstate
        self._interconnectivity = conf["interconnectivity"]
        self._fully_con = conf["fully_connected"]

    @property
    def interconnectivity(self):
        return self._interconnectivity

    @property
    def fully_con(self):
        return self._fully_con


def intra_com_cons(g, u):
    c = g.nodes[u]['community']

    adj_in_com = 0
    for adj in g[u]:
        if adj in c:
            adj_in_com += 1
    return adj_in_com


def suboptimal(g, u, target_intra):
    adj_in_com = intra_com_cons(g, u)
    return (adj_in_com < target_intra) and target_intra > 0


def supoptimal(g, u, target_intra):
    adj_in_com = intra_com_cons(g, u)
    return (adj_in_com > target_intra)


@py_random_state(6)
def _powerlaw_sequence(gamma, low, high, condition, length, max_iters, seed):
    """Returns a list of numbers obeying a constrained power law distribution.
    ``gamma`` and ``low`` are the parameters for the Zipf distribution.
    ``high`` is the maximum allowed value for values draw from the Zipf
    distribution. For more information, see :func:`_zipf_rv_below`.
    ``condition`` and ``length`` are Boolean-valued functions on
    lists. While generating the list, random values are drawn and
    appended to the list until ``length`` is satisfied by the created
    list. Once ``condition`` is satisfied, the sequence generated in
    this way is returned.
    ``max_iters`` indicates the number of times to generate a list
    satisfying ``length``. If the number of iterations exceeds this
    value, :exc:`~networkx.exception.ExceededMaxIterations` is raised.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    for i in range(max_iters):
        seq = []
        while not length(seq):
            seq.append(_zipf_rv_below(gamma, low, high, seed))
        if condition(seq):
            return seq
    raise nx.ExceededMaxIterations("Could not create power law sequence")


@py_random_state(4)
def _generate_communities(degree_seq, community_sizes, mu, max_iters, seed):
    """Returns a list of sets, each of which represents a community.

    ``degree_seq`` is the degree sequence that must be met by the
    graph.

    ``community_sizes`` is the community size distribution that must be
    met by the generated list of sets.

    ``mu`` is a float in the interval [0, 1] indicating the fraction of
    intra-community edges incident to each node.

    ``max_iters`` is the number of times to try to add a node to a
    community. This must be greater than the length of
    ``degree_seq``, otherwise this function will always fail. If
    the number of iterations exceeds this value,
    :exc:`~networkx.exception.ExceededMaxIterations` is raised.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    The communities returned by this are sets of integers in the set {0,
    ..., *n* - 1}, where *n* is the length of ``degree_seq``.

    """
    # This assumes the nodes in the graph will be natural numbers.
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    free = list(range(n))
    com_indices = range(len(community_sizes))
    for i in range(max_iters):
        v = free.pop()
        c = seed.choice(com_indices)
        # s = int(degree_seq[v] * (1 - mu) + 0.5)
        s = round(degree_seq[v] * (1 - mu))
        # If the community is large enough, add the node to the chosen
        # community. Otherwise, return it to the list of unaffiliated
        # nodes.
        if s < community_sizes[c]:
            result[c].add(v)
        else:
            free.append(v)
        # If the community is too big, remove a node from it.
        if len(result[c]) > community_sizes[c]:
            rnd_node = seed.choice(tuple(result[c]))
            free.append(rnd_node)
            result[c].remove(rnd_node)
        if not free:
            return result
    msg = 'Could not assign communities; try increasing min_community'
    raise nx.ExceededMaxIterations(msg)


class NetworkXWrappers(object):

    @staticmethod
    def add_lfr_weights(g):
        g.remove_edges_from(nx.selfloop_edges(g))
        edge_weights = {}

        inter_actions_rvs = construct_pdf(
                config["population"]["nx"]["inter freq pdf"]).rvs
        intra_actions_rvs = construct_pdf(
                config["population"]["nx"]["intra freq pdf"]).rvs

        inter_actions = inter_actions_rvs(len(g))
        intra_actions = intra_actions_rvs(len(g))
        for edge in g.edges:

            if edge[0] in g.nodes[edge[1]]["community"]:
                intra_rate_0 = intra_actions[edge[0]] / g.degree[edge[0]]
                intra_rate_1 = intra_actions[edge[1]] / g.degree[edge[1]]

                avg_int = 0.5 * (intra_rate_0 + intra_rate_1)
                # mu is the fraction of inter-community interacions
                edge_weights[edge] = avg_int
            else:
                inter_rate_0 = inter_actions[edge[0]] / g.degree[edge[0]]
                inter_rate_1 = inter_actions[edge[1]] / g.degree[edge[1]]

                avg_int = 0.5 * (inter_rate_0 + inter_rate_1)
                edge_weights[edge] = avg_int

        nx.set_edge_attributes(g, edge_weights, "weight")
        return g

    @staticmethod
    def lfr_benchmark(pop_size, **kwargs):
        kwargs["seed"] = config["runtime"]["random state"]
        g = nx.generators.community.LFR_benchmark_graph(pop_size, **kwargs)
        g = NetworkXWrappers.add_lfr_weights(g)

        return g

    @staticmethod
    def lfr_ba(pop_size, **kwargs):
        seed = config["general"]["random state seed"]
        random.seed(seed)
        state = config["runtime"]["random state"]
        kwargs["seed"] = seed
        mu = kwargs["mu"]

        g = nx.barabasi_albert_graph(pop_size, kwargs["m"], seed=seed)

        deg_seq = list(dict(nx.degree(g)).values())
        min_community = kwargs.get("min_community", None)
        max_community = kwargs.get("max_community", None)
        n = pop_size

        # Validate parameters for generating the community size sequence.
        if min_community is None:
            min_community = min(deg_seq)+1
        else:
            if min_community < min(deg_seq)+1:
                print("Min community is smaller than min(k)+1. Adjusting")
                min_community = min(deg_seq)+1
        if max_community is None:
            max_community = max(deg_seq)+1
        else:
            if max_community < max(deg_seq)+1:
                print("Max community is smaller than max(k)+1. Adjusting")
                max_community = int(1.5*(max(deg_seq)))

        low, high = min_community, max_community

        def condition(seq): return sum(seq) == n

        def length(seq): return sum(seq) >= n
        comms = _powerlaw_sequence(
            kwargs["tau"], low, high, condition,
            length, kwargs["max_iters"], seed)

        communities = _generate_communities(
            deg_seq, comms, mu, 50*n, seed)

        g.remove_edges_from(nx.selfloop_edges(g))
        for c in communities:
            for u in c:
                g.nodes[u]['community'] = c

        node_degrees = np.asarray(list(dict(g.degree).values()))

        num_inter_con = state.binomial(node_degrees, mu)
        num_intra_con = node_degrees - num_inter_con

        # print("Target mu: ", np.sum(num_inter_con) / np.sum(node_degrees))

        max_it = 75
        it = -1
        last_mu = 0
        no_change_for = 0
        while True:
            it += 1
            """
            if it % 5 == 4:
                num_inter_con = state.binomial(node_degrees, mu)
                num_intra_con = node_degrees - num_inter_con
            """

            intra_cnt = np.sum(
                [v in g.nodes[u]["community"] for u, v in g.edges])
            cur_mu = 1 - intra_cnt / g.number_of_edges()
            if (
                    np.abs(cur_mu/mu - 1) < kwargs["tolerance"] * mu or
                    cur_mu < mu
               ):
                break

            if cur_mu == last_mu:
                no_change_for += 1
                if no_change_for == 5:
                    print("No change for five steps. Current mu: ", cur_mu,
                          " Target: ", mu)
                    break

            else:
                no_change_for = 0
                last_mu = cur_mu

            if it > max_it:
                print("Max iterations reached. Current mu: ", cur_mu,
                      " Target: ", mu)
                break

            # First find all sub- and sup-optimal nodes

            all_sub_optimal_nodes = set()
            all_sup_optimal_nodes = set()

            for u, n_inter_con, n_intra_con in zip(
                    g, num_inter_con, num_intra_con):
                c = g.nodes[u]['community']

                if supoptimal(g, u, n_intra_con):
                    all_sup_optimal_nodes.add(u)
                elif suboptimal(g, u, n_intra_con):
                    all_sub_optimal_nodes.add(u)
                assert(len(all_sup_optimal_nodes & all_sub_optimal_nodes) == 0)

            for u, n_inter_con, n_intra_con in zip(
                    g, num_inter_con, num_intra_con):
                if node_degrees[u] < 2:
                    continue
                c = g.nodes[u]['community']
                if (u not in all_sub_optimal_nodes
                        and u not in all_sup_optimal_nodes):
                    continue

                sub_optimal_nodes = all_sub_optimal_nodes & c
                sup_optimal_nodes = all_sup_optimal_nodes & c

                not_optimal_nodes = sub_optimal_nodes | sup_optimal_nodes

                attempted_vs = set()
                if u in sub_optimal_nodes:
                    sub_optimal_nodes.remove(u)
                    not_optimal_nodes.remove(u)
                    all_sub_optimal_nodes.remove(u)

                    while True:
                        if len(not_optimal_nodes) < 1:
                            break
                        if not suboptimal(g, u, n_intra_con):
                            break

                        candidates = tuple(not_optimal_nodes - attempted_vs)
                        if not candidates:
                            break

                        if kwargs["pref_attach"]:
                            v = random.choices(
                                candidates,
                                weights=node_degrees[list(candidates)])[0]
                        else:
                            v = random.choice(candidates)
                        attempted_vs.add(v)

                        if v in sup_optimal_nodes:
                            # Strategy:
                            #  -Rewire an internal connection from v to u
                            #  -Rewire an external connection from u to v

                            # Get external adjacent node of u
                            target_1 = None

                            shuffled_adj = list(g[u])
                            random.shuffle(shuffled_adj)

                            for adj in shuffled_adj:
                                if (
                                        adj not in c and
                                        adj not in g[v] and
                                        adj != v
                                   ):
                                    target_1 = adj
                                    break

                            if target_1 is None:
                                continue
                            # Get internal adjacent node of v
                            target_2 = None
                            for adj in g[v]:
                                if (
                                        adj in c and
                                        adj not in g[u] and
                                        adj != u
                                   ):
                                    target_2 = adj
                                    break
                            if target_2 is None:
                                continue
                            g.remove_edge(u, target_1)
                            g.remove_edge(v, target_2)
                            g.add_edge(u, target_2)
                            g.add_edge(v, target_1)

                            if not supoptimal(g, v, num_intra_con[v]):
                                sup_optimal_nodes.remove(v)
                                all_sup_optimal_nodes.remove(v)
                                not_optimal_nodes.remove(v)
                        else:
                            # Strategy:
                            #  -Rewire an external connection from v to u
                            #  -Rewire an external connection from u to v
                            #  -Connect the two external nodes
                            # Pick a sub-optimal node from community

                            # v = random.choices(
                            #     tuple(sub_optimal_nodes),
                            #     weights=[g.degree[node]
                            #              for node in sub_optimal_nodes])[0]

                            if v in g[u]:
                                continue

                            # From edges of u
                            shuffled_adj = list(g[u])
                            random.shuffle(shuffled_adj)
                            target_1 = None
                            for adj in shuffled_adj:
                                if adj not in c:
                                    target_1 = adj
                                    break

                            if target_1 is None:
                                break

                            target_2 = None
                            for adj in g[v]:
                                if (adj not in c
                                        # and adj in all_sup_optimal_nodes
                                        and adj != target_1
                                        and target_2 not in
                                        g.nodes[target_1]["community"]
                                        and target_2 not in g[target_1]):
                                    target_2 = adj
                                    break
                            if target_2 is None:
                                break

                            g.add_edge(u, v)
                            g.remove_edge(u, target_1)
                            g.remove_edge(v, target_2)
                            g.add_edge(target_1, target_2)

                            if not suboptimal(g, v, num_intra_con[v]):
                                sub_optimal_nodes.remove(v)
                                all_sub_optimal_nodes.remove(v)
                                not_optimal_nodes.remove(v)

                    if suboptimal(g, u, num_intra_con[u]):
                        sub_optimal_nodes.add(u)
                        all_sub_optimal_nodes.add(u)
                        not_optimal_nodes.add(u)

                        # TODO: check targets?
                else:
                    sup_optimal_nodes.remove(u)
                    all_sup_optimal_nodes.remove(u)
                    not_optimal_nodes.remove(u)

                    while True:
                        if len(sub_optimal_nodes) < 1:
                            break

                        if not supoptimal(g, u, n_intra_con):
                            break

                        candidates = tuple(sub_optimal_nodes - attempted_vs)
                        if not candidates:
                            break

                        if kwargs["pref_attach"]:
                            v = random.choices(
                                candidates,
                                weights=node_degrees[list(candidates)])[0]
                        else:
                            v = random.choice(candidates)
                        attempted_vs.add(v)
                        """
                        v = random.choices(
                            tuple(sub_optimal_nodes),
                            weights=[g.degree[node]
                                     for node in sub_optimal_nodes])[0]
                        """
                        # Pick adjacent internal node
                        # u - target1
                        target_1 = None
                        shuffled_adj = list(g[u])
                        random.shuffle(shuffled_adj)
                        for adj in shuffled_adj:
                            if (
                                    adj in c
                                    and adj not in g[v]
                                    and adj != v
                               ):
                                target_1 = adj
                                break

                        if target_1 is None:
                            # No luck this turn
                            break

                        target_2 = None
                        # Choose an inter-community edge from v
                        # v - target_2
                        for adj in g[v]:
                            if (
                                    adj not in c and
                                    adj not in g[u]
                               ):
                                target_2 = adj
                                break
                        if target_2 is None:
                            break
                        g.remove_edge(u, target_1)  # u-1i, target1-1i
                        g.remove_edge(v, target_2)  # v-1e, target2-1e
                        g.add_edge(u, target_2)  # u+1e, target2+1e
                        g.add_edge(v, target_1)  # v+1i, target1+1i

                        if not suboptimal(g, v, num_intra_con[v]):
                            sub_optimal_nodes.remove(v)
                            all_sub_optimal_nodes.remove(v)
                            not_optimal_nodes.remove(v)

                    if not supoptimal(g, u, num_intra_con[u]):
                        sup_optimal_nodes.add(u)
                        all_sup_optimal_nodes.add(u)
                        not_optimal_nodes.add(u)

        g = NetworkXWrappers.add_lfr_weights(g)
        return g

    @staticmethod
    def relaxed_caveman_graph(pop_size, **kwargs):
        rstate = config["runtime"]["random state"]
        clique_size = kwargs["clique_size"]
        n_cliques = pop_size // clique_size
        p = kwargs["p"]

        inter_actions_rvs = construct_pdf(
                config["population"]["nx"]["inter freq pdf"]).rvs
        intra_actions_rvs = construct_pdf(
                config["population"]["nx"]["intra freq pdf"]).rvs

        g = nx.caveman_graph(n_cliques, clique_size)

        inter_actions = inter_actions_rvs(len(g))
        intra_actions = intra_actions_rvs(len(g))

        edge_weights = {}

        nodes = list(g)
        for (u, v) in g.edges():
            if rstate.random() < p:  # rewire the edge
                x = rstate.choice(nodes)
                if g.has_edge(u, x):
                    continue
                g.remove_edge(u, v)
                g.add_edge(u, x)
                inter_rate_0 = inter_actions[u] / g.degree[u]
                inter_rate_1 = inter_actions[x] / g.degree[x]

                avg_int = np.average([inter_rate_0, inter_rate_1])
                edge_weights[(u, x)] = avg_int
            else:
                intra_rate_0 = intra_actions[u] / g.degree[u]
                intra_rate_1 = intra_actions[v] / g.degree[v]

                avg_int = np.average([intra_rate_0, intra_rate_1])
                # mu is the fraction of inter-community interacions
                edge_weights[(u, v)] = avg_int

        nx.set_edge_attributes(g, edge_weights, "weight")
        g.remove_edges_from(nx.selfloop_edges(g))
        return g


class NetworkXPopulation(Population):
    def __init__(self, interaction_rate_scaling=1, *args, **kwargs):
        super().__init__(
            interaction_rate_scaling=interaction_rate_scaling,
            *args, **kwargs)

        self._random_interact_pdf = construct_pdf(
                config["population"]["random interactions pdf"])

        self._random_interact_intensity_pdf = construct_pdf(
                config["population"]["random interactions intensity pdf"])

        gen_func = getattr(
            NetworkXWrappers, config["population"]["nx"]["func"])
        self._graph = gen_func(
            self._pop_size, **(config["population"]["nx"]["kwargs"]))

        for node in self._graph:
            self._graph.nodes[node]["history"] = {}

    def get_contacts(
            self,
            rows: np.ndarray,
            cols: np.ndarray,
            return_rows=False)\
            -> Union[Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        contact_cols = []
        contact_rows = []
        contact_strengths = []
        n_rnd_contacts = np.asarray(np.round(
            self._random_interact_pdf.rvs(rows.shape[0])), dtype=np.int)
        rnd_indices_all = np.split(
            self._rstate.randint(
                0, len(rows), size=np.sum(n_rnd_contacts), dtype=np.int),
            np.cumsum(n_rnd_contacts))[:-1]

        rnd_ctc_intens_all = np.split(
            self._random_interact_intensity_pdf.rvs(
                np.sum(n_rnd_contacts)),
            np.cumsum(n_rnd_contacts))[:-1]
        col_set = set(cols)
        for row, n_rnd_contact, rnd_indices, rnd_ctc_intens in zip(
                rows, n_rnd_contacts, rnd_indices_all, rnd_ctc_intens_all):

            sel_cols = []
            strs = []
            sel_rows = []

            adj = self._graph[row]

            for ctc_ind, node_attrs in adj.items():
                if ctc_ind not in col_set:
                    continue
                sel_cols.append(ctc_ind)
                rate = node_attrs["weight"]

                # if ctc_ind in node["community"]:
                rate *= self.interaction_rate_scaling

                strs.append(rate)
                sel_rows.append(row)

            for rnd_ind, intens in zip(rnd_indices, rnd_ctc_intens):
                if rnd_ind not in col_set:
                    continue
                if rnd_ind not in adj:
                    sel_cols.append(rnd_ind)
                    sel_rows.append(row)
                    strs.append(intens)

            contact_cols.append(np.array(sel_cols, dtype=int))
            contact_strengths.append(np.array(strs, dtype=float))
            contact_rows.append(np.array(sel_rows, dtype=int))

        if contact_cols:

            contact_cols = np.concatenate(contact_cols)
            contact_rows = np.concatenate(contact_rows)
            contact_strengths = np.concatenate(contact_strengths)

        else:
            contact_cols = np.empty(0, dtype=int)
            contact_rows = np.empty(0, dtype=int)
            contact_strengths = np.empty(0, dtype=int)

        if return_rows:
            return contact_cols, contact_strengths, contact_rows
        else:
            return contact_cols, contact_strengths


"""
class BucketSocialCircle(SocialCircle):

    def __init__(self, conf, pop_size, rstate):
        super().__init__(conf, pop_size, rstate)

        self._soc_circ_pdf = construct_pdf(
            conf["social circle pdf"])
        self._soc_circ_int_pdf = construct_pdf(
             conf["social circle interactions pdf"])

        buckets = BucketSocialCircle.build_buckets(
            self._pop_size, self._rstate,
            self._soc_circ_pdf.rvs)
        self._buckets = buckets

    def build_matrix(self):

        adj_matrix = sparse.lil_matrix((self._pop_size, self._pop_size))

        for bucket in self.buckets:
            if len(bucket) == 0:
                continue
            int_strength = self._soc_circ_int_pdf.rvs(1)
            int_rate = int_strength / bucket.shape
            for iter_index in range(bucket.shape[0]):
                bucket_index = bucket[iter_index]
                others_mask = np.ones_like(bucket, dtype=np.bool)
                others_mask[iter_index] = False
                others = bucket[others_mask]

                if not self.fully_con:
                    others = self._rstate.choice(others, int_strength)

                adj_matrix[bucket_index, others] = int_rate
        self.adj_matrix = adj_matrix

    def build_matrix_from_bucket_sc(self, bsc):
        self.adj_matrix = bsc.adj_matrix
        print(self.buckets)

        sub_buckets = bsc.buckets
        con_rate = self.n_buckets * self.interconnectivity

        for bucket in self.buckets:
            if len(bucket) < 2:
                continue

            # Check if this bucket is connected
            if self._rstate.uniform() > self.interconnectivity:
                # no connection
                continue

            # How many connections for this bucket
            int_number = int(np.round(self._soc_circ_int_pdf.rvs(1)))
            if int_number == 0:
                continue

            int_rate = int_number / self.n_buckets

            for i in range(int_number):
                # buckets are randomized
                bucket_index = self._rstate.randint(len(bucket))
                bucket_elem = bucket[bucket_index]

                others_mask = np.ones_like(bucket, dtype=np.bool)
                others_mask[bucket_index] = False
                others = bucket[others_mask]

                # select random other bucket
                other = self._rstate.choice(others, 1)[0]

                sub_bucket = sub_buckets[bucket_elem]
                rnd_ind_this_bucket = sub_bucket[
                    self._rstate.randint(len(sub_bucket))]

                other_sub_bucket = sub_buckets[other]
                rnd_ind_others_bucket = other_sub_bucket[
                            self._rstate.randint(len(other_sub_bucket))
                            ]

                self.adj_matrix[
                    rnd_ind_this_bucket, rnd_ind_others_bucket] = int_rate

    @property
    def buckets(self):
        return self._buckets

    @property
    def n_buckets(self):
        return len(self._buckets)

    @staticmethod
    def build_buckets(pop_size, rstate, rvs):
        shuffled_indices = np.arange(pop_size, dtype=np.int)
        rstate.shuffle(shuffled_indices)
        buckets = []
        split_sizes = []
        tot_size = 0
        while tot_size < pop_size:
            sc_size = max(1, int(rvs(1)))
            split_sizes.append(sc_size)
            tot_size += sc_size

        split_indices = np.cumsum(split_sizes)[:-1]

        buckets = np.split(shuffled_indices, split_indices)

        return buckets


class IndividualBasedSocialCircle(SocialCircle):

    def __init__(self, conf, pop_size):
        super().__init__(conf, pop_size)

        soc_circ_pdf = construct_pdf(
            conf["social circle pdf"])
        soc_circ_int_pdf = construct_pdf(
             conf["social circle interactions pdf"])

        self._social_circle_sizes = (
            np.asarray(
                np.ceil(
                    soc_circ_pdf.rvs(
                        self._pop_size)
                ),
                dtype=np.int)
        )

        self._sc_interactions = (
            np.asarray(
                np.ceil(
                    soc_circ_int_pdf.rvs(
                        self._pop_size)
                ),
                np.int)
            )

        self._pop_set = set()

    @property
    def social_circle_sizes(self):
        return self._social_circle_sizes

    @property
    def sc_interactions(self):
        return self._sc_interactions


    @property
    def pop_set(self):
        return self._pop_set
"""


# class HierarchicalPopulation(AccuratePopulation):
#     """
#     class: Population
#     Class to help with the construction of a realistic population
#     Paremeters:
#         -obj log:
#             The logger
#     Returns:
#         -None
#     """

#     def construct_population(self):
#         start = time()
#         soc_circ_pdfs = []
#         soc_circ_int_pdfs = []
#         sc_con = []
#         fully_con = []
#         for layer in config["population"]["hierarchy"]:
#             soc_circ_pdfs.append(construct_pdf(
#                 layer["social circle pdf"]))

#             soc_circ_int_pdfs.append(construct_pdf(
#                 layer["social circle interactions pdf"]))

#             sc_con.append(layer["interconnectivity"])
#             fully_con.append(layer["fully_connected"])

#         self._social_circles = [
#             SocialCircle(layer, self._pop_size)
#             for layer in config["population"]["hierarchy"]]

#         _log.debug("Constructing population")
#         # LIL sparse matrices are efficient for row-wise construction
#         interaction_matrix = (
#             sparse.lil_matrix((self._pop_size, self._pop_size),
#                               dtype=np.float)
#         )

#         # Here, the interaction matrix stores the connections of every
#         # person, aka their social circle.

#         social_circle_sets = [set() for _ in self._social_circles]
#         full_set = set(range(self._pop_size))
#         sel_indices = []

#         shuffled_indices = np.arange(self._pop_size)
#         self._rstate.shuffle(shuffled_indices)

#         for i in shuffled_indices:
#             # Get unique indices
#             sel_indices = set()
#             for sc in self._social_circles:
#                 sc_sizes = sc.social_circle_sizes
#                 sc_int = sc.sc_interactions
#                 sc_con = sc.interconnectivity
#                 sc_set = sc.pop_set
#                 sc_fc = sc.fully_connected

#                 if sc_sizes[i] == 0:
#                     continue

#                 # First check if this person is can have a additional
#                 # social circle

#                 if i in sc_set:
#                     if sc_con == 0:
#                         continue
#                     # Check if this person can be in multiple circles
#                     if self._rstate.uniform() > sc_con:
#                         continue
#                     else:
#                         # If yes, also allow all connections
#                         # TODO: Potentially slow
#                         possible_contacts = list(full_set - sel_indices)
#                 else:
#                     # Only allows contacts not yet in circles
#                     # TODO: Potentially slow
#                     possible_contacts = list(full_set - sc_set - sel_indices)

#                 pos_cont_len = len(possible_contacts)
#                 if pos_cont_len == 0:
#                     continue

#                 circle_size = min(sc_sizes[i], pos_cont_len)

#                 this_sel_indices = []
#                 for _ in range(circle_size):
#                     while True:
#                         rndint = self._rstate.randint(0, pos_cont_len)
#                         ind = possible_contacts[rndint]
#                         if ind not in sel_indices:
#                             sel_indices.add(ind)
#                             this_sel_indices.append(ind)
#                             break

#                 n_int = sc_int[i]
#                 interaction_matrix[i, this_sel_indices] = n_int / circle_size
#                 sc_set.update(this_sel_indices)
#                 sc_set.add(i)

#         interaction_matrix.setdiag(0)

#         # For each two person encounter (A <-> B) there are now two rates,
#         # one from person A and one from B. Pick the max for both.
#         # This re-symmetrizes the matrix

#         upper_triu = sparse.triu(interaction_matrix, 1)
#         upper_triu_transp = sparse.triu(
#             interaction_matrix.transpose(), 1)

#         max_inter = upper_triu.maximum(upper_triu_transp)

#         interaction_matrix = max_inter + max_inter.transpose()

#         self._interaction_matrix = interaction_matrix.tocsr()
#         end = time()
#         _log.debug("Population construction took: %.1f" % (end - start))
