# -*- coding: utf-8 -*-
"""
Name: population.py
Authors: Christian Haack, Stephan Meighen-Berger, Andrea Turcati
Constructs the population.
"""
from typing import Union, Tuple
import random
import numpy as np  # type: ignore
import logging
import networkx as nx  # type: ignore
import scipy.stats

from networkx.utils import py_random_state
from networkx.generators.community import _zipf_rv_below

from ..config import config
from ..pdfs import construct_pdf
from .population_base import Population

_log = logging.getLogger(__name__)


def intra_com_cons(g, u):
    c = g.nodes[u]["community"]

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
    return adj_in_com > target_intra


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
    msg = "Could not assign communities; try increasing min_community"
    raise nx.ExceededMaxIterations(msg)


class NetworkXWrappers(object):
    @staticmethod
    def add_lfr_weights(g):
        g.remove_edges_from(nx.selfloop_edges(g))
        edge_weights = {}

        inter_actions_rvs = construct_pdf(
            config["population"]["nx"]["inter freq pdf"]
        ).rvs
        intra_actions_rvs = construct_pdf(
            config["population"]["nx"]["intra freq pdf"]
        ).rvs

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
            min_community = min(deg_seq) + 1
        else:
            if min_community < min(deg_seq) + 1:
                print("Min community is smaller than min(k)+1. Adjusting")
                min_community = min(deg_seq) + 1
        if max_community is None:
            max_community = 3*max(deg_seq)
        else:
            if max_community < max(deg_seq) + 1:
                print("Max community is smaller than max(k)+1. Adjusting")
                max_community = int(2 * (max(deg_seq)))

        low, high = min_community, max_community

        def condition(seq):
            return sum(seq) == n

        def length(seq):
            return sum(seq) >= n

        comms = _powerlaw_sequence(
            kwargs["tau"],
            low,
            high,
            condition,
            length,
            kwargs["max_iters"],
            seed,
        )

        communities = _generate_communities(deg_seq, comms, mu, 50 * n, seed)

        g.remove_edges_from(nx.selfloop_edges(g))
        for c in communities:
            for u in c:
                g.nodes[u]["community"] = c

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
                [v in g.nodes[u]["community"] for u, v in g.edges]
            )
            cur_mu = 1 - intra_cnt / g.number_of_edges()
            if (
                np.abs(cur_mu / mu - 1) < kwargs["tolerance"] * mu
                or cur_mu < mu
            ):
                break

            if cur_mu == last_mu:
                no_change_for += 1
                if no_change_for == 5:
                    print(
                        "No change for five steps. Current mu: ",
                        cur_mu,
                        " Target: ",
                        mu,
                    )
                    break

            else:
                no_change_for = 0
                last_mu = cur_mu

            if it > max_it:
                print(
                    "Max iterations reached. Current mu: ",
                    cur_mu,
                    " Target: ",
                    mu,
                )
                break

            # First find all sub- and sup-optimal nodes

            all_sub_optimal_nodes = set()
            all_sup_optimal_nodes = set()

            for u, n_inter_con, n_intra_con in zip(
                g, num_inter_con, num_intra_con
            ):
                c = g.nodes[u]["community"]

                if supoptimal(g, u, n_intra_con):
                    all_sup_optimal_nodes.add(u)
                elif suboptimal(g, u, n_intra_con):
                    all_sub_optimal_nodes.add(u)
                assert len(all_sup_optimal_nodes & all_sub_optimal_nodes) == 0

            for u, n_inter_con, n_intra_con in zip(
                g, num_inter_con, num_intra_con
            ):
                if node_degrees[u] < 2:
                    continue
                c = g.nodes[u]["community"]
                if (
                    u not in all_sub_optimal_nodes
                    and u not in all_sup_optimal_nodes
                ):
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
                                weights=node_degrees[list(candidates)],
                            )[0]
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
                                    adj not in c
                                    and adj not in g[v]
                                    and adj != v
                                ):
                                    target_1 = adj
                                    break

                            if target_1 is None:
                                continue
                            # Get internal adjacent node of v
                            target_2 = None
                            for adj in g[v]:
                                if adj in c and adj not in g[u] and adj != u:
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
                                if (
                                    adj not in c
                                    # and adj in all_sup_optimal_nodes
                                    and adj != target_1
                                    and target_2
                                    not in g.nodes[target_1]["community"]
                                    and target_2 not in g[target_1]
                                ):
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
                                weights=node_degrees[list(candidates)],
                            )[0]
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
                            if adj in c and adj not in g[v] and adj != v:
                                target_1 = adj
                                break

                        if target_1 is None:
                            # No luck this turn
                            break

                        target_2 = None
                        # Choose an inter-community edge from v
                        # v - target_2
                        for adj in g[v]:
                            if adj not in c and adj not in g[u]:
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
    def hierarchical_lfr_ba(pop_size, **kwargs):
        seed = config["general"]["random state seed"]
        n = pop_size
        random.seed(seed)

        def condition(seq):
            return sum(seq) == n

        def length(seq):
            return sum(seq) >= n

        graph_sizes = _powerlaw_sequence(
            kwargs["tau_graphs"],
            kwargs["min_graph"],
            kwargs["max_graph"],
            condition,
            length,
            kwargs["max_iters"],
            seed,
        )

        cur_size = 0

        combined = nx.Graph()
        for hier_com, gs in enumerate(graph_sizes):
            g = NetworkXWrappers.lfr_ba(gs, **kwargs)

            mapping = {i: i+cur_size for i in range(gs)}
            nx.relabel_nodes(g, mapping, copy=False)

            for node in g:
                g.nodes[node]["hier_comm"] = hier_com
                comm = g.nodes[node]["community"]

                relabeled_comm = set()
                for val in list(comm):
                    relabeled_comm.add(val+cur_size)

            combined.add_nodes_from(g.nodes(data=True))
            combined.add_edges_from(g.edges)
            cur_size += gs

        for u in combined:
            this_hcomm = combined.nodes[u]["hier_comm"]
            adjs = combined[u]
            for adj in list(adjs):
                if (adj not in combined.nodes[u]["community"]
                        and random.uniform(0, 1) < kwargs["mu_hier"]/2):

                    while True:
                        randint = random.randint(0, pop_size-1)
                        v = combined.nodes[randint]
                        if randint == u:
                            continue
                        if randint in combined.nodes[u]["community"]:
                            continue
                        if v["hier_comm"] == this_hcomm:
                            continue
                        partner = None
                        for adj2 in list(combined[randint]):
                            if (adj2 not in v["community"] and
                                 adj2 not in combined.nodes[u]["community"]):
                                partner = adj2
                                break
                        if partner is not None:
                            break

                    combined.remove_edge(u, adj)
                    combined.remove_edge(randint, partner)
                    combined.add_edge(u, randint)
                    combined.add_edge(adj, partner)
        combined = NetworkXWrappers.add_lfr_weights(combined)
        return combined

    @staticmethod
    def relaxed_caveman_graph(pop_size, **kwargs):
        clique_size = kwargs["clique_size"]
        n_cliques = pop_size // clique_size
        p = kwargs["p"]

        g = nx.relaxed_caveman_graph(n_cliques, clique_size, p)
        g.remove_edges_from(nx.selfloop_edges(g))

        if kwargs["pruning_frac"] > 0:
            rem_edges = random.sample(
                g.edges,
                k=int(kwargs["pruning_frac"] * len(g.edges))
            )
        g.remove_edges_from(rem_edges)

        return g

    @staticmethod
    def schools_model(pop_size, **kwargs):
        rstate = config["runtime"]["random state"]

        school_graph = NetworkXWrappers.relaxed_caveman_graph(
            pop_size, **kwargs
        )

        # add families
        family_sizes = scipy.stats.nbinom.rvs(
            8, 0.9, size=len(school_graph), random_state=rstate) + 1

        cur_size = len(school_graph)
        combined = nx.Graph()
        combined.add_nodes_from(school_graph.nodes(data=True))
        combined.add_edges_from(school_graph.edges)

        for node, fam_size in zip(school_graph.nodes, family_sizes):
            combined.nodes[node]["type"] = "school"
            combined.nodes[node]["testable"] = True
            f_graph = nx.generators.complete_graph(fam_size)
            mapping = {i: i+cur_size for i in range(fam_size)}
            nx.relabel_nodes(f_graph, mapping, copy=False)

            for v in f_graph.nodes:
                f_graph.nodes[v]["type"] = "family"
                f_graph.nodes[v]["testable"] = False
            combined.add_nodes_from(f_graph.nodes(data=True))
            combined.add_edges_from(f_graph.edges)
            combined.add_edge(node, list(f_graph.nodes.keys())[0])

            cur_size += fam_size
        return combined


class NetworkXPopulation(Population):
    def __init__(self, interaction_rate_scaling=1, *args, **kwargs):
        super().__init__(
            interaction_rate_scaling=interaction_rate_scaling, *args, **kwargs
        )

        self._random_interact_pdf = construct_pdf(
            config["population"]["random interactions pdf"]
        )

        self._random_interact_intensity_pdf = construct_pdf(
            config["population"]["random interactions intensity pdf"]
        )

        gen_func = getattr(
            NetworkXWrappers, config["population"]["nx"]["func"]
        )
        self._graph = gen_func(
            self._pop_size, **(config["population"]["nx"]["kwargs"])
        )

        for node in self._graph:
            self._graph.nodes[node]["history"] = {}

    def get_contacts(
        self, rows: np.ndarray, cols: np.ndarray, return_rows=False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:

        contact_cols = []
        contact_rows = []
        n_rnd_contacts = np.asarray(
            np.round(self._random_interact_pdf.rvs(rows.shape[0])),
            dtype=np.int,
        )
        rnd_indices_all = np.split(
            self._rstate.randint(
                0, len(rows), size=np.sum(n_rnd_contacts), dtype=np.int
            ),
            np.cumsum(n_rnd_contacts),
        )[:-1]

        rnd_ctc_intens_all = np.split(
            self._random_interact_intensity_pdf.rvs(np.sum(n_rnd_contacts)),
            np.cumsum(n_rnd_contacts),
        )[:-1]
        col_set = set(cols)
        for row, n_rnd_contact, rnd_indices, rnd_ctc_intens in zip(
                rows, n_rnd_contacts, rnd_indices_all, rnd_ctc_intens_all):

            sel_cols = []
            sel_rows = []

            adj = self._graph[row]

            for ctc_ind, node_attrs in adj.items():
                if ctc_ind not in col_set:
                    continue
                sel_cols.append(ctc_ind)
                sel_rows.append(row)

            for rnd_ind, intens in zip(rnd_indices, rnd_ctc_intens):
                if rnd_ind not in col_set:
                    continue
                if rnd_ind not in adj:
                    sel_cols.append(rnd_ind)
                    sel_rows.append(row)

            contact_cols.append(np.array(sel_cols, dtype=int))
            contact_rows.append(np.array(sel_rows, dtype=int))

        if contact_cols:

            contact_cols = np.concatenate(contact_cols)
            contact_rows = np.concatenate(contact_rows)

            unique_indices, ind, counts = np.unique(
                contact_cols, return_index=True, return_counts=True
            )

            contact_cols = unique_indices
            # contact_rates = contact_rates[ind] * counts
            contact_rows = contact_rows[ind]
            contact_strengths = np.ones_like(unique_indices) * counts

        else:
            contact_cols = np.empty(0, dtype=int)
            contact_rows = np.empty(0, dtype=int)
            contact_strengths = np.empty(0, dtype=int)

        if return_rows:
            return contact_cols, contact_strengths, contact_rows
        else:
            return contact_cols, contact_strengths
