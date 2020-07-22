from collections import defaultdict
from typing import Dict

from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import pandas as pd


def plot_infection_history(
        g: nx.Graph,
        pos: Dict[int, np.ndarray],
        stats: pd.DataFrame,
        outfile: str,
        cumulative=False):
    """
    Create an animation of the infection spread through the graph

    Paramaters:
        g: nx.Graph
        pos: Dict[int, np.ndarray]
            Node positions. Call e.g. nx.spring_layout
        stats: pd.DataFrame
            Contagion run statistics
        cumulative:
            Plot all infections rather than only currently infected
    """

    fig = plt.figure(figsize=(16, 9))
    spec = gs.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[3, 2])
    spec2 = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec[1])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec2[0])
    ax3 = fig.add_subplot(spec2[1])

    max_val = 0
    for node in g:
        if "is_infected" in g.nodes[node]["history"]:
            state_history = g.nodes[node]["history"]["is_infected"]
            max_val = max(max_val, state_history[0][0])
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max(stats["is_recovered"])*1.1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Infected")
    ax3.set_ylabel("Infected")

    pcol = nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=list(g.nodes),
        node_color="lightgrey",
        node_size=10,
        ax=ax1)

    line, = ax2.plot([], [], ls="-", lw=2, color="r")
    line2, = ax2.plot([], [], ls="-", lw=2, color="lightgreen")

    def update(t):
        node_cols = []
        inf_by = defaultdict(int)
        for node in g:
            col = "lightgrey"

            if "is_infected" in g.nodes[node]["history"]:
                state_history = g.nodes[node]["history"]["is_infected"]

                if cumulative:
                    if state_history[0][0] <= t:
                        col = "r"
                else:
                    if ((state_history[0][0] <= t) and
                        ((len(state_history) == 1) or
                         (state_history[1][0] > t))):
                        if (("traced" in g.nodes[node]["history"] )
                                and (min(g.nodes[node]["history"]["traced"]) <= t)):
                            col = "darkorange"
                        else:                            
                            col = "r"
                    if ((len(state_history) == 2) and
                            (state_history[1][0] <= t)):
                        if ((("traced" in g.nodes[node]["history"])
                                and (min(g.nodes[node]["history"]["traced"]) <=
                                     state_history[1][0]))
                             or "symptomatic" in g.nodes[node]["history"]):
                            col = "darkgreen"
                        else:
                            col = "purple"
            if "infected_by" in g.nodes[node]["history"]:
                if state_history[0][0] <= t:
                    inf_by[g.nodes[node]["history"]["infected_by"]] += 1

            node_cols.append(col)
        pcol.set_facecolor(node_cols)
        pcol.set_alpha(0.8)
        line.set_data(np.arange(t), stats["is_infected"][:t])
        line2.set_data(np.arange(t), stats["is_recovered"][:t])

        if inf_by:
            inf_by = {k: v for k, v in
                      sorted(
                        inf_by.items(),
                        key=lambda item: item[1],
                        reverse=True)}
            min_len = min(10, len(inf_by))
            ax3.clear()
            ax3.bar(np.arange(min_len), list(inf_by.values())[:min_len])
            ax3.set_xticks(np.arange(min_len))
            ax3.set_xticklabels(list(inf_by.keys())[:min_len])
            ax3.set_ylabel("Num infected")
        return pcol, line, line2

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, max_val),
        interval=200,
        blit=False)
    anim.save(outfile)
