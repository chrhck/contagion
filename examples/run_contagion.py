#! /bin/env python
from copy import deepcopy
import os
import numpy as np
import pandas as pd
import pickle
import yaml
from time import time
from argparse import ArgumentParser
from contagion import Contagion

parser = ArgumentParser()
parser.add_argument("-c", "--config", required=True, dest="config")
parser.add_argument("-n", default=1, dest="n_rep", type=int)
parser.add_argument("--seed_offset", default=0, dest="seed_offset", type=int)
parser.add_argument("-o", "--outfile", required=True, dest="outfile")

args = parser.parse_args()

config = yaml.unsafe_load(open(args.config))

seed = config["general"]["random state seed"]

results = []

for i in range(args.n_rep):
    start = time()
    print("Running rep: {}".format(i))
    config["general"]["random state seed"] = seed+i + args.seed_offset
    contagion = Contagion(deepcopy(config))
    contagion.sim()
    inf_hist = np.atleast_2d(np.squeeze(np.hstack(contagion.trace_infection)))
    stats = pd.DataFrame(contagion.statistics)
    
    if config["population"]["nx"]["func"] == "schools_model":
        g = contagion.pop._graph
        traced_inf = 0
        inf_school = 0
        inf_family = set()
        traced_inf_school = 0
        traced_inf_fam = set()

        for node in g:    
            hist = g.nodes[node]["history"]
            if "infected_at" in hist:
                if g.nodes[node]["type"] == "school":
                    inf_school += 1
                else:
                    inf_family.add(g.nodes[node]["family_index"])
                if "traced" in hist:
                    if g.nodes[node]["type"] == "school":
                        traced_inf_school += 1
                    traced_inf_fam.add(g.nodes[node]["family_index"])
        stats["inf_school"] = inf_school
        stats["traced_inf_school"] = traced_inf_school
        stats["inf_family"] = len(inf_family)
        stats["traced_inf_fam"] = len(inf_family & traced_inf_fam)
    
        results.append((stats, inf_hist, g))
    else:
        results.append((stats, inf_hist))
    print("Last round took: {}".format(time()-start))
        
    
    
    

res_dict = {
    "args": args,
    "results": results}
pickle.dump(res_dict, open(args.outfile, "wb"))

    
