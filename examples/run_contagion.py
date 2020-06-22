#! /bin/env python
import os
import numpy as np
import pandas as pd
import pickle
import yaml

from argparse import ArgumentParser
from contagion import Contagion

parser = ArgumentParser()
parser.add_argument("-c", "--config", required=True, dest="config")
parser.add_argument("-n", default=1, dest="n_rep", type=int)
parser.add_argument("-o", "--outfile", required=True, dest="outfile")

args = parser.parse_args()

config = yaml.unsafe_load(open(args.config))

seed = config["general"]["random state seed"]

results = []
for i in range(args.n_rep):
    config["general"]["random state seed"] = seed+i
    contagion = Contagion(config)
    contagion.sim()
    inf_hist = np.atleast_2d(np.squeeze(np.hstack(contagion.trace_infection)))
    stats = pd.DataFrame(contagion.statistics)
    results.append((stats, inf_hist))

res_dict = {
    "args": args,
    "results": results}
pickle.dump(res_dict, open(args.outfile, "wb"))

    
