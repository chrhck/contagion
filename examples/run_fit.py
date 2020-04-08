 # General imports
import logging
logging.basicConfig(level="WARN")


import pyabc    
from pyabc.sampler import DaskDistributedSampler
import numpy as np
import sys
import scipy
import os

from contagion import Contagion, config
import contagion
from contagion.config import ConfigClass, _baseconfig
from dask.distributed import Client


my_config = {
    "simulation length": 100,
    "population size": 10000
}

contagion = Contagion(userconfig=my_config)
contagion.sim()

fields = ["is_dead", "is_hospitalized"]
data = {field: np.asarray(contagion.statistics[field]) for field in fields}

def model(parameters):
    this_config = dict(my_config)
    for key, val in parameters.items():
        key = key.replace("_", " ")
        this_config[key] = val
    this_config["re-use population"] = True
    contagion = Contagion(userconfig=this_config)
    contagion.sim()
    
    return contagion.statistics


def make_sum_stats(fields):

    def gen_summary_stats(simulation):
        sum_stats = {}
        
        for field in fields:
            sum_stats[field+"_xmax_diff"] = np.argmax(np.diff(simulation[field]))
            sum_stats[field+"_ymax_diff"] = np.max(np.diff(simulation[field]))
            sum_stats[field+"_xmin_diff"] = np.argmin(np.diff(simulation[field]))
            sum_stats[field+"_ymin_diff"] = np.min(np.diff(simulation[field]))
            sum_stats[field+"_xmax"] = np.argmax(simulation[field])
            sum_stats[field+"_ymax"] = np.max(simulation[field])
            sum_stats[field+"_avg_growth_rate"] = np.average(np.diff(simulation[field]))
            sum_stats[field+"_val_end"] = simulation[field][-1]
                                   
        return sum_stats
    return gen_summary_stats

sum_stat_func = make_sum_stats(fields)
    
distance = pyabc.AdaptivePNormDistance(
    p=2, scale_function=pyabc.distance.root_mean_square_deviation)

#distance = pyabc.AggregatedDistance([distance0, distance1])
prior = pyabc.Distribution(
    {"infectious duration mean": pyabc.RV("uniform", 1, 10),
     "incubation duration mean": pyabc.RV("uniform", 1, 10),     
      "mortality rate mean":  pyabc.RV("uniform", 0.05, 0.5)
    })



client = Client("tcp://10.152.133.30:46571")

#sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=8)
sampler = DaskDistributedSampler(client, batch_size=1, client_max_jobs=400)
population=pyabc.populationstrategy.AdaptivePopulationSize(200, max_population_size=10000,
                                                          mean_cv=0.1, n_bootstrap=5)
epsilon = pyabc.epsilon.QuantileEpsilon()
abc = pyabc.ABCSMC(model, prior, distance, population_size=population, sampler=sampler,                 
                   acceptor = pyabc.UniformAcceptor(use_complete_history=True),
                   summary_statistics=sum_stat_func,
                   eps=epsilon
                   )
db_path = "sqlite:///" + os.path.join("/scratch4/chaack/", "abc.db")

logging.getLogger().setLevel("INFO")

abc.new(db_path, sum_stat_func(data))
history1 = abc.run(max_nr_populations=10)


