 # General imports
import logging
logging.basicConfig(level="WARN")

import argparse
import pyabc    
from pyabc.sampler import DaskDistributedSampler
import numpy as np
import sys
import scipy
import os
import itertools

from contagion import Contagion, config
import contagion
from contagion.config import ConfigClass, _baseconfig
from dask.distributed import Client, LocalCluster

from summary_stats import make_sum_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Continue runid", default=argparse.SUPPRESS, const=None, nargs="?", dest="cont")
    args = parser.parse_args()
    my_config = dict(_baseconfig)
    my_config["general"]["simulation length"] = 100
    my_config["population"]["population size"] = 10000
    my_config["population"]["store population"] = True
       
    if "cont" in args:
        my_config["population"]["re-use population"] = True
        
    

    contagion = Contagion(userconfig=my_config)
    contagion.sim()

    fields = ["is_dead", "is_hospitalized"]
    data = {field: np.asarray(contagion.statistics[field]) for field in fields}

    
    
    def model(parameters):
        this_config = dict(_baseconfig)
        this_config.update(my_config)
        this_config["infection"]["latency duration pdf"]["mean"] = parameters["latency mean"]
        this_config["infection"]["incubation duration pdf"]["mean"] = parameters["incubation mean"]
        this_config["infection"]["mortality prob pdf"]["mean"] = parameters["mortality mean"]
        this_config["population"]["re-use population"] = True
        contagion = Contagion(userconfig=this_config)
        contagion.sim()

        return contagion.statistics
    
    def make_chi2_distance(fields):
        distances = []
        for field in fields:
            def distance(simulation, data):
                sane = np.asarray(data[field]) > 6
                simulation = np.asarray(simulation[field])[sane]
                data = np.asarray(data[field])[sane]
                return np.sum((simulation - data)**2 / data)
            distances.append(distance)
        return pyabc.distance.AggregatedDistance(distances)
    
    
    sum_stat_func = make_sum_stats(fields)

    distance = pyabc.AdaptivePNormDistance(
        p=2, scale_function=pyabc.distance.median_absolute_deviation_to_observation)
    
    # distance = make_chi2_distance(fields)
    
    prior = pyabc.Distribution(
        {"latency mean": pyabc.RV("uniform", 1, 20),
         "incubation mean": pyabc.RV("uniform", 1, 20),     
          "mortality mean":  pyabc.RV("uniform", 0.011, 0.2)
        })




    client = Client(scheduler_file="scheduler.json")

    #cluster = LocalCluster(n_workers=8, processes=True)
    #client = Client(cluster)

    #sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=8)
    sampler = DaskDistributedSampler(client, batch_size=1, client_max_jobs=400)
    population=pyabc.populationstrategy.AdaptivePopulationSize(
        100,
        max_population_size=2000,
        mean_cv=0.1,
        n_bootstrap=10,
        client=client)
    #population = 300
    epsilon = pyabc.epsilon.QuantileEpsilon()
    abc = pyabc.ABCSMC(model, prior, distance, population_size=population, sampler=sampler,                 
                       acceptor = pyabc.UniformAcceptor(use_complete_history=True),
                       summary_statistics=sum_stat_func,
                       eps=epsilon
                       )
    db_path = "sqlite:///" + os.path.join("/scratch4/chaack/", "abc.db")

    logging.getLogger().setLevel("DEBUG")

    if "cont" in args:
        abc.load(db_path, args.cont)
    else:
        print(sum_stat_func(data))
        abc.new(db_path, sum_stat_func(data))
        #abc.new(db_path, data)
    history1 = abc.run(max_nr_populations=15)


