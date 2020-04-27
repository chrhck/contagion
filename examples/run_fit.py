import logging
import argparse
import pyabc
from pyabc.sampler import DaskDistributedSampler
import numpy as np
import os
from contagion import Contagion
from contagion.config import _baseconfig
from dask.distributed import Client
from summary_stats import make_sum_stats

logging.basicConfig(level="WARN")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Continue runid",
                        default=argparse.SUPPRESS,
                        const=None, nargs="?", dest="cont")
    args = parser.parse_args()
    my_config = dict(_baseconfig)
    my_config["general"]["simulation length"] = 150
    my_config["population"]["population size"] = 9999
    my_config["population"]["store population"] = True
    #my_config["population"]["population class"] = "AccuratePopulation"
    my_config["population"]['population class'] = 'HomogeneousPopulation'
    # Infection
    my_config['infection']['infected'] = 1
    my_config['infection']['infection probability pdf']['mean'] = 3.
    my_config['infection']['infection probability pdf']['sd'] = 2.42
    my_config['infection']['infection probability pdf']['max_val'] = 0.15
    my_config['infection']["latency duration pdf"] = {
        "class": "Gamma",
        "sd": 3,
        "mean": 3}
    my_config['infection']["incubation duration pdf"]['mean'] = 3 
    my_config['infection']["incubation duration pdf"]['sd'] = 0.01 # 4.270138625384426
    my_config['infection']["hospitalization probability pdf"]['mean'] = 0.0001
    my_config['infection']["hospitalization probability pdf"]['sd'] = 0.00001
    my_config['infection']['will have symptoms prob pdf']['mean'] = 0.5
    my_config['infection']['will have symptoms prob pdf']['sd'] = 0.1
    # Quarantine
    my_config['measures']['type'] = 'contact_tracing'
    my_config['measures']["tracked fraction"] = 1.

    if "cont" in args:
        my_config["population"]["re-use population"] = True
        
    

    #contagion = Contagion(userconfig=my_config)
    #contagion.sim()
    data = np.loadtxt("cpp_model.csv", delimiter=",")
    
    fields = ["is_recovered", "is_infectious"]
    #data = {field: np.asarray(contagion.statistics[field]) for field in fields}
    data = {"is_recovered": data[:, 1], "is_infectious": data[:, 0]}
    
    
    def model(parameters):
        this_config = dict(_baseconfig)
        this_config.update(my_config)
        this_config['population']['social circle pdf']["mean"] = parameters["soc circ mean"]
        this_config['population']['social circle interactions pdf']["mean"] = parameters["soc circ mean"]
        this_config['infection']["latency duration pdf"]['mean'] =  parameters["latency mean"]
        #this_config['infection']["incubation duration pdf"]['mean'] =  parameters["incubation mean"]
        
        this_config["population"]["re-use population"] = False
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
        {"soc circ mean": pyabc.RV("uniform", 5, 15),
         "latency mean": pyabc.RV("uniform", 1, 10)        
        })

    client = Client(scheduler_file="scheduler.json")

    # cluster = LocalCluster(n_workers=8, processes=True)
    # client = Client(cluster)

    # sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=8)
    sampler = DaskDistributedSampler(client, batch_size=1, client_max_jobs=400)
    population = pyabc.populationstrategy.AdaptivePopulationSize(
        50,
        max_population_size=200,
        mean_cv=0.1,
        n_bootstrap=10,
        client=client)
    #population = 300
    epsilon = pyabc.epsilon.QuantileEpsilon(alpha=0.4)
    abc = pyabc.ABCSMC(model, prior, distance,
                       population_size=population, sampler=sampler,
                       acceptor=pyabc.UniformAcceptor(
                           use_complete_history=True
                        ),
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
        # abc.new(db_path, data)
    history1 = abc.run(max_nr_populations=15)
