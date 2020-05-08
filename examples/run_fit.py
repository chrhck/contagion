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
import yaml
import pandas as pd

logging.basicConfig(level="WARN")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Continue runid",
                        default=argparse.SUPPRESS,
                        const=None, nargs="?", dest="cont", type=int)
    parser.add_argument("-d", choices=["summary", "chi2", "ad"], dest="distance", default="summary")
    parser.add_argument("-v", choices=["mean", "minimal"], dest="variance_mode", default="mean")
    args = parser.parse_args()
    my_config = yaml.safe_load(open("benchmark_config.yaml"))
   

    if "cont" in args:
        my_config["population"]["re-use population"] = True

    #contagion = Contagion(userconfig=my_config)
    #contagion.sim()
    data = np.loadtxt("cpp_model.csv", delimiter=",")
    
    fields = ["is_recovered", "is_infectious", "is_latent"]
    #data = {field: np.asarray(contagion.statistics[field]) for field in fields}
    data = {"is_recovered": data[:, 1], "is_infectious": data[:, 0], "is_latent": data[:, 2]}
    
    
    def model(parameters):
        this_config = dict(_baseconfig)
        this_config.update(my_config)
        this_config['population']['social circle pdf']["mean"] = parameters["soc circ mean"]        
        this_config['population']['social circle interactions pdf']["mean"] = parameters["soc circ mean"]        
        this_config['infection']["latency duration pdf"]['mean'] =  parameters["latency mean"]        
        this_config['infection']["infectious duration pdf"]['mean'] =  parameters["infectious dur mean"]        
        this_config['infection']["recovery time pdf"]['mean'] =  parameters["recovery dur mean"]        
        this_config['infection']["incubation duration pdf"]['mean'] =  parameters["incub dur mean"]        
        this_config['infection']["infection probability pdf"]['max_val'] =  parameters["inf prob max"]
        this_config['infection']["incubation duration pdf"]['sd'] =  parameters["incub dur sd"] 
        
        
        if args.variance_mode == "mean":
            this_config['infection']["recovery time pdf"]['sd'] =  np.sqrt(parameters["recovery dur mean"])
            this_config['infection']["infectious duration pdf"]['sd'] =  np.sqrt(parameters["infectious dur mean"])
            this_config['infection']["latency duration pdf"]['sd'] =  np.sqrt(parameters["latency mean"])
            this_config['population']['social circle interactions pdf']["sd"] = np.sqrt(parameters["soc circ mean"])
            this_config['population']['social circle pdf']["sd"] = np.sqrt(parameters["soc circ mean"])
        elif args.variance_mode == "minimal":
            this_config['infection']["recovery time pdf"]['sd'] =  0.1
            this_config['infection']["infectious duration pdf"]['sd'] =  0.1
            this_config['infection']["latency duration pdf"]['sd'] =  0.1
            this_config['population']['social circle interactions pdf']["sd"] = 0.1
            this_config['population']['social circle pdf']["sd"] = 0.1
        else:
            raise RuntimeError("Unknown variance mode: {}".format(args.variance_mode))
        
        #this_config['infection']["incubation duration pdf"]['mean'] =  parameters["incubation mean"]
        
        this_config["population"]["re-use population"] = False
        contagion = Contagion(userconfig=this_config)
        contagion.sim()

        stats = pd.DataFrame(contagion.statistics)
        stats["is_recovered"] = stats["is_recovered"] + stats["is_recovering"]
        return stats

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
    
    def make_ad_distance(fields):
        distances = []
        for field in fields:
            def distance(simulation, data):
                return np.sum((np.abs(simulation[field] - data[field])))
            distances.append(distance)
        return pyabc.distance.AggregatedDistance(distances)
        

    sum_stat_func = make_sum_stats(fields)

    if args.distance == "summary":
        distance = pyabc.AdaptivePNormDistance(
            p=2,
            scale_function=pyabc.distance.median_absolute_deviation)
        fit_data = sum_stat_func(data)
        summary_statistics = sum_stat_func
    elif args.distance == "chi2":
        distance = make_chi2_distance(fields)
        fit_data = data
        summary_statistics = lambda x: x
    elif args.distance == "ad":
        distance = make_ad_distance(fields)
        fit_data = data
        summary_statistics = lambda x: x
    else:
        raise RuntimeError("Unknown distance type: {}".format(args.distance))

    prior = pyabc.Distribution(
        {"soc circ mean": pyabc.RV("uniform", 5, 15),
         "latency mean": pyabc.RV("uniform", 1, 10) ,
         "infectious dur mean": pyabc.RV("uniform", 1, 15),
         "incub dur mean": pyabc.RV("uniform", 1, 15),
         "incub dur sd": pyabc.RV("uniform", 1, 15),
         "recovery dur mean": pyabc.RV("uniform", 0.1, 10),
         "inf prob max": pyabc.RV("uniform", 0.1, 0.3)
        })

    client = Client(scheduler_file="scheduler.json")
    client.restart()
    # cluster = LocalCluster(n_workers=8, processes=True)
    # client = Client(cluster)

    # sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=8)
    sampler = DaskDistributedSampler(client, batch_size=1, client_max_jobs=800)
    population = pyabc.populationstrategy.AdaptivePopulationSize(
        150,
        max_population_size=300,
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
                       summary_statistics=summary_statistics,
                       eps=epsilon
                       )
    db_path = "sqlite:///" + os.path.join(os.environ["HOME"], "abc.db")

    logging.getLogger().setLevel("DEBUG")

    if "cont" in args:
        abc.load(db_path, args.cont)
    else:
        abc.new(db_path, fit_data, meta_info={"args": args})
        # abc.new(db_path, data)
    history1 = abc.run(max_nr_populations=20)
