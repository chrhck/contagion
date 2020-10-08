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
import scipy.stats

logging.basicConfig(level="WARN")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Continue runid",
                        default=argparse.SUPPRESS,
                        const=None, nargs="?", dest="cont", type=int)
    parser.add_argument("-d", choices=["summary", "chi2", "ad"], dest="distance", default="summary")
    parser.add_argument("--soft_ld", help="add soft lockdown phase", action="store_true", dest="soft_ld")
    parser.add_argument("--full", help="use full dataset", action="store_true", dest="full")
    args = parser.parse_args()
    my_config = yaml.unsafe_load(open("fit_germany_conf.yaml"))
   

    if "cont" in args:
        my_config["population"]["re-use population"] = True

    data = pd.read_csv("data_germany.csv")
    data["Date"] = pd.to_datetime(data['Date'])
    if args.full:
        data = data.set_index("Date").loc[pd.to_datetime('2020-02-24'):]
    else:
        data = data.set_index("Date").loc[pd.to_datetime('2020-02-24'):pd.to_datetime('2020-05-11')]
    fields = ["is_recovered", "is_infected_total", "is_dead"]
    #data = {field: np.asarray(contagion.statistics[field]) for field in fields}
    data = {
        "is_recovered": data["recovered"]/80E6*1E5,
        "is_infected_total": data["total_cases"]/80E6*1E5,
        "is_dead": data["total_deaths"]/80E6*1E5
    }
    
    def model(parameters):
        infect_pdf = scipy.stats.gamma(
            2.08,
            scale=1.56,
        )
        this_config = dict(_baseconfig)
        this_config.update(my_config)
        this_config['population']['population size'] = 100000

        ppl_met_per_day_pdf = scipy.stats.gamma(2, scale=parameters["soc circ mean"]/2)
        this_config["population"]["social circle interactions pdf"] = {
            "class": "Gamma",
            "mean": ppl_met_per_day_pdf.mean(),
            "sd": ppl_met_per_day_pdf.std(),
            "upper": np.inf
        }


        this_config["infection"]["latency duration pdf"] = {
            "upper": np.floor(parameters["latency mean"]+1),
            "lower": np.floor(parameters["latency mean"]),
            "class": "Uniform"
        }

        this_config["infection"]["infection probability pdf"] = {
            "class": "Gamma",
            "scaling": (infect_pdf.ppf(0.99))*parameters["transmission prob"],
            "mean":  infect_pdf.mean(),
            "sd":  infect_pdf.std()
        }


        #this_config['infection']["infectious duration pdf"]['mean'] =  parameters["infectious dur mean"]        
        this_config['infection']["recovery time pdf"]['mean'] =  parameters["recovery dur mean"]        
        this_config['infection']["incubation duration pdf"]['mean'] =  parameters["incub dur mean"]
        this_config['infection']["incubation duration pdf"]['sd'] =  parameters["incub dur sd"]    
        this_config['infection']["hospitalization probability pdf"]["mean"] = parameters["hospit prob mean"]
        this_config['infection']["hospitalization duration pdf"]["mean"] = parameters["hospit dur mean"]
        this_config['infection']["hospitalization duration pdf"]["sd"] = parameters["hospit dur sd"]
        this_config["infection"]["mortality prob pdf"]["mean"] = parameters["mort mean"]
        this_config["infection"]["will have symptoms prob pdf"]["mean"] = parameters["symp prob mean"]
   
        this_config["scenario"]["class"] = "SocialDistancing"

        start_scaling = int(parameters["t_start_dist"])
        end_scaling = start_scaling + int(parameters["scaling_dur"])
        final_inf_per_day = parameters["int_per_day_dist"]

        slope = (parameters["soc circ mean"] - final_inf_per_day) / (start_scaling-end_scaling)
        offset =  parameters["soc circ mean"] - slope * start_scaling
        t_steps = np.arange(start_scaling)

        scalings =  list(slope*t_steps + offset)
        t_steps = list(t_steps)


        if args.soft_ld:
            start_soft = int(parameters["t_start_soft"])
            soft_inf_per_day = parameters["int_per_day_soft"]
            t_steps.append(start_soft)
            scalings.append(soft_inf_per_day)

        this_config["scenario"]["t_steps"] = t_steps
        this_config["scenario"]["contact_rate_scalings"] = scalings
        this_config["scenario"]["n_contacts_per_day_baseline"] = parameters["soc circ mean"]

            # this_config["scenario"]["class"] = "StandardScenario"

        this_config["population"]["re-use population"] = False
        this_config["population"]["store population"] = False
        this_config["measures"]["tracing efficiency"]
        contagion = Contagion(userconfig=this_config)
        contagion.sim()

        stats = pd.DataFrame(contagion.statistics)
        stats["is_infected_total"] = stats["is_index_case"]
        
        time_shift = int(parameters["timeshift"])
        if time_shift < 0:
            stats = stats.iloc[-time_shift:].reset_index()
        else:            
            zero_rows = pd.DataFrame({col: np.zeros(time_shift) for col in stats.columns})
            stats = pd.concat([zero_rows, stats]).reset_index()
        return stats.iloc[:len(data["is_recovered"])]
    

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
        {"soc circ mean": pyabc.RV("uniform", 5, 10),
         "latency mean": pyabc.RV("uniform", 1, 9) ,
         "incub dur mean": pyabc.RV("uniform", 1, 14),
         "incub dur sd": pyabc.RV("uniform", 1, 14),
         "recovery dur mean": pyabc.RV("uniform", 0.1, 10),
         "transmission prob": pyabc.RV("uniform", 1E-2, 0.1),
         "mort mean": pyabc.RV("uniform", 0.01, 0.1),
        
         "hospit prob mean": pyabc.RV("uniform", 0.01, 0.1),
       
         "hospit dur mean": pyabc.RV("uniform", 5, 20),
         "hospit dur sd": pyabc.RV("uniform", 1, 10),
         "symp prob mean": pyabc.RV("uniform", 0.1, 0.6),

         "timeshift": pyabc.RV("uniform", 0, 10),
         "t_start_dist": pyabc.RV("uniform", 20, 9),
         "scaling_dur": pyabc.RV("uniform", 1, 14),
         "int_per_day_dist": pyabc.RV("uniform", 0.1, 0.9),
         "tracing eff": pyabc.RV("uniform", 0, 1),
         "t_start_soft": pyabc.RV("uniform", 40, 30),
         "int_per_day_soft": pyabc.RV("uniform", 0.1, 0.9),
        })

    client = Client(scheduler_file="scheduler.json")
    client.restart()
    # cluster = LocalCluster(n_workers=8, processes=True)
    # client = Client(cluster)

    # sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=8)
    sampler = DaskDistributedSampler(client, batch_size=1, client_max_jobs=800)
    """
    population = pyabc.populationstrategy.AdaptivePopulationSize(
        150,
        max_population_size=1000,
        mean_cv=0.1,
        n_bootstrap=10,
        client=client)
    """
    population = 300
    epsilon = pyabc.epsilon.QuantileEpsilon(alpha=0.5)
    abc = pyabc.ABCSMC(model, prior, distance,
                       population_size=population, sampler=sampler,
                       acceptor=pyabc.UniformAcceptor(
                           use_complete_history=True
                        ),
                       summary_statistics=summary_statistics,
                       eps=epsilon
                       )
    db_path = "sqlite:///" + os.path.join(os.environ["HOME"], "abc_ger.db")

    logging.getLogger().setLevel("DEBUG")

    if "cont" in args:
        abc.load(db_path, args.cont)
    else:
        abc.new(db_path, fit_data, meta_info={"args": args})
        # abc.new(db_path, data)
    history1 = abc.run(max_nr_populations=20)
