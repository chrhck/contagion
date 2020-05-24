import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns
import networkx as nx
import yaml
from networkx.algorithms import approximation as nx_approx
from tqdm.notebook import tqdm
from copy import deepcopy
# Adding path to module
sys.path.append("../")
# picture path
PICS = '../pics/'

# Module imports
from contagion import Contagion, config
from contagion.config import _baseconfig
from contagion.plotting import plot_infection_history
from memory_profiler import profile
my_config = yaml.safe_load(open("test_social_graph_cpp_params.yaml"))


contagion = Contagion(my_config)
contagion.sim()
results = pd.DataFrame(contagion.statistics)


