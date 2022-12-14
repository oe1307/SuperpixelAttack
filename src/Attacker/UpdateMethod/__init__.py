from utils import config_parser

from .greedy_local_search import GreedyLocalSearch
from .hals import HALS
from .refine_search import RefineSearch
from .uniform_distribution import UniformDistribution

config = config_parser()


def set_update_method():
    if config.update_method == "greedy_local_search":
        update_method = GreedyLocalSearch()
    elif config.update_method == "hals":
        update_method = HALS()
    elif config.update_method == "refine_search":
        update_method = RefineSearch()
    elif config.update_method == "uniform_distribution":
        update_method = UniformDistribution()
    else:
        raise NotImplementedError()
    return update_method