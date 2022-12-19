from utils import config_parser

from .efficient_search import EfficientSearch
from .hals import HALS
from .refine_search import RefineSearch
from .uniform_distribution import UniformDistribution

config = config_parser()


def set_update_method(update_area):
    if config.update_method == "efficient_search":
        update_method = EfficientSearch(update_area)
    elif config.update_method == "hals":
        update_method = HALS(update_area)
    elif config.update_method == "refine_search":
        update_method = RefineSearch(update_area)
    elif config.update_method == "uniform_distribution":
        update_method = UniformDistribution(update_area)
    else:
        raise NotImplementedError(config.update_method)
    return update_method
