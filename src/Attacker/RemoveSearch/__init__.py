from utils import config_parser

from .base_remover import Remover
from .loss_search_remover import LossSearchRemover
from .saliency_remover import SaliencyRemover

config = config_parser()


def set_search_remover(update_area, update_method):
    if config.remove_search is None:
        return Remover(update_area, update_method)
    elif config.remove_search == "loss":
        return LossSearchRemover(update_area, update_method)
    elif config.remove_search == "saliency":
        return SaliencyRemover(update_area, update_method)
    else:
        raise NotImplementedError(config.remove_search)
