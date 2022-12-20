from utils import config_parser

from .loss_based_remover import LossBasedRemover

config = config_parser()


def set_search_method(update_area):
    if config.search_method == "loss_based":
        update_method = LossBasedRemover(update_area)
    else:
        raise NotImplementedError(config.update_method)
    return update_method


__all__ = ["LossBasedRemover"]
