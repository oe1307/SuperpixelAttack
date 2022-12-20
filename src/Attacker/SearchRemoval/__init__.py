from utils import config_parser

from .updated_base import UpdatedBaseRemover

config = config_parser()


def set_search_method(update_area):
    if config.removal == "wasted":
        update_method = UpdatedBaseRemover(update_area)
    else:
        raise NotImplementedError(config.update_method)
    return update_method
