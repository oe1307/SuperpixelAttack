from utils import config_parser

from .initial_point import InitialPoint

config = config_parser()


class UpdateMethod(InitialPoint):
    def __init__(self):
        if config.update_method not in (
            "greedy_local_search",
            "accelerated_local_search",
            "refine_search",
            "uniform_distribution",
        ):
            raise NotImplementedError(config.update_method)

    def update(self):
        if config.update_method == "greedy_local_search":
            pass

        elif config.update_method == "accelerated_local_search":
            pass

        elif config.update_method == "refine_search":
            pass

        elif config.update_method == "uniform_distribution":
            pass

        else:
            raise ValueError(config.update_method)
