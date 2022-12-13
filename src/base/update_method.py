from typing import List

import numpy as np

from utils import config_parser, setup_logger

from .initial_point import InitialPoint

logger = setup_logger(__name__)
config = config_parser()


class UpdateMethod(InitialPoint):
    def __init__(self):
        super().__init__()

        if config.update_area not in (
            "superpixel",
            "random_square",
            "divisional_square",
        ):
            raise NotImplementedError(config.update_area)

        if config.update_method not in (
            "greedy_local_search",
            "accelerated_local_search",
            "refine_search",
            "uniform_distribution",
        ):
            raise NotImplementedError(config.update_method)

    def step(self, update_area: np.ndarray, targets: List[np.ndarray]):
        if config.update_method == "greedy_local_search":
            assert False

        elif config.update_method == "accelerated_local_search":
            assert False

        elif config.update_method == "refine_search":
            assert False

        elif config.update_method == "uniform_distribution":
            if config.update_area == "superpixel":
                breakpoint()
            elif config.update_area == "random_square":
                breakpoint()

        else:
            raise ValueError(config.update_method)

        return self.x_best, self.forward
