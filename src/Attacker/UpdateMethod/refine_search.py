import heapq

import numpy as np
import torch

from utils import config_parser, pbar

from .base_method import BaseMethod

config = config_parser()


class RefineSearch(BaseMethod):
    def __init__(self):
        super().__init__()
        self.local_search = True
        if config.update_area == "random_square":
            raise NotImplementedError("RefineSearch does not support random_square")

    def step(self, update_area: np.ndarray, targets):
        breakpoint()

        return self.x_best, self.forward
