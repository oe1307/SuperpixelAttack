import numpy as np

from .base_method import BaseMethod


class GreedyLocalSearch(BaseMethod):
    def __init__(self):
        super().__init__()

    def step(self, update_area: np.ndarray, targets):
        raise NotImplementedError()
