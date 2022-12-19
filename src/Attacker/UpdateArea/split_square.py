import numpy as np
from torch import Tensor

from utils import config_parser

config = config_parser()


class SplitSquare:
    def __init__(self):
        pass

    def initialize(self, x: Tensor, level: np.ndarray):
        self.batch, self.n_channel, self.height, self.width = x.shape
        n_split = max(config.initial_split // 2**level, 1)
        assert self.height % n_split == 0
        h = self.height // n_split
        assert self.width % n_split == 0
        w = self.width // n_split
        update_area = np.arange(h * w).reshape(1, h, w)
        update_area = np.repeat(update_area, self.batch, axis=0)
        update_area = np.repeat(update_area, n_split, axis=0)
        update_area = np.repeat(update_area, n_split, axis=1)
        assert update_area.shape == (self.batch, self.height, self.width)
        return update_area

    def update(self, idx: int, level: np.ndarray):
        n_split = max(config.initial_split // 2**level, 1)
        assert self.height % n_split == 0
        h = self.height // n_split
        assert self.width % n_split == 0
        w = self.width // n_split
        update_area = np.arange(h * w).reshape(h, w)
        update_area = np.repeat(update_area, n_split, axis=0)
        update_area = np.repeat(update_area, n_split, axis=1)
        return update_area
