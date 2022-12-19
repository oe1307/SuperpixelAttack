import numpy as np
from torch import Tensor

from utils import config_parser

config = config_parser()


class RandomSquare:
    def __init__(self):
        self.half_point = (
            np.array([0.001, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8]) * config.step
        )

    def initialize(self, x: Tensor, level: np.ndarray):
        self.batch, self.n_channel, self.height, self.width = x.shape
        update_area = np.zeros((self.batch, self.height, self.width), dtype=int)
        for idx in range(self.batch):
            n_half = (self.half_point < level[idx]).sum()
            p = config.p_init / 2**n_half
            h = np.sqrt(p * self.height * self.width).round().astype(int)
            r = np.random.randint(0, self.height - h)
            s = np.random.randint(0, self.width - h)
            update_area[idx, r : r + h, s : s + h] = 1
        return update_area

    def update(self, idx: int, level: np.ndarray):
        update_area = np.zeros((self.height, self.width), dtype=int)
        if config.channel_wise:
            n_half = (self.half_point < level * self.n_channel).sum()
        else:
            n_half = (self.half_point < level).sum()
        p = config.p_init / 2**n_half
        h = np.sqrt(p * self.height * self.width).round().astype(int)
        r = np.random.randint(0, self.height - h)
        s = np.random.randint(0, self.width - h)
        update_area[r : r + h, s : s + h] = 1
        return update_area
