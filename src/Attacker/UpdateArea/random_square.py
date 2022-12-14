import numpy as np
from torch import Tensor

from utils import config_parser

config = config_parser()


class RandomSquare:
    def __init__(self):
        self.half_point = (
            np.array([0.001, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8]) * config.step
        )

    def initialize(self, x: Tensor, forward: np.ndarray):
        self.batch, self.n_channel, self.height, self.width = x.shape
        self.update_area = np.zeros((self.batch, self.height, self.width), dtype=bool)
        for idx in range(self.batch):
            n_half = (self.half_point < forward[idx]).sum()
            p = config.p_init / 2**n_half
            h = np.sqrt(p * self.height * self.width).round().astype(int)
            r = np.random.randint(0, self.height - h)
            s = np.random.randint(0, self.width - h)
            self.update_area[idx, r : r + h, s : s + h] = True
        if config.channel_wise:
            targets = np.random.permutation(np.arange(self.n_channel))
        else:
            targets = np.ones(1, dtype=int)
        return self.update_area, targets

    def next(self, forward: np.ndarray, targets):
        if targets.shape[0] == 0:
            self.update_area = np.zeros_like(self.update_area)
            for idx in range(self.batch):
                n_half = (self.half_point < forward[idx]).sum()
                p = config.p_init / 2**n_half
                h = np.sqrt(p * self.height * self.width).round().astype(int)
                r = np.random.randint(0, self.height - h)
                s = np.random.randint(0, self.width - h)
                self.update_area[idx, r : r + h, s : s + h] = True
            if config.channel_wise:
                targets = np.random.permutation(np.arange(self.n_channel))
            else:
                targets = np.ones(1, dtype=int)
        return self.update_area, targets
