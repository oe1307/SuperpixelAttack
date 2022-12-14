import numpy as np
from torch import Tensor

from utils import config_parser

config = config_parser()


class SplitSquare:
    def __init__(self):
        pass

    def initialize(self, x: Tensor, forward: np.ndarray):
        self.batch, self.n_channel, self.height, self.width = x.shape
        self.split = config.initial_split
        assert self.height % self.split == 0
        h = self.height // self.split
        assert self.width % self.split == 0
        w = self.width // self.split
        self.update_area = np.arange(h * w).reshape(h, w)
        self.update_area = np.repeat(self.update_area, self.split, axis=0)
        self.update_area = np.repeat(self.update_area, self.split, axis=1)
        if config.channel_wise:
            channel = np.tile(np.arange(self.n_channel), h * w)
            labels = np.repeat(range(h * w), self.n_channel)
            self.targets = np.stack([channel, labels], axis=1)
            np.random.shuffle(self.targets)
        else:
            self.targets = np.arange(h * w)
            np.random.shuffle(self.targets)
        return self.update_area, self.targets

    def next(self, forward: np.ndarray):
        if self.targets.shape[0] == 1:
            self.split //= 2
            h = self.height // self.split
            w = self.width // self.split
            self.update_area = np.arange(h * w).reshape(h, w)
            self.update_area = np.repeat(self.update_area, self.split, axis=0)
            self.update_area = np.repeat(self.update_area, self.split, axis=1)
            if config.channel_wise:
                channel = np.tile(np.arange(self.n_channel), h * w)
                labels = np.repeat(range(h * w), self.n_channel)
                self.targets = np.stack([channel, labels], axis=1)
                np.random.shuffle(self.targets)
            else:
                self.targets = np.arange(h * w)
                np.random.shuffle(self.targets)
        else:
            if config.channel_wise:
                self.targets = np.delete(self.targets, 0, axis=0)
            else:
                self.targets = np.delete(self.targets, 0)
        return self.update_area, self.targets
