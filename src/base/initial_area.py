import numpy as np
from torch import Tensor

from utils import config_parser, setup_logger

from .superpixel import SuperpixelManager

logger = setup_logger(__name__)
config = config_parser()


class InitialArea:
    def __init__(self):
        if config.update_area not in (
            "superpixel",
            "random_square",
            "split_square",
        ):
            raise NotImplementedError(config.update_area)

    def initialize(self, x: Tensor, forward: np.ndarray):
        self.batch, self.n_chanel, self.height, self.width = x.shape

        if config.update_area == "superpixel":
            self.superpixel = SuperpixelManager().cal_superpixel(x)
            self.level = np.zeros(self.batch, dtype=int)
            self.update_area = self.superpixel[np.arange(self.batch), self.level]
            n_update_area = self.update_area.max(axis=(1, 2))
            if config.channel_wise:
                self.targets = []
                for idx in range(self.batch):
                    chanel = np.tile(np.arange(self.n_chanel), n_update_area[idx])
                    labels = np.repeat(range(1, n_update_area[idx] + 1), self.n_chanel)
                    _target = np.stack([chanel, labels], axis=1)
                    self.targets.append(np.random.permutation(_target))
            else:
                self.targets = []
                for idx in range(self.batch):
                    labels = range(1, n_update_area[idx] + 1)
                    self.targets.append(np.random.permutation(labels))

        elif config.update_area == "random_square":
            self.half_point = (
                np.array([0.001, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8]) * config.step
            )
            self.update_area = np.zeros(
                (self.batch, self.height, self.width), dtype=bool
            )
            for idx in range(self.batch):
                n_half = (self.half_point < forward[idx]).sum()
                p = config.p_init / 2**n_half
                h = np.sqrt(p * self.height * self.width).round().astype(int)
                r = np.random.randint(0, self.height - h)
                s = np.random.randint(0, self.width - h)
                self.update_area[idx, r : r + h, s : s + h] = True
            if config.channel_wise:
                self.targets = np.random.permutation(np.arange(self.n_chanel))
            else:
                self.targets = np.ones(1, dtype=int)

        elif config.update_area == "split_square":
            breakpoint()
            self.split = config.initial_split
            assert self.height % self.split == 0
            h = self.height // self.split
            assert self.width % self.split == 0
            w = self.width // self.split
            self.update_area = np.arange(h * w).reshape(h, w)
            self.update_area = np.repeat(self.update_area, self.split, axis=0)
            self.update_area = np.repeat(self.update_area, self.split, axis=1)
            if config.channel_wise:
                chanel = np.tile(np.arange(self.n_chanel), h * w)
                labels = np.repeat(range(1, h * w + 1), self.n_chanel)
                self.targets = np.stack([chanel, labels], axis=1)
                np.random.shuffle(self.targets)
            else:
                self.targets = np.arange(1, h * w + 1)
                breakpoint()
                np.random.shuffle(self.targets)

        else:
            raise ValueError(config.update_method)

        return self.update_area, self.targets
