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

    def initialize(self, x: Tensor, forward: np.ndarray):
        self.batch, self.n_chanel, self.height, self.width = x.shape

        if config.update_area == "superpixel":
            self.superpixel = SuperpixelManager().cal_superpixel(x)
            self.level = np.zeros(self.batch, dtype=int)
            self.update_area = self.superpixel[np.arange(self.batch), self.level]
            self.n_update_area = self.update_area.max(axis=(1, 2))

            if config.update_method in ("greedy_local_search",):
                assert False
            elif config.update_method in ("uniform_distribution",):
                self.targets = [np.arange(1, n + 1) for n in self.n_update_area]
                self.checkpoint = np.array([len(t) for t in self.targets]) + 1
                breakpoint()
            else:
                raise ValueError(config.update_method)

        elif config.update_area == "random_square":
            self.update_area = np.zeros(
                (self.batch, self.height, self.width), dtype=int
            )
            self.half_point = (
                np.array([0.001, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8]) * config.steps
            )
            for idx in range(self.batch):
                n_half = (self.half_point < forward[idx]).sum()
                p = config.p_init / 2**n_half
                h = np.sqrt(p * self.height * self.width).round().astype(int)
                r = np.random.randint(0, self.height - h)
                s = np.random.randint(0, self.width - h)
                self.update_area[:, r : r + h, s : s + h] = 1

            if config.update_method in ("greedy_local_search",):
                assert False
            elif config.update_method in ("uniform_distribution",):
                self.targets = np.ones(self.batch, dtype=int)[:, None]
                self.checkpoint = forward + 1
            else:
                raise ValueError(config.update_method)

        elif config.update_area == "divisional_square":
            assert False

        else:
            raise ValueError(config.update_method)

        return self.update_area, self.targets
