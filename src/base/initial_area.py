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

    def initialize(self, x: Tensor):
        self.batch, self.n_chanel, self.height, self.width = x.shape

        if config.update_area == "superpixel":
            self.superpixel = SuperpixelManager().cal_superpixel(x)
            self.level = np.zeros(self.batch, dtype=int)
            self.update_area = self.superpixel[np.arange(self.batch), self.level]
            self.n_update_area = self.update_area.max(axis=(1, 2))

            if config.update_method in ("greedy_local_search",):
                self.targets = []
                for idx in range(self.batch):
                    chanel = np.tile(np.arange(self.n_chanel), self.n_update_area[idx])
                    labels = np.repeat(
                        range(1, self.n_update_area[idx] + 1), self.n_chanel
                    )
                    _target = np.stack([chanel, labels], axis=1)
                    np.random.shuffle(_target)
                    self.targets.append(_target)
                self.checkpoint = np.array([len(t) for t in self.targets]) + 1
                breakpoint()
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
            h = np.sqrt(self.height * self.width * config.p_init).round().astype(int)
            r = np.random.randint(0, self.height - h)
            s = np.random.randint(0, self.width - h)
            self.update_area[:, r : r + h, s : s + h] = 1

            if config.update_method in ("greedy_local_search"):
                chanel = np.zeros(self.batch, dtype=int)
                labels = np.ones(self.batch, dtype=int)
                self.targets = np.stack([chanel, labels], axis=1)
                breakpoint()
            elif config.update_method in ("uniform_distribution"):
                self.targets = np.ones(self.batch, dtype=int)[:, None]
            else:
                raise ValueError(config.update_method)

        elif config.update_area == "divisional_square":
            assert False

        else:
            raise NotImplementedError(config.update_area)

        return self.update_area, self.targets
