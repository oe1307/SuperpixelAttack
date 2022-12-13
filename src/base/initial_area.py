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

    def initialize(self, x: Tensor, forward: np.ndarray):
        self.batch, self.n_chanel, self.height, self.width = x.shape

        if config.update_area == "superpixel":
            self.superpixel = SuperpixelManager().cal_superpixel(x)
            self.level = np.zeros(self.batch, dtype=int)
            self.update_area = self.superpixel[np.arange(self.batch), self.level]
            self.targets = []
            for idx in range(self.batch):
                n_update_area = self.update_area[idx].max()
                chanel = np.tile(np.arange(self.n_chanel), n_update_area)
                labels = np.repeat(range(1, n_update_area + 1), self.n_chanel)
                _target = np.stack([chanel, labels], axis=1)
                self.targets.append(np.random.permutation(_target))
            self.checkpoint = forward + np.array([len(t) for t in self.targets])

        elif config.update_area == "random_square":
            self.half_point = (
                np.array([0.001, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8]) * config.steps
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
            self.targets = None

        elif config.update_area == "divisional_square":
            assert False

        else:
            raise ValueError(config.update_method)

        return self.update_area, self.targets
