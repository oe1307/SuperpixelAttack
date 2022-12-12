import numpy as np
from torch import Tensor

from utils import config_parser, setup_logger

from .superpixel import SuperpixelManager

logger = setup_logger(__name__)
config = config_parser()


class UpdateArea:
    def __init__(self):
        if config.update_area not in (
            "superpixel",
            "random_square",
            "divisional_square",
        ):
            raise NotImplementedError(config.update_area)

    def initialize(self, x: Tensor):
        self.batch, self.n_chanel = x.shape[:2]

        if config.update_area == "superpixel":
            self.superpixel = SuperpixelManager().cal_superpixel(x)
            self.level = np.zeros(self.batch, dtype=int)
            self.update_area = self.superpixel[np.arange(self.batch), self.level]
            self.n_update_area = self.update_area.max(axis=(1, 2))

            self.targets = []
            for idx in range(self.batch):
                chanel = np.tile(np.arange(self.n_chanel), self.n_update_area[idx])
                labels = np.repeat(range(1, self.n_update_area[idx] + 1), self.n_chanel)
                _target = np.stack([chanel, labels], axis=1)
                np.random.shuffle(_target)
                self.targets.append(_target)
            self.checkpoint = np.array([len(t) for t in self.targets]) + 1

        elif config.update_area == "random_square":
            pass

        elif config.update_area == "divisional_square":
            pass

        else:
            raise NotImplementedError(config.update_area)

        return self.update_area, self.targets

    def next(self, forward: np.ndarray):
        if config.update_area == "superpixel":
            for idx in range(self.batch):
                if forward[idx] >= self.checkpoint[idx]:
                    self.level[idx] = min(self.level[idx] + 1, len(config.segments) - 1)
                    self.update_area[idx] = self.superpixel[idx, self.level[idx]]
                    self.n_update_area = self.update_area[idx].max()
                    chanel = np.tile(np.arange(self.n_chanel), self.n_update_area)
                    labels = np.repeat(range(1, self.n_update_area + 1), self.n_chanel)
                    self.targets[idx] = np.stack([chanel, labels], axis=1)
                    np.random.shuffle(self.targets[idx])
                    self.checkpoint[idx] += self.n_update_area * self.n_chanel
                else:
                    self.targets[idx] = np.delete(self.targets[idx], 0, axis=0)

        elif config.update_area == "random_square":
            pass

        elif config.update_area == "divisional_square":
            pass

        else:
            raise NotImplementedError(config.update_area)

        return self.update_area, self.targets
