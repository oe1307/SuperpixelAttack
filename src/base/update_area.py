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
        self.batch, self.n_chanel, self.height, self.width = x.shape

        if config.update_area == "superpixel":
            self.superpixel = SuperpixelManager().cal_superpixel(x)
            self.level = np.zeros(self.batch, dtype=int)
            self.update_area = self.superpixel[np.arange(self.batch), self.level]
            self.n_update_area = self.update_area.max(axis=(1, 2))

            targets = []
            for idx in range(self.batch):
                chanel = np.tile(np.arange(self.n_chanel), self.n_update_area[idx])
                labels = np.repeat(range(1, self.n_update_area[idx] + 1), self.n_chanel)
                _target = np.stack([chanel, labels], axis=1)
                np.random.shuffle(_target)
                targets.append(_target)
            self.checkpoint = np.array([len(t) for t in targets]) + 1

        elif config.update_area == "random_square":
            self.update_area = np.zeros((self.batch, self.height, self.width))
            h = np.sqrt(self.height * self.width * config.p_init).round().astype(int)
            r = np.random.randint(0, self.height - h)
            s = np.random.randint(0, self.width - h)
            self.update_area[:, r : r + h, s : s + h] = 1
            chanel, labels = np.zeros(self.batch), np.ones(self.batch)
            targets = np.stack([chanel, labels], axis=1)[:, None, :]

        elif config.update_area == "divisional_square":
            pass

        else:
            raise NotImplementedError(config.update_area)

        return self.update_area, targets

    def next(self, forward: np.ndarray, targets: np.ndarray):
        if config.update_area == "superpixel":
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    self.level[idx] = min(self.level[idx] + 1, len(config.segments) - 1)
                    self.update_area[idx] = self.superpixel[idx, self.level[idx]]
                    self.n_update_area = self.update_area[idx].max()
                    chanel = np.tile(np.arange(self.n_chanel), self.n_update_area)
                    labels = np.repeat(range(1, self.n_update_area + 1), self.n_chanel)
                    targets[idx] = np.stack([chanel, labels], axis=1)
                    np.random.shuffle(targets[idx])
                    self.checkpoint[idx] += self.n_update_area * self.n_chanel
                else:
                    targets[idx] = np.delete(targets[idx], 0, axis=0)

        elif config.update_area == "random_square":
            assert (np.ones_like(forward) * forward[0] == forward).all()
            self.update_area = np.zeros((self.batch, self.height, self.width))
            half_point = (
                np.array([0.001, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8]) * config.steps
            )
            p = config.p_init * (half_point < forward.min()).sum() / len(half_point)
            h = np.sqrt(self.height * self.width * p).round().astype(int)
            r = np.random.randint(0, self.height - h)
            s = np.random.randint(0, self.width - h)
            self.update_area[:, r : r + h, s : s + h] = 1
            chanel, labels = np.zeros(self.batch), np.ones(self.batch)
            targets = np.stack([chanel, labels], axis=1)[:, None, :]

        elif config.update_area == "divisional_square":
            pass

        else:
            raise NotImplementedError(config.update_area)

        return self.update_area, targets
