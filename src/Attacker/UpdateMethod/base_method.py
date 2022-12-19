import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class BaseMethod:
    def __init__(self, update_area):
        if config.initial_point not in ("random", "lower", "upper"):
            raise NotImplementedError(config.initial_point)
        self.update_area = update_area

    def set(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def initialize(self, x: Tensor, y: Tensor, lower: Tensor, upper: Tensor):
        self.batch, self.n_channel = x.shape[:2]
        self.y = y.clone()
        self.upper = upper.clone()
        self.lower = lower.clone()

        self.level = np.zeros(self.batch, dtype=int)
        self.area = self.update_area.initialize(x, self.level)
        if config.channel_wise:
            self.targets = []
            for idx in range(self.batch):
                n_update_area = self.area[idx].max()
                channel = np.tile(np.arange(self.n_channel), n_update_area)
                labels = np.repeat(range(1, n_update_area + 1), self.n_channel)
                _target = np.stack([channel, labels], axis=1)
                self.targets.append(np.random.permutation(_target))
        else:
            for idx in range(self.batch):
                _target = np.arange(1, self.area[idx].max())
                self.targets.append(np.random.permutation(_target))

        if config.initial_point == "random":
            is_upper = torch.randint_like(x, 0, 2, dtype=torch.bool)
            x_adv = torch.where(is_upper, upper, lower)
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        elif config.initial_point == "lower":
            is_upper = torch.zeros_like(x, dtype=torch.bool)
            x_adv = lower.clone()
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        elif config.initial_point == "upper":
            is_upper = torch.ones_like(x, dtype=torch.bool)
            x_adv = upper.clone()
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        else:
            raise ValueError(config.initial_point)

        self.is_upper_best = is_upper.clone()
        self.x_best = x_adv.clone()
        self.best_loss = loss.clone()
        return self.forward
