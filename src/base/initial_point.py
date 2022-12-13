import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class InitialPoint:
    def __init__(self):
        if config.initial_point not in ("random", "lower", "upper"):
            raise NotImplementedError(config.initial_point)

    def set(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def initialize(self, x: Tensor, y: Tensor, lower: Tensor, upper: Tensor):
        self.batch = x.shape[0]
        self.y = y
        self.upper = upper
        self.lower = lower

        if config.initial_point == "random":
            self.is_upper = torch.randint_like(x, 0, 2, dtype=torch.bool)
            self.x_adv = torch.where(self.is_upper, upper, lower)
            pred = self.model(self.x_adv).softmax(1)
            self.loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        elif config.initial_point == "lower":
            self.is_upper = torch.zeros_like(x, dtype=torch.bool)
            self.x_adv = lower.clone()
            pred = self.model(self.x_adv).softmax(1)
            self.loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        elif config.initial_point == "upper":
            self.is_upper = torch.ones_like(x, dtype=torch.bool)
            self.x_adv = upper.clone()
            pred = self.model(self.x_adv).softmax(1)
            self.loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        else:
            raise ValueError(config.initial_point)

        self.is_upper_best = self.is_upper.clone()
        self.x_best = self.x_adv.clone()
        self.best_loss = self.loss.clone()
        return self.forward
