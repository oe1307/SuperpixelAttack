import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class InitialPoint:
    def __init__(self):
        if config.initial_point not in (
            "random",
            "lower",
            "upper",
        ):
            raise NotImplementedError(config.initial_point)

    def set(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def initialize(
        self,
        x: Tensor,
        y: Tensor,
        update_area: np.ndarray,
        lower: Tensor,
        upper: Tensor,
    ):
        batch, n_chanel = x.shape[:2]
        n_update_area = update_area.max(axis=(1, 2))

        if config.initial_point == "random":
            is_upper = torch.randint(0, 2, x.shape, dtype=torch.bool)
            x_adv = torch.where(is_upper, upper, lower)
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(batch, dtype=np.int)

        elif config.initial_point == "lower":
            is_upper = torch.zeros_like(x, dtype=torch.bool)
            x_adv = lower.clone()
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(batch, dtype=np.int)

        elif config.initial_point == "upper":
            is_upper = torch.ones_like(x, dtype=torch.bool)
            x_adv = upper.clone()
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(batch, dtype=np.int)

        else:
            raise ValueError(config.initial_point)

        self.is_upper_best = is_upper.clone()
        self.x_best = x_adv.clone()
        self.best_loss = loss.clone()

        self.targets = []
        for idx in range(batch):
            chanel = np.tile(np.arange(n_chanel), n_update_area[idx])
            labels = np.repeat(range(1, n_update_area[idx] + 1), n_chanel)
            _target = np.stack([chanel, labels], axis=1)
            np.random.shuffle(_target)
            self.targets.append(_target)
        self.checkpoint = [len(_target) for _target in self.targets]
        self.checkpoint = np.array(self.checkpoint) + self.forward
        self.pre_checkpoint = self.forward
