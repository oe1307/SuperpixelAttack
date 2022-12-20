import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class BaseMethod:
    def __init__(self, update_area):
        if config.initial_point not in ("random", "lower", "upper", "stripes"):
            raise NotImplementedError(config.initial_point)
        self.update_area = update_area

    def set(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def initialize(self, x: Tensor, y: Tensor, lower: Tensor, upper: Tensor):
        self.batch, self.n_channel, self.height, self.width = x.shape
        self.y = y.clone()
        self.upper = upper.clone()
        self.lower = lower.clone()

        self.level = np.zeros(self.batch, dtype=int)
        self.area = self.update_area.initialize(x, self.level)
        self.targets = []
        if config.channel_wise:
            for idx in range(self.batch):
                labels = np.unique(self.area[idx])
                labels = labels[labels != 0]
                channel = np.tile(np.arange(self.n_channel), len(labels))
                labels = np.repeat(labels, self.n_channel)
                if config.shuffle:
                    channel_labels = np.stack([channel, labels], axis=1)
                    self.targets.append(np.random.permutation(channel_labels))
                else:
                    self.targets.append(np.stack([channel, labels], axis=1))
        else:
            for idx in range(self.batch):
                labels = np.unique(self.area[idx])
                if config.shuffle:
                    labels = labels[labels != 0]
                    self.targets.append(np.random.permutation(labels))
                else:
                    self.targets.append(labels[labels != 0])

        if config.initial_point == "random":
            is_upper = torch.randint_like(x, 0, 2, device=config.device, dtype=bool)
            x_adv = torch.where(is_upper, upper, lower)
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        elif config.initial_point == "lower":
            is_upper = torch.zeros_like(x, device=config.device, dtype=bool)
            x_adv = lower.clone()
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        elif config.initial_point == "upper":
            is_upper = torch.ones_like(x, device=config.device, dtype=bool)
            x_adv = upper.clone()
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        elif config.initial_point == "stripes":
            is_upper = torch.randint(
                0,
                2,
                (self.batch, self.n_channel, 1, self.width),
                device=config.device,
                dtype=bool,
            ).repeat_interleave(self.height, dim=2)
            x_adv = torch.where(is_upper, upper, lower)
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            self.forward = np.ones(self.batch, dtype=int)

        else:
            raise ValueError(config.initial_point)

        self.is_upper_best = is_upper.clone()
        self.x_best = x_adv.clone()
        self.best_loss = loss.clone()
        return self.forward
