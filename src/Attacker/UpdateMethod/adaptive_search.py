import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class AdaptiveSearch:
    def __init__(self, update_area):
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
        for idx in range(self.batch):
            labels = np.unique(self.area[idx])
            labels = labels[labels != 0]
            channel = np.tile(np.arange(self.n_channel), len(labels))
            labels = np.repeat(labels, self.n_channel)
            channel_labels = np.stack([channel, labels], axis=1)
            self.targets.append(np.random.permutation(channel_labels))

        self.is_upper_best = torch.zeros_like(x, device=config.device, dtype=bool)
        self.x_best = lower.clone()
        pred = self.model(self.x_best).softmax(1)
        self.best_loss = self.criterion(pred, y)
        self.forward = np.ones(self.batch, dtype=int)
        return self.forward

    def step(self):
        is_upper = self.is_upper_best.clone()
        for idx in range(self.batch):
            c, label = self.targets[idx][0]
            self.targets[idx] = self.targets[idx][1:]
            is_upper[idx, c, self.area[idx] == label] = ~is_upper[
                idx, c, self.area[idx] == label
            ]
            if self.targets[idx].shape[0] == 0:
                self.level[idx] += 1
                self.area[idx] = self.update_area.update(idx, self.level[idx])
                labels = np.unique(self.area[idx])
                labels = labels[labels != 0]
                channel = np.tile(np.arange(self.n_channel), len(labels))
                labels = np.repeat(labels, self.n_channel)
                channel_labels = np.stack([channel, labels], axis=1)
                self.targets[idx] = np.random.permutation(channel_labels)
        x_adv = torch.where(is_upper, self.upper, self.lower)
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        self.forward += 1
        update = loss >= self.best_loss
        self.is_upper_best[update] = is_upper[update]
        self.x_best[update] = x_adv[update]
        self.best_loss[update] = loss[update]
        return self.x_best, self.forward
