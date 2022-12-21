import time

import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class AdaptiveSearch:
    def __init__(self, update_area):
        self.update_area = update_area
        config.cal_forward_time = 0

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

        self.searched, self.updated = np.ones(x.shape), np.ones(x.shape)
        return self.forward

    def step(self):
        is_upper = self.is_upper_best.clone()
        for idx in range(self.batch):
            c, label = self.targets[idx][0]
            updated = self.updated[idx, c, self.area[idx] == label].sum()
            searched = self.searched[idx, c, self.area[idx] == label].sum()
            probability = 0.5 * (updated / searched) + 0.5
            if np.random.rand() > probability and config.removal:
                continue
            is_upper[idx, c, self.area[idx] == label] = ~is_upper[
                idx, c, self.area[idx] == label
            ]
            self.forward[idx] += 1
        x_adv = torch.where(is_upper, self.upper, self.lower)
        timekeeper = time.time()
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        config.cal_forward_time += time.time() - timekeeper
        update = loss >= self.best_loss
        self.is_upper_best[update] = is_upper[update]
        self.x_best[update] = x_adv[update]
        self.best_loss[update] = loss[update]
        for idx in range(self.batch):
            c, label = self.targets[idx][0]
            self.targets[idx] = self.targets[idx][1:]
            self.updated[idx, c, self.area[idx] == label] += update[idx].item()
            self.searched[idx, c, self.area[idx] == label] += 1
            if self.targets[idx].shape[0] == 0:
                self.level[idx] += 1
                self.area[idx] = self.update_area.update(idx, self.level[idx])
                labels = np.unique(self.area[idx])
                labels = labels[labels != 0]
                channel = np.tile(np.arange(self.n_channel), len(labels))
                labels = np.repeat(labels, self.n_channel)
                channel_labels = np.stack([channel, labels], axis=1)
                self.targets[idx] = np.random.permutation(channel_labels)
        return self.x_best, self.forward
