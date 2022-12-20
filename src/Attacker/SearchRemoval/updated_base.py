import numpy as np
import torch

from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class UpdatedBaseRemover(BaseMethod):
    def __init__(self, update_area):
        super().__init__(update_area)
        config.forward_time = 0

    def initialize(self, x, y, lower, upper):
        super().initialize(x, y, lower, upper)
        self.updated = np.ones(self.x_best.shape)
        self.searched = np.ones(self.x_best.shape)
        return self.forward

    def step(self):
        is_upper = self.is_upper_best.clone()
        for idx in range(self.batch):
            c, label = self.targets[idx][0]
            probability = (
                0.5
                * (
                    self.updated[idx, c, self.area[idx] == label].sum()
                    / self.searched[idx, c, self.area[idx] == label].sum()
                )
                + 0.5
            )
            if np.random.rand() > probability:
                continue
            is_upper[idx, c, self.area[idx] == label] = ~is_upper[
                idx, c, self.area[idx] == label
            ]
            self.forward[idx] += 1
        x_adv = torch.where(is_upper, self.upper, self.lower)
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
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
