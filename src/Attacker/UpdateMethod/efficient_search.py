import time

import numpy as np
import torch

from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class EfficientSearch(BaseMethod):
    def __init__(self, update_area):
        super().__init__(update_area)
        config.forward_time = 0

    def step(self):
        is_upper = self.is_upper_best.clone()
        if config.channel_wise:
            for idx in range(self.batch):
                c, label = self.targets[idx][0]
                is_upper[idx, c, self.area[idx] == label] = ~is_upper[
                    idx, c, self.area[idx] == label
                ]
                self.targets[idx] = self.targets[idx][1:]
                if self.targets[idx].shape[0] == 0:
                    self.level[idx] += 1
                    self.area[idx] = self.update_area.update(idx, self.level[idx])
                    n_update_area = self.area[idx].max()
                    channel = np.tile(np.arange(self.n_channel), n_update_area)
                    labels = np.repeat(range(1, n_update_area + 1), self.n_channel)
                    self.targets[idx] = np.stack([channel, labels], axis=1)
            x_adv = torch.where(is_upper, self.upper, self.lower)
        else:
            is_upper = is_upper.permute(0, 2, 3, 1)
            for idx in range(self.batch):
                label = self.targets[idx][0]
                is_upper[idx, self.area[idx] == label] = ~is_upper[
                    idx, self.area[idx] == label
                ]
                self.targets[idx] = self.targets[idx][1:]
                if self.targets[idx].shape[0] == 0:
                    self.area[idx] = self.update_area.update(idx, self.level[idx])
                    self.targets[idx] = np.arange(1, self.area[idx].max())
        timekeeper = time.time()
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        config.forward_time += time.time() - timekeeper
        self.forward += 1
        update = loss >= self.best_loss
        self.is_upper_best[update] = is_upper[update]
        self.x_best[update] = x_adv[update]
        self.best_loss[update] = loss[update]
        return self.x_best, self.forward
