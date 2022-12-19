import time

import numpy as np
import torch

from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class EfficientSearch(BaseMethod):
    def __init__(self):
        super().__init__()
        config.forward_time = 0

    def step(self, update_area: np.ndarray, targets):
        is_upper = self.is_upper_best.clone()
        if config.channel_wise:
            for idx in range(self.batch):
                c, label = targets[idx][0]
                is_upper[idx, c, update_area[idx] == label] = ~is_upper[
                    idx, c, update_area[idx] == label
                ]
                targets[idx] = targets[idx][1:]
                if targets[idx].shape[0] == 0:
                    self.new_area[idx] += 1
            x_adv = torch.where(is_upper, self.upper, self.lower)
        else:
            is_upper = is_upper.permute(0, 2, 3, 1)
            for idx in range(self.batch):
                label = targets[idx][0]
                is_upper[idx, update_area[idx] == label] = ~is_upper[
                    idx, update_area[idx] == label
                ]
                targets[idx] = targets[idx][1:]
                if targets[idx].shape[0] == 0:
                    self.new_area[idx] += 1
        timekeeper = time.time()
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        config.forward_time += time.time() - timekeeper
        self.forward += 1
        update = loss >= self.best_loss
        self.is_upper_best[update] = is_upper[update]
        self.x_best[update] = x_adv[update]
        self.best_loss[update] = loss[update]
        return self.x_best, self.forward, targets, self.new_area
