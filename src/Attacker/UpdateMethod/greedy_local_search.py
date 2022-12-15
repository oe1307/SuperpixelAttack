import time

import numpy as np
import torch

from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class GreedyLocalSearch(BaseMethod):
    def __init__(self):
        super().__init__()
        config.forward_time = 0

    def step(self, update_area: np.ndarray, targets):
        is_upper = self.is_upper_best.clone()
        if config.update_area == "superpixel" and config.channel_wise:
            for idx in range(self.batch):
                c, label = targets[idx][0]
                is_upper[idx, c, update_area[idx] == label] = ~is_upper[
                    idx, c, update_area[idx] == label
                ]
                targets[idx] = targets[idx][1:]
            x_adv = torch.where(is_upper, self.upper, self.lower)
        elif config.update_area == "superpixel":
            is_upper = is_upper.permute(0, 2, 3, 1)
            labels = [t[0] for t in targets]
            labels = np.repeat(labels, update_area.shape[1])
            labels = np.repeat(labels, update_area.shape[2])
            labels = labels.reshape(update_area.shape)
            is_upper[update_area == labels] = ~is_upper[update_area == labels]
            is_upper = is_upper.permute(0, 3, 1, 2)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            targets = [targets[idx][1:] for idx in range(self.batch)]
        elif config.update_area == "split_square" and config.channel_wise:
            is_upper = is_upper.permute(1, 2, 3, 0)
            c, label = targets[0]
            is_upper[c, update_area == label] = ~is_upper[c, update_area == label]
            is_upper = is_upper.permute(3, 0, 1, 2)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            targets = targets[1:]
        elif config.update_area == "split_square":
            is_upper = is_upper.permute(2, 3, 0, 1)
            label = targets[0]
            is_upper[update_area == label] = ~is_upper[update_area == label]
            is_upper = is_upper.permute(2, 3, 0, 1)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            targets = targets[1:]
        elif config.update_area == "saliency_map" and config.channel_wise:
            for idx in range(self.batch):
                c, label = targets[idx][0]
                is_upper[idx, c, update_area[idx] == label] = ~is_upper[
                    idx, c, update_area[idx] == label
                ]
                targets[idx] = targets[idx][1:]
            x_adv = torch.where(is_upper, self.upper, self.lower)
        elif config.update_area == "saliency_map":
            is_upper = is_upper.permute(0, 2, 3, 1)
            labels = [t[0] for t in targets]
            labels = np.repeat(labels, update_area.shape[1])
            labels = np.repeat(labels, update_area.shape[2])
            labels = labels.reshape(update_area.shape)
            is_upper[update_area == labels] = ~is_upper[update_area == labels]
            is_upper = is_upper.permute(0, 3, 1, 2)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            targets = [targets[idx][1:] for idx in range(self.batch)]
        elif config.update_area == "random_square" and config.channel_wise:
            is_upper = is_upper.permute(1, 0, 2, 3)
            c = targets[0]
            is_upper[c, update_area] = ~is_upper[c, update_area]
            is_upper = is_upper.permute(1, 0, 2, 3)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            targets = targets[1:]
        elif config.update_area == "random_square":
            is_upper = is_upper.permute(0, 2, 3, 1)
            is_upper[update_area] = ~is_upper[update_area]
            is_upper = is_upper.permute(0, 3, 1, 2)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            targets = targets[1:]
        timekeeper = time.time()
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        config.forward_time += time.time() - timekeeper
        self.forward += 1
        update = loss >= self.best_loss
        self.is_upper_best[update] = is_upper[update]
        self.x_best[update] = x_adv[update]
        self.best_loss[update] = loss[update]
        return self.x_best, self.forward, targets
