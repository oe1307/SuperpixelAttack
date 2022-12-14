import numpy as np
import torch

from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class GreedyLocalSearch(BaseMethod):
    def __init__(self):
        super().__init__()

    def step(self, update_area: np.ndarray, targets):
        self.is_upper = self.is_upper_best.clone()
        if config.update_area == "superpixel" and config.channel_wise:
            for idx in range(self.batch):
                c, label = targets[idx][0]
                self.is_upper[idx, c, update_area[idx] == label] = ~self.is_upper[
                    idx, c, update_area[idx] == label
                ]
            self.x_adv = torch.where(self.is_upper, self.upper, self.lower)
        elif config.update_area == "superpixel":
            self.is_upper = self.is_upper.permute(0, 2, 3, 1)
            labels = [t[0] for t in targets]
            labels = np.repeat(labels, update_area.shape[1])
            labels = np.repeat(labels, update_area.shape[2])
            labels = labels.reshape(update_area.shape)
            self.is_upper[update_area == labels] = ~self.is_upper[update_area == labels]
            self.is_upper = self.is_upper.permute(0, 3, 1, 2)
            self.x_adv = torch.where(self.is_upper, self.upper, self.lower)
        elif config.update_area == "random_square" and config.channel_wise:
            self.is_upper = self.is_upper.permute(1, 0, 2, 3)
            c = targets[0]
            self.is_upper[c, update_area] = ~self.is_upper[c, update_area]
            self.is_upper = self.is_upper.permute(1, 0, 2, 3)
            self.x_adv = torch.where(self.is_upper, self.upper, self.lower)
        elif config.update_area == "random_square":
            self.is_upper = self.is_upper.permute(0, 2, 3, 1)
            self.is_upper[update_area] = ~self.is_upper[update_area]
            self.is_upper = self.is_upper.permute(0, 3, 1, 2)
            self.x_adv = torch.where(self.is_upper, self.upper, self.lower)
        elif config.update_area == "split_square" and config.channel_wise:
            self.is_upper = self.is_upper.permute(1, 0, 2, 3)
            c, label = targets[0]
            self.is_upper[c, update_area == label] = ~self.is_upper[
                c, update_area == label
            ]
            self.is_upper = self.is_upper.permute(1, 0, 2, 3)
            self.x_adv = torch.where(self.is_upper, self.upper, self.lower)
        elif config.update_area == "split_square":
            self.is_upper = self.is_upper.permute(0, 2, 3, 1)
            label = targets[0]
            self.is_upper[update_area == label] = ~self.is_upper[update_area == label]
            self.is_upper = self.is_upper.permute(0, 3, 1, 2)
            self.x_adv = torch.where(self.is_upper, self.upper, self.lower)
        pred = self.model(self.x_adv).softmax(dim=1)
        self.loss = self.criterion(pred, self.y)
        self.forward += 1
        update = self.loss >= self.best_loss
        self.is_upper_best[update] = self.is_upper[update]
        self.x_best[update] = self.x_adv[update]
        self.best_loss[update] = self.loss[update]
        return self.x_best, self.forward
