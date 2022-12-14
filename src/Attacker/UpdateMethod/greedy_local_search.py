import torch
import numpy as np

from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class GreedyLocalSearch(BaseMethod):
    def __init__(self):
        super().__init__()

    def step(self, update_area: np.ndarray, targets):
        is_upper = self.is_upper_best.clone()
        if config.update_area == "superpixel" and config.channel_wise:
            for idx in range(self.batch):
                c, label = targets[idx][0]
                is_upper[idx, c, update_area[idx] == label] = ~is_upper[
                    idx, c, update_area[idx] == label
                ]
            self.x_adv = torch.where(is_upper, self.upper, self.lower)
        elif config.update_area == "superpixel":
            self.x_adv = self.x_adv.permute(0, 2, 3, 1)
            labels = [t[0] for t in targets]
            labels = np.repeat(labels, update_area.shape[1])
            labels = np.repeat(labels, update_area.shape[2])
            breakpoint()
            is_upper[update_area == label] = ~is_upper[update_area == label]
            breakpoint()
            self.x_adv = torch.where(is_upper, self.upper, self.lower)
            self.x_adv = self.x_adv.permute(0, 3, 1, 2)
        elif config.update_area == "random_square" and config.channel_wise:
            self.x_adv = self.x_adv.permute(1, 0, 2, 3)
            c = targets[0]
            is_upper[c, update_area] = ~is_upper[c, update_area]
            self.x_adv = torch.where(is_upper, self.upper, self.lower)
            self.x_adv = self.x_adv.permute(1, 0, 2, 3)
        elif config.update_area == "random_square":
            self.x_adv = self.x_adv.permute(0, 2, 3, 1)
            is_upper[update_area] = ~is_upper[update_area]
            self.x_adv = torch.where(is_upper, self.upper, self.lower)
            self.x_adv = self.x_adv.permute(0, 3, 1, 2)
        elif config.update_area == "split_square" and config.channel_wise:
            self.x_adv = self.x_adv.permute(1, 0, 2, 3)
            c, label = targets[0]
            is_upper[c, update_area == label] = ~is_upper[c, update_area == label]
            self.x_adv = torch.where(is_upper, self.upper, self.lower)
            self.x_adv = self.x_adv.permute(1, 0, 2, 3)
        elif config.update_area == "split_square":
            self.x_adv = self.x_adv.permute(0, 2, 3, 1)
            label = targets[0]
            is_upper[update_area == label] = ~is_upper[update_area == label]
            self.x_adv = torch.where(is_upper, self.upper, self.lower)
            self.x_adv = self.x_adv.permute(0, 3, 1, 2)
        pred = self.model(self.x_adv).softmax(dim=1)
        self.loss = self.criterion(pred, self.y)
        self.forward += 1
        update = self.loss >= self.best_loss
        self.is_upper_best[update] = self.is_upper[update]
        self.x_best[update] = self.x_adv[update]
        self.best_loss[update] = self.loss[update]
        return self.x_best, self.forward
