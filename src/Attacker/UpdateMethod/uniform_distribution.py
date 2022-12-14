import numpy as np
import torch

from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class UniformDistribution(BaseMethod):
    def __init__(self):
        super().__init__()

    def step(self, update_area: np.ndarray, targets):
        x_adv = self.x_best.clone()
        if config.update_area == "superpixel" and config.channel_wise:
            for idx in range(self.batch):
                c, label = targets[idx][0]
                rand = torch.rand_like(x_adv[idx, c, update_area[idx] == label])
                rand = (2 * rand - 1) * config.epsilon
                x_adv[idx, c, update_area[idx] == label] += rand
                targets[idx] = targets[idx][1:]
            x_adv = x_adv.clamp(self.lower, self.upper)
        elif config.update_area == "superpixel":
            x_adv = x_adv.permute(0, 2, 3, 1)
            for idx in range(self.batch):
                label = targets[idx][0]
                rand = torch.rand_like(x_adv[idx, update_area[idx] == label])
                rand = (2 * rand - 1) * config.epsilon
                x_adv[idx, update_area[idx] == label] += rand
                targets[idx] = targets[idx][1:]
            x_adv = x_adv.permute(0, 3, 1, 2)
            x_adv = x_adv.clamp(self.lower, self.upper)
        elif config.update_area == "split_square" and config.channel_wise:
            x_adv = x_adv.permute(1, 2, 3, 0)
            c, label = targets[0]
            rand = torch.rand_like(x_adv[c, update_area == label])
            rand = (2 * rand - 1) * config.epsilon
            x_adv[c, update_area == label] += rand
            x_adv = x_adv.permute(3, 0, 1, 2)
            x_adv = x_adv.clamp(self.lower, self.upper)
            targets = targets[1:]
        elif config.update_area == "split_square":
            x_adv = x_adv.permute(2, 3, 0, 1)
            label = targets[0]
            rand = torch.rand_like(x_adv[update_area == label])
            rand = (2 * rand - 1) * config.epsilon
            x_adv[update_area == label] += rand
            x_adv = x_adv.permute(2, 3, 0, 1)
            x_adv = x_adv.clamp(self.lower, self.upper)
            targets = targets[1:]
        elif config.update_area == "saliency_map" and config.channel_wise:
            for idx in range(self.batch):
                c, label = targets[idx][0]
                rand = torch.rand_like(x_adv[idx, c, update_area[idx] == label])
                rand = (2 * rand - 1) * config.epsilon
                x_adv[idx, c, update_area[idx] == label] += rand
                targets[idx] = targets[idx][1:]
            x_adv = x_adv.clamp(self.lower, self.upper)
        elif config.update_area == "saliency_map":
            x_adv = x_adv.permute(0, 2, 3, 1)
            for idx in range(self.batch):
                label = targets[idx][0]
                rand = torch.rand_like(x_adv[idx, update_area[idx] == label])
                rand = (2 * rand - 1) * config.epsilon
                x_adv[idx, update_area[idx] == label] += rand
                targets[idx] = targets[idx][1:]
            x_adv = x_adv.permute(0, 3, 1, 2)
            x_adv = x_adv.clamp(self.lower, self.upper)
        elif config.update_area == "random_square" and config.channel_wise:
            x_adv = x_adv.permute(1, 0, 2, 3)
            c = targets[0]
            rand = torch.rand_like(x_adv[c, update_area])
            rand = (2 * rand - 1) * config.epsilon
            x_adv[c, update_area] += rand
            x_adv = x_adv.permute(1, 0, 2, 3)
            x_adv = x_adv.clamp(self.lower, self.upper)
            targets = targets[1:]
        elif config.update_area == "random_square":
            x_adv = x_adv.permute(0, 2, 3, 1)
            rand = 2 * torch.rand_like(x_adv[update_area]) - 1
            x_adv[update_area] += rand * config.epsilon
            x_adv = x_adv.permute(0, 3, 1, 2)
            x_adv = x_adv.clamp(self.lower, self.upper)
            targets = targets[1:]
        else:
            raise NotImplementedError(config.update_area)
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        self.forward += 1
        update = loss >= self.best_loss
        self.x_best[update] = x_adv[update]
        self.best_loss[update] = loss[update]
        return self.x_best, self.forward, targets
