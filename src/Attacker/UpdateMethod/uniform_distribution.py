import numpy as np
import torch

from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class UniformDistribution(BaseMethod):
    def __init__(self, update_area):
        super().__init__(update_area)

    def step(self):
        x_adv = self.x_best.clone()
        if config.channel_wise:
            for idx in range(self.batch):
                c, label = self.targets[idx][0]
                rand = torch.rand_like(x_adv[idx, c, self.area[idx] == label])
                rand = (2 * rand - 1) * config.epsilon
                x_adv[idx, c, self.area[idx] == label] += rand
                self.targets[idx] = self.targets[idx][1:]
                if self.targets[idx].shape[0] == 0:
                    self.level[idx] += 1
                    self.area[idx] = self.update_area.update(idx, self.level[idx])
                    labels = np.unique(self.area[idx])
                    labels = labels[labels != 0]
                    channel = np.tile(np.arange(self.n_channel), len(labels))
                    labels = np.repeat(labels, self.n_channel)
                    channel_labels = np.stack([channel, labels], axis=1)
                    self.targets[idx] = np.random.permutation(channel_labels)
        else:
            x_adv = x_adv.permute(0, 2, 3, 1)
            for idx in range(self.batch):
                label = self.targets[idx][0]
                rand = torch.rand_like(x_adv[idx, self.area[idx] == label])
                rand = (2 * rand - 1) * config.epsilon
                x_adv[idx, self.area[idx] == label] += rand
                self.targets[idx] = self.targets[idx][1:]
                if self.targets[idx].shape[0] == 0:
                    self.level[idx] += 1
                    self.area[idx] = self.update_area.update(idx, self.level[idx])
                    labels = np.unique(self.area[idx])
                    labels = labels[labels != 0]
                    self.targets[idx] = np.random.permutation(labels)
            x_adv = x_adv.permute(0, 3, 1, 2)
        x_adv = x_adv.clamp(self.lower, self.upper)
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        self.forward += 1
        update = loss >= self.best_loss
        self.x_best[update] = x_adv[update]
        self.best_loss[update] = loss[update]
        return self.x_best, self.forward
