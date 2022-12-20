import numpy as np
import torch

from utils import config_parser

config = config_parser()


class UniformDistribution:
    def __init__(self, update_area):
        if config.update_area != "superpixel":
            raise ValueError("Update area is only available for superpixel.")
        self.update_area = update_area

    def step(self):
        x_adv = self.x_best.permute(0, 2, 3, 1).clone()
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
