import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class UniformDistribution:
    def __init__(self, update_area):
        if config.update_area != "superpixel":
            raise ValueError("Update area is only available for superpixel.")
        self.update_area = update_area

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

        is_upper = torch.randint(
            0,
            2,
            (self.batch, self.n_channel, 1, self.width),
            device=config.device,
            dtype=bool,
        ).repeat_interleave(self.height, dim=2)
        x_adv = torch.where(is_upper, upper, lower)
        pred = self.model(x_adv).softmax(1)
        loss = self.criterion(pred, y)
        self.forward = np.ones(self.batch, dtype=int)
        self.is_upper_best = is_upper.clone()
        self.x_best = x_adv.clone()
        self.best_loss = loss.clone()
        return self.forward

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
