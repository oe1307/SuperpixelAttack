import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class BaseMethod:
    def __init__(self, update_area):
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

        is_upper = torch.zeros_like(x, device=config.device, dtype=bool)
        x_adv = lower.clone()
        pred = self.model(x_adv).softmax(1)
        loss = self.criterion(pred, y)
        self.forward = np.ones(self.batch, dtype=int)

        self.is_upper_best = is_upper.clone()
        self.x_best = x_adv.clone()
        self.best_loss = loss.clone()
        return self.forward
