import heapq

import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class HALS:
    def __init__(self, update_area):
        if config.update_area != "superpixel":
            raise ValueError("Update area is only available for superpixel.")
        self.update_area = update_area
        self.mode = "insert"

    def initialize(self, x: Tensor, y: Tensor, lower: Tensor, upper: Tensor):
        forward = super().initialize(x, y, lower, upper)
        self.max_heap = [[] for _ in range(self.batch)]
        return forward

    def step(self):
        if self.mode == "insert":
            self.insert_deletion()
            search_end = np.array([t.shape[0] == 0 for t in self.targets]).all()
            if search_end or self.forward.min() >= config.step:
                self.update()
                self.max_heap = [[] for _ in range(self.batch)]
                for idx in range(self.batch):
                    self.area = self.update_area.update(idx, self.level[idx])
                self.mode = "deletion"

        if self.mode == "deletion":
            self.insert_deletion()
            search_end = np.array([t.shape[0] == 0 for t in self.targets]).all()
            if search_end or self.forward.min() >= config.step:
                self.update()
                self.max_heap = [[] for _ in range(self.batch)]
                self.level += 1
                for idx in range(self.batch):
                    self.area = self.update_area.update(idx, self.level[idx])
                    if config.channel_wise:
                        labels = np.unique(self.area[idx])
                        labels = labels[labels != 0]
                        channel = np.tile(np.arange(self.n_channel), len(labels))
                        labels = np.repeat(labels, self.n_channel)
                        channel_labels = np.stack([channel, labels], axis=1)
                        self.targets[idx] = np.random.permutation(channel_labels)
                    else:
                        labels = np.unique(self.area[idx])
                        labels = labels[labels != 0]
                        self.targets[idx] = np.random.permutation(labels)
                self.mode = "inverse"

        if self.mode == "inverse":
            x_inverse = torch.where(~self.is_upper_best, self.upper, self.lower)
            pred = self.model(x_inverse).softmax(dim=1)
            loss_inverse = self.criterion(pred, self.y)
            update1 = self.forward < config.step
            self.forward += update1
            update2 = (loss_inverse > self.best_loss).cpu().numpy()
            update = np.logical_and(update1, update2)
            self.is_upper_best[update] = ~self.is_upper_best[update]
            self.best_loss[update] = loss_inverse[update]
            self.x_best[update] = x_inverse[update]
            self.mode = "insert"

        return self.x_best, self.forward

    def insert_deletion(self):
        is_upper = self.is_upper_best.clone()

        for idx in range(self.batch):
            if self.targets[idx].shape[0] == 0:
                continue
            if config.channel_wise:
                c, label = self.targets[idx][0]
                is_upper[idx, c, self.area[idx] == label] = ~is_upper[
                    idx, c, self.area[idx] == label
                ]  # FIXME: this is wrong
            else:
                label = self.targets[idx][0]
                is_upper[idx, :, self.area[idx] == label] = ~is_upper[
                    idx, :, self.area[idx] == label
                ]
            self.forward[idx] += 1
        x_adv = torch.where(is_upper, self.upper, self.lower)
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        for idx in range(self.batch):
            if self.targets[idx].shape[0] == 0:
                continue
            if config.channel_wise:
                c, label = self.targets[idx][0]
                delta = (self.best_loss[idx] - loss[idx]).item()
                heapq.heappush(self.max_heap[idx], (delta, (c, label)))
            else:
                label = self.targets[idx][0]
                delta = (self.best_loss[idx] - loss[idx]).item()
                heapq.heappush(self.max_heap[idx], (delta, label))
            self.targets[idx] = self.targets[idx][1:]

    def update(self):
        is_upper = self.is_upper_best.clone()
        d = self.mode == "insert"
        for idx, _max_heap in enumerate(self.max_heap):
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    if config.channel_wise:
                        c, label = element_hat
                        is_upper[idx, c, self.area[idx] == label] = d
                    else:
                        label = element_hat
                        is_upper.permute(0, 2, 3, 1)[idx, self.area[idx] == label] = d
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
        x_adv = torch.where(is_upper, self.upper, self.lower)
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        update = loss > self.best_loss
        self.x_best[update] = x_adv[update]
        self.best_loss[update] = loss[update]
