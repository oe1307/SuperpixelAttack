import heapq

import numpy as np
import torch

from utils import config_parser, pbar

from .base_method import BaseMethod

config = config_parser()


class HALS(BaseMethod):
    def __init__(self):
        super().__init__()
        self.mode = "init_insert"
        if config.update_area == "random_square":
            raise NotImplementedError("HALS does not support random_square")

    def step(self, update_area: np.ndarray, targets):
        if self.mode == "init_insert":
            self.backup = targets.copy()
            self.max_heap = [[] for _ in range(self.batch)]
            self.mode = "insert"

        if self.mode == "insert":
            targets = self.insert_deletion(update_area, targets)
            if targets.shape[0] == 0 or self.forward.min() >= config.step:
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.x_best[update] = x_adv[update]
                self.best_loss[update] = loss[update]
                self.mode = "init_deletion"

        if self.mode == "init_deletion":
            targets = self.backup
            self.max_heap = [[] for _ in range(self.batch)]
            self.mode = "deletion"

        if self.mode == "deletion":
            targets = self.insert_deletion(update_area, targets)
            if targets.shape[0] == 0 or self.forward.min() >= config.step:
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.x_best[update] = x_adv[update]
                self.best_loss[update] = loss[update]
                self.mode = "inverse"

        if self.mode == "inverse":
            loss_inverse = self.calculate_loss(~is_upper)
            update1 = self.forward < config.step
            self.forward += update1
            update2 = (loss_inverse > self.best_loss).cpu().numpy()
            update = np.logical_and(update1, update2)
            self.is_upper_best[update] = ~is_upper[update]
            self.best_loss[update] = loss_inverse[update]
            self.mode = "init_insert"

        return self.x_best, self.forward, targets

    def insert_deletion(self, targets):
        is_upper = self.is_upper_best.clone()
        loss = self.best_loss.clone()

        if config.udpate_area == "superpixel" and config.channel_wise:
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                c, label = targets[idx][0]
                is_upper[idx, c, update_area[idx] == label] = ~is_upper[
                    idx, c, update_area[idx] == label
                ]
            self.x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(self.x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                c, label = targets[idx][0]
                delta = (self.best_loss[idx] - loss[i]).item()
                heapq.heappush(
                    self.max_heap[idx],
                    (-loss[idx, c].cpu().numpy(), (c, label)),
                )
        return is_upper, loss, targets
