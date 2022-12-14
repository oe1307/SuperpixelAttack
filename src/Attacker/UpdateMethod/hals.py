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
            insert_end = np.array([t.shape[0] for t in targets]) == 0
            if insert_end.all() or self.forward.min() >= config.step:
                is_upper, x_adv, loss = self.update(update_area)
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
            deletion_end = np.array([t.shape[0] for t in targets]) == 0
            if deletion_end.all() or self.forward.min() >= config.step:
                is_upper, x_adv, loss = self.update(update_area)
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

    def insert_deletion(self, update_area, targets):
        is_upper = self.is_upper_best.clone()

        if config.update_area == "superpixel" and config.channel_wise:
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                c, label = targets[idx][0]
                is_upper[idx, c, update_area[idx] == label] = ~is_upper[
                    idx, c, update_area[idx] == label
                ]
                self.forward[idx] += 1
                targets[idx] = targets[idx][1:]
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                c, label = targets[idx][0]
                delta = (self.best_loss[idx] - loss[idx]).item()
                heapq.heappush(self.max_heap[idx], (delta, (c, label)))
        elif config.udpate_area == "superpixel":
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                label = targets[idx][0]
                is_upper[idx, :, update_area[idx] == label] = ~is_upper[
                    idx, :, update_area[idx] == label
                ]
                self.forward[idx] += 1
                targets[idx] = targets[idx][1:]
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                label = targets[idx][0]
                delta = (self.best_loss[idx] - loss[idx]).item()
                heapq.heappush(self.max_heap[idx], (delta, label))
        elif config.update_area == "split_square" and config.channel_wise:
            is_upper = is_upper.permute(1, 2, 3, 0)
            c, label = targets[0]
            is_upper[c, update_area == label] = ~is_upper[c, update_area == label]
            is_upper = is_upper.permute(3, 0, 1, 2)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            self.forward += 1
            targets = targets[1:]
            for idx in range(self.batch):
                delta = (self.best_loss[idx] - loss[idx]).item()
                heapq.heappush(self.max_heap[idx], (delta, (c, label)))
        elif config.update_area == "split_square":
            is_upper = is_upper.permute(0, 2, 3, 1)
            label = targets[0]
            is_upper[update_area == label] = ~is_upper[update_area == label]
            is_upper = is_upper.permute(0, 3, 1, 2)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            self.forward += 1
            targets = targets[1:]
            for idx in range(self.batch):
                delta = (self.best_loss[idx] - loss[idx]).item()
                heapq.heappush(self.max_heap[idx], (delta, label))
        elif config.update_area == "saliency_map" and config.channel_wise:
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                c, label = targets[idx][0]
                is_upper[idx, c, update_area[idx] == label] = ~is_upper[
                    idx, c, update_area[idx] == label
                ]
                self.forward[idx] += 1
                targets[idx] = targets[idx][1:]
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                c, label = targets[idx][0]
                delta = (self.best_loss[idx] - loss[idx]).item()
                heapq.heappush(self.max_heap[idx], (delta, (c, label)))
        elif config.update_area == "saliency_map":
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                label = targets[idx][0]
                is_upper[idx, :, update_area[idx] == label] = ~is_upper[
                    idx, :, update_area[idx] == label
                ]
                self.forward[idx] += 1
                targets[idx] = targets[idx][1:]
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            for idx in range(self.batch):
                if targets[idx].shape[0] == 0:
                    continue
                label = targets[idx][0]
                delta = (self.best_loss[idx] - loss[idx]).item()
                heapq.heappush(self.max_heap[idx], (delta, label))
        return targets

    def update(self, update_area):
        is_upper = self.is_upper_best.clone()
        d = self.mode == "insert"
        for idx, _max_heap in enumerate(self.max_heap):
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    if config.channel_wise:
                        c, label = element_hat
                        is_upper[idx, c, update_area[idx] == label] = d
                    else:
                        label = element_hat
                        is_upper[idx][idx, :, update_area[idx] == label] = d
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
        x_adv = torch.where(is_upper, self.upper, self.lower)
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        return is_upper, x_adv, loss
