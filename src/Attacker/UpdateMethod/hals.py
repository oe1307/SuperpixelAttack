import heapq

import numpy as np
import torch
from torch import Tensor

from utils import config_parser, pbar

from .base_method import BaseMethod

config = config_parser()


class HALS(BaseMethod):
    def __init__(self):
        super().__init__()

    def step(self, update_area: np.ndarray, targets):
        self.is_upper = self.is_upper_best.clone()
        if config.update_area == "superpixel" and config.channel_wise:
            pass
            self.forward += None
        elif config.update_area == "superpixel":
            pass
            self.forward += None
        elif config.update_area == "random_square" and config.channel_wise:
            pass
            self.forward += None
        elif config.update_area == "random_square":
            pass
            self.forward += None
        elif config.update_area == "split_square" and config.channel_wise:
            pass
            self.forward += None
        elif config.update_area == "split_square":
            pass
            self.forward += None
        pred = self.model(self.x_adv).softmax(dim=1)
        self.loss = self.criterion(pred, self.y)
        update = self.loss >= self.best_loss
        self.is_upper_best[update] = self.is_upper[update]
        self.x_best[update] = self.x_adv[update]
        self.best_loss[update] = self.loss[update]
        return self.x_best, self.forward

    def insert(self, is_upper_all, all_loss):
        n_images = is_upper_all.shape[0]
        max_heap = [[] for _ in range(n_images)]
        all_elements = (~is_upper_all).nonzero()

        # search in elementary
        num_batch = np.ceil(all_elements.shape[0] / self.model.batch_size)
        for i in range(num_batch):
            pbar.debug(i + 1, num_batch, "insert")
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[start:end]
            searched = []
            is_upper = is_upper_all[elements[:, 0]].clone()
            for i, (idx, c, h, w) in enumerate(elements):
                if self.forward[idx] >= config.step:
                    continue
                assert is_upper[i, c, h, w].item() is False
                is_upper[i, c, h, w] = True
                self.forward[idx] += 1
                searched.append((i, idx, c, h, w))
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=2)
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=3)
            upper = self.upper[elements[:, 0]]
            lower = self.lower[elements[:, 0]]
            x_adv = torch.where(is_upper, upper, lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y[elements[:, 0]])
            for i, idx, c, h, w in searched:
                delta = (all_loss[idx] - loss[i]).item()
                heapq.heappush(max_heap[idx], (delta, (c, h, w)))

        # update
        for idx, _max_heap in enumerate(max_heap):
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    assert is_upper_all[idx][element_hat].item() is False
                    is_upper_all[idx][element_hat] = True
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
        all_loss = self.calculate_loss(is_upper_all)
        self.forward += 1
        return is_upper_all, all_loss

    def deletion(self, is_upper_all, all_loss):
        n_images = is_upper_all.shape[0]
        max_heap = [[] for _ in range(n_images)]
        all_elements = is_upper_all.nonzero()

        # search in elementary
        num_batch = np.ceil(all_elements.shape[0] / self.model.batch_size)
        for i in range(num_batch):
            pbar.debug(i + 1, num_batch, "deletion")
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[start:end]
            searched = []
            is_upper = is_upper_all[elements[:, 0]].clone()
            for i, (idx, c, h, w) in enumerate(elements):
                if self.forward[idx] >= config.step:
                    continue
                assert is_upper[i, c, h, w].item() is True
                is_upper[i, c, h, w] = False
                self.forward[idx] += 1
                searched.append((i, idx, c, h, w))
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=2)
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=3)
            upper = self.upper[elements[:, 0]]
            lower = self.lower[elements[:, 0]]
            x_adv = torch.where(is_upper, upper, lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y[elements[:, 0]])
            for i, idx, c, h, w in searched:
                delta = (all_loss[idx] - loss[i]).item()
                heapq.heappush(max_heap[idx], (delta, (c, h, w)))

        # update
        for idx, _max_heap in enumerate(max_heap):
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    assert is_upper_all[idx][element_hat].item() is True
                    is_upper_all[idx][element_hat] = False
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
        loss = self.calculate_loss(is_upper_all)
        self.forward += 1
        return is_upper_all, loss

    def calculate_loss(self, is_upper_all: Tensor) -> Tensor:
        n_images = is_upper_all.shape[0]
        loss = torch.zeros(n_images, device=config.device)
        num_batch = np.ceil(n_images / self.model.batch_size)
        for i in range(num_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            upper = self.upper[start:end]
            lower = self.lower[start:end]
            is_upper = is_upper_all[start:end]
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=2)
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=3)
            x_adv = torch.where(is_upper, upper, lower)
            y = self.y[start:end]
            pred = self.model(x_adv).softmax(dim=1)
            loss[start:end] = self.criterion(pred, y)
        return loss
