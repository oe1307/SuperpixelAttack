import heapq
import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class ParsimoniousAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.criterion = get_criterion()
        config.n_forward = config.step

    def _attack(self, x: Tensor, y: Tensor):
        self.y = y
        self.upper = (x + config.epsilon).clamp(0, 1)
        self.lower = (x - config.epsilon).clamp(0, 1)
        self.split = config.initial_split
        n_images, n_channel, height, width = x.shape

        # initialize
        assert height % self.split == 0 and width % self.split == 0
        init_block = (n_images, n_channel, height // self.split, width // self.split)
        is_upper = torch.zeros(init_block, dtype=torch.bool, device=config.device)
        loss = self.calculate_loss(is_upper)
        self.forward = np.ones(n_images)

        # main loop
        while True:
            is_upper, loss = self.accelerated_local_search(is_upper, loss)
            if self.forward.min() >= config.step:
                break
            elif self.split > 1:
                is_upper = torch.repeat_interleave(is_upper, 2, dim=2)
                is_upper = torch.repeat_interleave(is_upper, 2, dim=3)
                if self.split % 2 == 1:
                    logger.critical(f"self.split is not even: {self.split}")
                self.split //= 2

        is_upper = torch.repeat_interleave(is_upper, self.split, dim=2)
        is_upper = torch.repeat_interleave(is_upper, self.split, dim=3)
        x_best = torch.where(is_upper, self.upper, self.lower)
        return x_best

    def accelerated_local_search(self, is_upper_best, best_loss):
        is_upper = is_upper_best.clone()
        loss = best_loss.clone()
        for _ in range(config.insert_and_deletion):
            is_upper, loss = self.insert(is_upper, loss)
            update = loss > best_loss
            is_upper_best[update] = is_upper[update]
            best_loss[update] = loss[update]
            if self.forward.min() >= config.step:
                break

            is_upper, loss = self.deletion(is_upper, loss)
            update = loss > best_loss
            is_upper_best[update] = is_upper[update]
            best_loss[update] = loss[update]
            if self.forward.min() >= config.step:
                break

        loss_inverse = self.calculate_loss(~is_upper)
        update = self.forward < config.step
        self.forward += update
        update = np.logical_and(update, (loss_inverse > best_loss).cpu().numpy())
        is_upper_best[update] = ~is_upper[update]
        best_loss[update] = loss_inverse[update]
        return is_upper_best, best_loss

    def insert(self, is_upper_all, all_loss):
        n_images = is_upper_all.shape[0]
        max_heap = [[] for _ in range(n_images)]
        all_elements = (~is_upper_all).nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
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
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
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
        num_batch = math.ceil(n_images / self.model.batch_size)
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
