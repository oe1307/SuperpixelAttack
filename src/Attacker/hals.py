import heapq
import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class HALS(Attacker):
    def __init__(self):
        super().__init__()
        self.criterion = get_criterion()
        config.n_forward = config.steps

    def _attack(self, x: Tensor, y: Tensor):
        self.y = y
        self.upper = (x + config.epsilon).clamp(0, 1)
        self.lower = (x - config.epsilon).clamp(0, 1)
        self.split = config.initial_split
        n_images, n_chanel, height, width = x.shape

        # initialize
        init_block = (n_images, n_chanel, height // self.split, width // self.split)
        is_upper = torch.zeros(init_block, dtype=torch.bool, device=config.device)
        is_upper_best = is_upper.clone()
        loss = self.cal_loss(is_upper)
        best_loss = loss.clone()
        self.forward = np.ones(n_images)

        # main loop
        while True:
            is_upper, loss, is_upper_best, best_loss = self.local_search(
                is_upper, loss, is_upper_best, best_loss
            )
            if self.forward.min() >= config.steps:
                break
            elif self.split > 1:
                is_upper = torch.repeat_interleave(is_upper, 2, dim=2)
                is_upper = torch.repeat_interleave(is_upper, 2, dim=3)
                is_upper_best = torch.repeat_interleave(is_upper_best, 2, dim=2)
                is_upper_best = torch.repeat_interleave(is_upper_best, 2, dim=3)
                if self.split % 2 == 1:
                    logger.critical(f"self.split is not even: {self.split}")
                self.split //= 2

        is_upper_best = torch.repeat_interleave(is_upper_best, self.split, dim=2)
        is_upper_best = torch.repeat_interleave(is_upper_best, self.split, dim=3)
        x_best = torch.where(is_upper_best, self.upper, self.lower)
        return x_best

    def local_search(self, is_upper, loss, is_upper_best, best_loss):
        for _ in range(config.insert_deletion):
            is_upper, loss = self.insert(is_upper, loss)
            update = loss > best_loss
            is_upper_best[update] = is_upper[update]
            best_loss[update] = loss[update]
            if self.forward.min() >= config.steps:
                break

            is_upper, loss = self.deletion(is_upper, loss)
            update = loss > best_loss
            is_upper_best[update] = is_upper[update]
            best_loss[update] = loss[update]
            if self.forward.min() >= config.steps:
                break

        loss_inverse = self.cal_loss(~is_upper)
        self.forward += 1

        update = (loss_inverse > loss).cpu().numpy()
        update = np.logical_and(update, self.forward < config.steps)
        is_upper[update] = ~is_upper[update]
        loss[update] = loss_inverse[update]

        update = (loss > best_loss).cpu().numpy()
        update = np.logical_and(update, self.forward < config.steps)
        is_upper_best[update] = is_upper[update]
        best_loss[update] = loss[update]
        return is_upper, loss, is_upper_best, best_loss

    def insert(self, is_upper, base_loss):
        n_images = is_upper.shape[0]
        max_heap = [[] for _ in range(n_images)]
        all_elements = (~is_upper).nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
        for i in range(num_batch):
            pbar.debug(i + 1, num_batch, "insert")
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[start:end]
            searched = []
            _is_upper = is_upper[elements[:, 0]].clone()
            for i, (idx, c, h, w) in enumerate(elements):
                if self.forward[idx] > config.steps:
                    continue
                assert _is_upper[i, c, h, w].item() is False
                _is_upper[i, c, h, w] = True
                self.forward[idx] += 1
                searched.append((i, idx, c, h, w))
            _is_upper = torch.repeat_interleave(_is_upper, self.split, dim=2)
            _is_upper = torch.repeat_interleave(_is_upper, self.split, dim=3)
            upper = self.upper[elements[:, 0]]
            lower = self.lower[elements[:, 0]]
            x_adv = torch.where(_is_upper, upper, lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y[elements[:, 0]])
            for i, idx, c, h, w in searched:
                delta = (base_loss[idx] - loss[i]).item()
                heapq.heappush(max_heap[idx], (delta, (c, h, w)))

        # update
        _is_upper = []
        for idx, _max_heap in enumerate(max_heap):
            idx_is_upper = is_upper[idx]
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    assert idx_is_upper[element_hat].item() is False
                    idx_is_upper[element_hat] = True
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
            _is_upper.append(idx_is_upper)
        is_upper = torch.stack(_is_upper)
        loss = self.cal_loss(is_upper)
        self.forward += 1
        return is_upper, loss

    def deletion(self, is_upper, base_loss):
        n_images = is_upper.shape[0]
        max_heap = [[] for _ in range(n_images)]
        all_elements = is_upper.nonzero()

        # search in elementary
        num_batch = math.ceil(all_elements.shape[0] / self.model.batch_size)
        for i in range(num_batch):
            pbar.debug(i + 1, num_batch, "deletion")
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, all_elements.shape[0])
            elements = all_elements[start:end]
            searched = []
            _is_upper = is_upper[elements[:, 0]].clone()
            for i, (idx, c, h, w) in enumerate(elements):
                if self.forward[idx] > config.steps:
                    continue
                assert _is_upper[i, c, h, w].item() is True
                _is_upper[i, c, h, w] = False
                self.forward[idx] += 1
                searched.append((i, idx, c, h, w))
            _is_upper = torch.repeat_interleave(_is_upper, self.split, dim=2)
            _is_upper = torch.repeat_interleave(_is_upper, self.split, dim=3)
            upper = self.upper[elements[:, 0]]
            lower = self.lower[elements[:, 0]]
            x_adv = torch.where(_is_upper, upper, lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y[elements[:, 0]])
            for i, idx, c, h, w in searched:
                delta = (base_loss[idx] - loss[i]).item()
                heapq.heappush(max_heap[idx], (delta, (c, h, w)))

        # update
        _is_upper = []
        for idx, _max_heap in enumerate(max_heap):
            idx_is_upper = is_upper[idx]
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    assert idx_is_upper[element_hat].item() is True
                    idx_is_upper[element_hat] = False
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
            _is_upper.append(idx_is_upper)
        is_upper = torch.stack(_is_upper)
        loss = self.cal_loss(is_upper)
        self.forward += 1
        return is_upper, loss

    def cal_loss(self, is_upper_all: Tensor) -> Tensor:
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
