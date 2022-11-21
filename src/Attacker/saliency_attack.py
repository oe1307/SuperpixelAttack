import heapq
import math

import torch
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SaliencyAttack(Attacker):
    def __init__(self):
        super().__init__()
        if config.dataset != "imagenet":
            raise ValueError("Saliency Attack is only for ImageNet")
        self.criterion = get_criterion()
        config.n_forward = config.steps

    def _attack(self, x_all: Tensor, y_all: Tensor):
        x_adv_all = []
        n_images = x_all.shape[0]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            self.y = y_all[start:end]
            self.upper = (x + config.epsilon).clamp(0, 1)
            self.lower = (x - config.epsilon).clamp(0, 1)
            self.split = config.initial_split
            self.batch, c, h, w = x.shape
            is_upper = torch.zeros(
                (self.batch, c, h // self.split, w // self.split),
                dtype=bool,
                device=config.device,
            )
            is_upper_best = is_upper.clone()
            x_adv = self.lower.clone()
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            best_loss = loss.clone()
            self.forward = torch.ones(self.batch, device=config.device)

            while True:
                is_upper, loss, is_upper_best, best_loss = self.local_search(
                    is_upper, loss, is_upper_best, best_loss
                )
                if self.forward.min().item() >= config.steps:
                    break
                if self.split > 1:
                    is_upper = is_upper.repeat([1, 1, 2, 2])
                    is_upper_best = is_upper_best.repeat([1, 1, 2, 2])
                    if self.split % 2 == 1:
                        logger.critical(f"self.split is not even: {self.split}")
                    self.split //= 2

            is_upper = is_upper.repeat([1, 1, self.split, self.split])
            x_adv = torch.where(is_upper, self.upper, self.lower)
            x_adv_all.append(x_adv)
        x_adv_all = torch.cat(x_adv_all)
        return x_adv_all

    def local_search(self, is_upper, loss, is_upper_best, best_loss):
        for _ in range(config.insert_deletion):
            is_upper, loss = self.insert(is_upper, loss)
            is_upper_best = torch.where(
                (loss > best_loss).view(-1, 1, 1, 1), is_upper, is_upper_best
            )
            best_loss = torch.max(loss, best_loss)
            if self.forward.min().item() >= config.steps:
                break
            is_upper, loss = self.deletion(is_upper, loss)
            is_upper_best = torch.where(
                (loss > best_loss).view(-1, 1, 1, 1), is_upper, is_upper_best
            )
            best_loss = torch.max(loss, best_loss)
            if self.forward.min().item() >= config.steps:
                break
        _is_upper = is_upper.repeat([1, 1, self.split, self.split])
        x_adv_inverse = torch.where(~_is_upper, self.upper, self.lower)
        pred = self.model(x_adv_inverse).softmax(dim=1)
        loss_inverse = self.criterion(pred, self.y)

        update = torch.logical_and(loss_inverse > loss, self.forward < config.steps)
        is_upper = torch.where(update.view(-1, 1, 1, 1), ~is_upper, is_upper)
        loss = torch.max(loss_inverse, loss)
        update = torch.logical_and(
            loss_inverse > best_loss, self.forward < config.steps
        )
        is_upper_best = torch.where(update.view(-1, 1, 1, 1), ~is_upper, is_upper_best)
        best_loss = torch.max(loss_inverse, best_loss)
        return is_upper, loss, is_upper_best, best_loss

    def insert(self, is_upper, base_loss):
        max_heap = [[] for _ in range(self.batch)]
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
            _is_upper = _is_upper.repeat([1, 1, self.split, self.split])
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
        _is_upper = is_upper.repeat([1, 1, self.split, self.split])
        x_adv = torch.where(_is_upper, self.upper, self.lower).clone()
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        return is_upper, loss

    def deletion(self, is_upper, base_loss):
        max_heap = [[] for _ in range(self.batch)]
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
            _is_upper = _is_upper.repeat([1, 1, self.split, self.split])
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
        _is_upper = is_upper.repeat([1, 1, self.split, self.split])
        x_adv = torch.where(_is_upper, self.upper, self.lower).clone()
        pred = self.model(x_adv).softmax(dim=1)
        loss = self.criterion(pred, self.y)
        return is_upper, loss
