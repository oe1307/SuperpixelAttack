import heapq

import numpy as np
import torch

from utils import config_parser, pbar

from .base_method import BaseMethod

config = config_parser()


class HALS(BaseMethod):
    def __init__(self):
        super().__init__()
        self.local_search = True
        if config.update_area == "random_square":
            raise NotImplementedError("HALS does not support random_square")

    def step(self, update_area: np.ndarray, targets):
        if self.local_search:
            is_upper = self.is_upper_best.clone()
            loss = self.best_loss.clone()
            # insert
            max_heap = [[] for _ in range(self.batch)]
            if config.channel_wise:
                for c, label in targets:
                    _is_upper = is_upper.clone()
                    n_upper = is_upper.permute(1, 2, 3, 0)[c, update_area == label]
                    n_upper = n_upper.sum(dim=0)
                    to_search = n_upper < (update_area == label).sum() // 2
                    to_search = to_search.cpu().numpy()[:, None, None]
                    to_search = to_search.repeat(update_area.shape[0], axis=1)
                    to_search = to_search.repeat(update_area.shape[0], axis=2)
                    to_search = np.logical_and(to_search, update_area == label)
                    _is_upper.permute(1, 0, 2, 3)[c, to_search] = True
                    x_adv = torch.where(_is_upper, self.upper, self.lower)
                    pred = self.model(x_adv).softmax(dim=1)
                    _loss = self.criterion(pred, self.y)
                    for idx in range(self.batch):
                        if self.forward[idx] < config.step:
                            delta = (loss[idx] - _loss[idx]).item()
                            heapq.heappush(max_heap[idx], (delta, (c, label)))
                            self.forward[idx] += 1
                    pbar.debug(self.forward.min(), config.step, "forward")
                    if self.forward.min() >= config.step:
                        break
                for idx, _max_heap in enumerate(max_heap):
                    while len(_max_heap) > 1:
                        delta_hat, (c, label) = heapq.heappop(_max_heap)
                        delta_tilde = _max_heap[0][0]
                        if delta_hat <= delta_tilde and delta_hat < 0:
                            is_upper[idx, c, update_area == label] = True
                        elif delta_hat <= delta_tilde and delta_hat >= 0:
                            break
                        else:
                            heapq.heappush(_max_heap, (delta_hat, (c, label)))
            else:
                for label in targets:
                    _is_upper = is_upper.clone()
                    n_upper = is_upper.permute(2, 3, 0, 1)[update_area == label]
                    n_upper = n_upper.sum(dim=(0, 2))
                    to_search = n_upper < (update_area == label).sum() // 2
                    to_search = to_search.cpu().numpy()[:, None, None]
                    to_search = to_search.repeat(update_area.shape[0], axis=1)
                    to_search = to_search.repeat(update_area.shape[0], axis=2)
                    to_search = np.logical_and(to_search, update_area == label)
                    _is_upper.permute(1, 0, 2, 3)[:, to_search] = True
                    x_adv = torch.where(_is_upper, self.upper, self.lower)
                    pred = self.model(x_adv).softmax(dim=1)
                    _loss = self.criterion(pred, self.y)
                    for idx in range(self.batch):
                        if self.forward[idx] < config.step:
                            delta = (loss[idx] - _loss[idx]).item()
                            heapq.heappush(max_heap[idx], (delta, label))
                            self.forward[idx] += 1
                    pbar.debug(self.forward.min(), config.step, "forward")
                    if self.forward.min() >= config.step:
                        break
                for idx, _max_heap in enumerate(max_heap):
                    while len(_max_heap) > 1:
                        delta_hat, label = heapq.heappop(_max_heap)
                        delta_tilde = _max_heap[0][0]
                        if delta_hat <= delta_tilde and delta_hat < 0:
                            is_upper[idx, :, update_area == label] = True
                        elif delta_hat <= delta_tilde and delta_hat >= 0:
                            break
                        else:
                            heapq.heappush(_max_heap, (delta_hat, label))
            x_adv = torch.where(is_upper, self.upper, self.lower)
            self.forward += 1
            update = loss > self.best_loss
            self.is_upper_best[update] = is_upper[update]
            self.best_loss[update] = loss[update]

            # deletion
            max_heap = [[] for _ in range(self.batch)]
            if config.channel_wise:
                for c, label in targets:
                    _is_upper = is_upper.clone()
                    n_upper = is_upper.permute(1, 2, 3, 0)[c, update_area == label]
                    n_upper = n_upper.sum(dim=0)
                    to_search = n_upper >= (update_area == label).sum() // 2
                    to_search = to_search.cpu().numpy()[:, None, None]
                    to_search = to_search.repeat(update_area.shape[0], axis=1)
                    to_search = to_search.repeat(update_area.shape[0], axis=2)
                    to_search = np.logical_and(to_search, update_area == label)
                    _is_upper.permute(1, 0, 2, 3)[c, to_search] = False
                    x_adv = torch.where(_is_upper, self.upper, self.lower)
                    pred = self.model(x_adv).softmax(dim=1)
                    _loss = self.criterion(pred, self.y)
                    for idx in range(self.batch):
                        if self.forward[idx] < config.step:
                            delta = (loss[idx] - _loss[idx]).item()
                            heapq.heappush(max_heap[idx], (delta, (c, label)))
                            self.forward[idx] += 1
                    pbar.debug(self.forward.min(), config.step, "forward")
                    if self.forward.min() >= config.step:
                        break
                for idx, _max_heap in enumerate(max_heap):
                    while len(_max_heap) > 1:
                        delta_hat, (c, label) = heapq.heappop(_max_heap)
                        delta_tilde = _max_heap[0][0]
                        if delta_hat <= delta_tilde and delta_hat < 0:
                            is_upper[idx, c, update_area == label] = False
                        elif delta_hat <= delta_tilde and delta_hat >= 0:
                            break
                        else:
                            heapq.heappush(_max_heap, (delta_hat, (c, label)))
            else:
                for label in targets:
                    _is_upper = is_upper.clone()
                    n_upper = is_upper.permute(2, 3, 0, 1)[update_area == label]
                    n_upper = n_upper.sum(dim=(0, 2))
                    to_search = n_upper >= (update_area == label).sum() // 2
                    to_search = to_search.cpu().numpy()[:, None, None]
                    to_search = to_search.repeat(update_area.shape[0], axis=1)
                    to_search = to_search.repeat(update_area.shape[0], axis=2)
                    to_search = np.logical_and(to_search, update_area == label)
                    _is_upper.permute(1, 0, 2, 3)[:, to_search] = False
                    x_adv = torch.where(_is_upper, self.upper, self.lower)
                    pred = self.model(x_adv).softmax(dim=1)
                    _loss = self.criterion(pred, self.y)
                    for idx in range(self.batch):
                        if self.forward[idx] < config.step:
                            delta = (loss[idx] - _loss[idx]).item()
                            heapq.heappush(max_heap[idx], (delta, label))
                            self.forward[idx] += 1
                    pbar.debug(self.forward.min(), config.step, "forward")
                    if self.forward.min() >= config.step:
                        break
                for idx, _max_heap in enumerate(max_heap):
                    while len(_max_heap) > 1:
                        delta_hat, label = heapq.heappop(_max_heap)
                        delta_tilde = _max_heap[0][0]
                        if delta_hat <= delta_tilde and delta_hat < 0:
                            is_upper[idx, :, update_area == label] = False
                        elif delta_hat <= delta_tilde and delta_hat >= 0:
                            break
                        else:
                            heapq.heappush(_max_heap, (delta_hat, label))
            x_adv = torch.where(is_upper, self.upper, self.lower)
            self.forward += 1
            update = loss > self.best_loss
            self.is_upper_best[update] = is_upper[update]
            self.best_loss[update] = loss[update]

            # check inverse
            x_adv_inverse = torch.where(~is_upper, self.upper, self.lower)
            pred = self.model(x_adv_inverse).softmax(dim=1)
            loss_inverse = self.criterion(pred, self.y)
            update = self.forward < config.step
            self.forward += update
            update = np.logical_and(
                update, (loss_inverse > self.best_loss).cpu().numpy()
            )
            self.is_upper_best[update] = ~is_upper[update]
            self.x_best = x_adv_inverse.clone()
            self.best_loss[update] = loss_inverse[update]
            self.local_search = False
        elif targets.shape[0] == 1:
            self.local_search = True
        return self.x_best, self.forward
