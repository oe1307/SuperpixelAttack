import heapq
import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, get_criterion
from utils import ProgressBar, config_parser, printc

config = config_parser()


class ParsimoniousAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.criterion = get_criterion()

    def _attack(self, x: Tensor, y: Tensor):
        self.y = y
        self.upper = (x + config.epsilon).clamp(0, 1)
        self.lower = (x - config.epsilon).clamp(0, 1)
        self.split = config.initial_split

        # initial point
        assert self.height % self.split == 0 and self.width % self.split == 0
        init_block = (
            self.n_images,
            self.n_channel,
            self.height // self.split,
            self.width // self.split,
        )
        is_upper = torch.zeros(init_block, dtype=bool)
        self.success_iter = torch.zeros(self.n_images, dtype=int)
        loss = self.calculate_loss(is_upper)
        self.forward = np.ones(self.n_images, dtype=int)

        # search
        while True:
            is_upper, loss = self.accelerated_local_search(is_upper, loss)
            if self.forward.min() >= config.iter:
                break
            elif self.split > 1:
                is_upper = torch.repeat_interleave(is_upper, 2, dim=2)
                is_upper = torch.repeat_interleave(is_upper, 2, dim=3)
                if self.split % 2 == 1:
                    printc("yellow", f"self.split is not even: {self.split}")
                self.split //= 2

        is_upper = torch.repeat_interleave(is_upper, self.split, dim=2)
        is_upper = torch.repeat_interleave(is_upper, self.split, dim=3)
        adv_data = self.lower.clone()
        adv_data[is_upper] = self.upper[is_upper]
        np.save(f"{config.savedir}/success_iter.npy", self.success_iter.numpy())
        return adv_data

    def accelerated_local_search(self, is_upper_best, best_loss):
        is_upper = is_upper_best.clone()
        loss = best_loss.clone()
        for _ in range(config.insert_and_deletion):
            is_upper, loss = self.insert_deletion("insert", is_upper, loss)
            update = loss > best_loss
            is_upper_best[update] = is_upper[update]
            best_loss[update] = loss[update]
            if self.forward.min() >= config.iter:
                return is_upper_best, best_loss

            is_upper, loss = self.insert_deletion("deletion", is_upper, loss)
            update = loss > best_loss
            is_upper_best[update] = is_upper[update]
            best_loss[update] = loss[update]
            if self.forward.min() >= config.iter:
                return is_upper_best, best_loss

        loss_inverse = self.calculate_loss(~is_upper)
        update = self.forward < config.iter
        self.forward += update
        update = np.logical_and(update, (loss_inverse > best_loss).cpu().numpy())
        is_upper_best[update] = ~is_upper[update]
        best_loss[update] = loss_inverse[update]
        return is_upper_best, best_loss

    def insert_deletion(self, mode: str, is_upper_all: Tensor, all_loss: Tensor):
        max_heap = self.elementary_search(mode, is_upper_all, all_loss)
        is_upper_all, all_loss = self.update(mode, max_heap, is_upper_all)
        return is_upper_all, all_loss

    def elementary_search(self, mode: str, is_upper_all, all_loss):
        max_heap = [[] for _ in range(self.n_images)]
        all_elements = self.get_elements(mode, is_upper_all)
        n_batch = math.ceil(len(all_elements) / config.batch_size)
        pbar = ProgressBar(n_batch, mode, color="cyan")
        for i in range(n_batch):
            start = i * config.batch_size
            end = min((i + 1) * config.batch_size, len(all_elements))
            elements = all_elements[start:end]
            searched = []
            is_upper = is_upper_all[elements[:, 0]].clone()
            for j, (idx, c, h, w) in enumerate(elements):
                if mode == "insert":
                    assert is_upper[j, c, h, w].item() is False
                    is_upper[j, c, h, w] = True
                elif mode == "deletion":
                    assert is_upper[j, c, h, w].item() is True
                    is_upper[j, c, h, w] = False
                self.forward[idx] += 1
                searched.append((j, idx, c, h, w))
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=2)
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=3)
            upper = self.upper[elements[:, 0]]
            lower = self.lower[elements[:, 0]]
            x_adv = lower.clone()
            x_adv[is_upper] = upper[is_upper]
            x_adv = x_adv.to(config.device)
            y = self.y[elements[:, 0]].to(config.device)
            prediction = self.model(x_adv).softmax(dim=1)
            attack_fail_index = elements[:, 0][(prediction.argmax(dim=1) == y).cpu()]
            index, count = np.unique(attack_fail_index.cpu(), return_counts=True)
            self.success_iter[index] += count
            loss = self.criterion(prediction, y)
            for i, idx, c, h, w in searched:
                delta = (all_loss[idx] - loss[i]).item()
                heapq.heappush(max_heap[idx], (delta, (c, h, w)))
            pbar.step()
        pbar.end()
        return max_heap

    def get_elements(self, mode, is_upper_all):
        if mode == "insert":
            all_elements = (~is_upper_all).nonzero()
        elif mode == "deletion":
            all_elements = is_upper_all.nonzero()
        removed_elements = []
        for idx in range(self.n_images):
            remain_forward = config.iter - self.forward[idx]
            elements = all_elements[all_elements[:, 0] == idx]
            if len(elements) > remain_forward:
                elements = elements[:remain_forward]
            removed_elements.append(elements)
        removed_elements = torch.cat(removed_elements, dim=0)
        return removed_elements

    def update(self, mode: str, max_heap, is_upper_all):
        for idx, _max_heap in enumerate(max_heap):
            while len(_max_heap) > 1:
                delta_hat, element_hat = heapq.heappop(_max_heap)
                delta_tilde = _max_heap[0][0]
                if delta_hat <= delta_tilde and delta_hat < 0:
                    if mode == "insert":
                        assert is_upper_all[idx][element_hat].item() is False
                        is_upper_all[idx][element_hat] = True
                    elif mode == "deletion":
                        assert is_upper_all[idx][element_hat].item() is True
                        is_upper_all[idx][element_hat] = False
                elif delta_hat <= delta_tilde and delta_hat >= 0:
                    break
                else:
                    heapq.heappush(_max_heap, (delta_hat, element_hat))
        all_loss = self.calculate_loss(is_upper_all)
        self.forward += 1
        return is_upper_all, all_loss

    def calculate_loss(self, is_upper_all: Tensor) -> Tensor:
        loss = torch.zeros(self.n_images)
        n_batch = math.ceil(self.n_images / config.batch_size)
        for b in range(n_batch):
            start = b * config.batch_size
            end = min((b + 1) * config.batch_size, self.n_images)
            upper = self.upper[start:end]
            lower = self.lower[start:end]
            is_upper = is_upper_all[start:end]
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=2)
            is_upper = torch.repeat_interleave(is_upper, self.split, dim=3)
            x_adv = lower.clone()
            x_adv[is_upper] = upper[is_upper]
            x_adv = x_adv.to(config.device)
            y = self.y[start:end].to(config.device)
            prediction = self.model(x_adv).softmax(dim=1)
            self.success_iter[start:end] += (prediction.argmax(dim=1) == y).cpu()
            loss[start:end] = self.criterion(prediction, y).cpu()
        return loss
