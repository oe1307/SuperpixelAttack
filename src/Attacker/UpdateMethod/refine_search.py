import numpy as np
import torch
import math

from utils import config_parser, setup_logger, pbar

from .base_method import BaseMethod

logger = setup_logger(__name__)
config = config_parser()


class RefineSearch(BaseMethod):
    def __init__(self, update_area):
        super().__init__(update_area)
        if config.update_area == "superpixel":
            self.max_level = len(config.segments)
        elif config.update_area == "equally_divided_squares":
            self.max_level = math.log2(config.initial_split)
        elif config.update_area == "saliency_map":
            self.max_level = math.log2(config.k_int)
        elif config.update_area == "random_square":
            raise NotImplementedError("HALS does not support random_square")

    def step(self):
        self.refine(self.area.copy(), self.targets.copy(), self.level.copy())
        self.level = np.minimum(self.level + 1, self.max_level).astype(int)
        for idx in range(self.batch):
            self.area[idx] = self.update_area.update(idx, self.level[idx])
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
        return self.x_best, self.forward

    def refine(self, area, targets, level):
        n_targets = np.array([len(t) for t in targets])
        loss_storage = [[] for _ in range(self.batch)]
        for t in range(n_targets.max()):
            is_upper = self.is_upper_best.clone()
            if config.channel_wise:
                for idx in range(self.batch):
                    if n_targets[idx] <= t or self.forward[idx] >= config.step:
                        continue
                    c, label = targets[idx][t]
                    is_upper[idx, c, area[idx] == label] = ~is_upper[
                        idx, c, area[idx] == label
                    ]
                    self.forward[idx] += 1
            else:
                for idx in range(self.batch):
                    if n_targets[idx] <= t or self.forward[idx] >= config.step:
                        continue
                    label = targets[idx][t]
                    is_upper.permute(0, 2, 3, 1)[
                        idx, area[idx] == label
                    ] = ~is_upper.permute(0, 2, 3, 1)[idx, area[idx] == label]
                    self.forward[idx] += 1
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            for idx in range(self.batch):
                if n_targets[idx] > t:
                    loss_storage[idx].append(loss[idx].item())
            pbar.debug(
                t + 1,
                max(n_targets),
                f"level = {level.max()}",
                f"forward = {self.forward.mean():.2f}",
            )
            if np.logical_or(n_targets <= t, self.forward >= config.step).all():
                logger.debug("")
                break
        loss_storage = [np.array(loss) for loss in loss_storage]
        indicies = [loss.argsort()[::-1] for loss in loss_storage]
        level += 1
        for t in range(n_targets.max()):
            next_targets = targets.copy()
            next_area = area.copy()
            for idx in range(self.batch):
                if len(indicies[idx]) > t:
                    loss = loss_storage[idx][indicies[idx][t]]
                    update = (loss >= self.best_loss[idx]).item()
                    if config.channel_wise and update:
                        c, label = targets[idx][indicies[idx][t]]
                        self.is_upper_best[
                            idx, c, area[idx] == label
                        ] = ~self.is_upper_best[idx, c, area[idx] == label]
                    elif update:
                        label = targets[idx][indicies[idx][t]]
                        self.x_best.permute(0, 2, 3, 1)[
                            idx, area[idx] == label
                        ] = self.lower.permute(0, 2, 3, 1)[idx, area[idx] == label]
                    if self.forward[idx] < config.step and level.max() < self.max_level:
                        next_area[idx] = self.update_area.update(idx, level[idx])
                        pair = np.stack(
                            [area[idx].reshape(-1), next_area[idx].reshape(-1)]
                        ).T
                        pair = np.unique(pair, axis=0)
                        if config.channel_wise:
                            c, label = targets[idx][indicies[idx][t]]
                            labels = pair[pair[:, 0] == label][:, 1]
                            channel = np.ones_like(labels) * c
                            _target = np.stack([channel, labels], axis=1)
                            next_targets[idx] = np.random.permutation(_target)
                        else:
                            label = targets[idx][indicies[idx][t]]
                            labels = pair[pair[:, 0] == label][:, 1]
                            next_targets[idx] = np.random.permutation(labels)
                else:
                    next_targets[idx] = np.array([])
            if self.forward.min() < config.step and level.max() < self.max_level:
                self.refine(next_area.copy(), next_targets.copy(), level.copy())
