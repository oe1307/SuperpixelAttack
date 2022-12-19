import numpy as np
import torch
import math

from utils import config_parser, setup_logger, pbar

from .base_method import BaseMethod

logger = setup_logger(__name__)
config = config_parser()


class RefineSearch(BaseMethod):
    def __init__(self):
        super().__init__()
        if config.update_area == "superpixel":
            self.max_level = len(config.segments)
        elif config.update_area == "equally_divided_squares":
            self.max_level = math.log2(config.initial_split)
        elif config.update_area == "saliency_map":
            self.max_level = math.log2(config.k_int)
        elif config.update_area == "random_square":
            raise NotImplementedError("HALS does not support random_square")

    def step(self):
        self.refine(self.level)
        self.level = (self.level + 1).clip(0, self.max_level)
        return self.forward

    def refine(self, level):
        n_targets = max([len(t) for t in self.targets])
        upper_loss = -100 * torch.ones((n_targets, self.batch), device=config.device)
        lower_loss = -100 * torch.ones((n_targets, self.batch), device=config.device)
        for t in range(n_targets):
            if config.chennel_wise:
                x_upper = self.x_best.clone()
                x_lower = self.x_best.clone()
                for idx in range(self.batch):
                    if self.targets[idx].shape[0] > t + 1:
                        continue
                    c, label = self.targets[idx][t]
                    x_upper[idx, c, self.area == label] = self.upper[
                        idx, c, self.area == label
                    ]
                    x_lower[idx, c, self.area == label] = self.lower[
                        idx, c, self.area == label
                    ]
                    self.forward[idx] += 2
            else:
                for idx in range(self.batch):
                    if self.targets[idx].shape[0] > t + 1:
                        continue
                    label = self.targets[idx][t]
                    x_upper.permute(0, 2, 3, 1)[
                        idx, self.area == label
                    ] = self.upper.permute(0, 2, 3, 1)[idx, self.area == label]
                    x_lower.permute(0, 2, 3, 1)[
                        idx, self.area == label
                    ] = self.lower.permute(0, 2, 3, 1)[idx, self.area == label]
                    self.forward[idx] += 2
                x_upper = x_upper.permute(0, 3, 1, 2)
                x_lower = x_lower.permute(0, 3, 1, 2)
            pred = self.model(x_upper).softmax(dim=1)
            upper_loss[t, self.forward < config.step] = self.criterion(pred, self.y)
            pred = self.model(x_lower).softmax(dim=1)
            lower_loss[t, self.forward < config.step] = self.criterion(pred, self.y)
            pbar.debug(t + 1, n_targets, f"forward = {self.forward.min()}")
            if self.forward.min() >= config.step:
                logger.debug("")
                break
        loss_storage, u_is_better = torch.stack([lower_loss, upper_loss]).max(dim=0)
        indices = loss_storage.argsort(dim=0, descending=True)
        for index in indices:
            is_upper = u_is_better[index, np.arange(self.batch)].to(torch.bool)
            loss = loss_storage[index, np.arange(self.batch)]
            update = loss >= self.best_loss
            upper_update = (update & is_upper).cpu().numpy()
            lower_update = (update & ~is_upper).cpu().numpy()
            for idx in range(self.batch):
                if config.channel_wise and upper_update[idx]:
                    c, label = self.targets[idx][index[idx]]
                    self.x_best[idx, c, self.area == label] = self.upper[
                        idx, c, self.area == label
                    ]
                elif upper_update[idx]:
                    label = self.targets[idx][index[idx]]
                    self.x_best.permute(0, 2, 3, 1)[
                        idx, self.area == label
                    ] = self.upper.permute(0, 2, 3, 1)[idx, self.area == label]
                elif config.channel_wise and lower_update[idx]:
                    c, label = self.targets[idx][index[idx]]
                    self.x_best[idx, c, self.area == label] = self.lower[
                        idx, c, self.area == label
                    ]
                elif lower_update[idx]:
                    label = self.targets[idx][index[idx]]
                    self.x_best.permute(0, 2, 3, 1)[
                        idx, self.area == label
                    ] = self.lower.permute(0, 2, 3, 1)[idx, self.area == label]
                if level[idx] < self.max_level:
                    level[idx] += 1
                    for idx in range(self.batch):
                        next_area = self.update_area.update(idx, level[idx])
                        pair = np.stack([self.area[idx], next_area], axis=0)
                        pair = np.unique(pair, axis=0)
                        if config.channel_wise:
                            c, label = self.targets[idx][index[idx]]
                            labels = pair[pair[:, 0] == label][:, 1]
                            channel = np.ones_like(labels) * c
                            _target = np.stack([channel, labels], axis=1)
                            self.targets[idx] = np.random.permutation(_target)
                            self.area = next_area
                        else:
                            label = self.targets[idx][index[idx]]
                            labels = pair[pair[:, 0] == label][:, 1]
                            self.targets[idx] = np.random.permutation(labels)
                            self.area = next_area
            self.refine(level)
