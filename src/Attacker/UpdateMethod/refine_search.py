import numpy as np
import torch
from torch import Tensor

from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class RefineSearch:
    def __init__(self, update_area):
        if config.update_area != "superpixel":
            raise ValueError("Update area is only available for superpixel.")
        self.update_area = update_area

    def set(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def initialize(self, x: Tensor, y: Tensor, lower: Tensor, upper: Tensor):
        self.batch, self.n_channel, self.height, self.width = x.shape
        self.y = y.clone()
        self.upper = upper.clone()
        self.lower = lower.clone()

        self.level = 0
        self.area = self.update_area.initialize(x, np.zeros(self.batch, dtype=int))
        self.targets = []
        for idx in range(self.batch):
            labels = np.unique(self.area[idx])
            labels = labels[labels != 0]
            channel = np.tile(np.arange(self.n_channel), len(labels))
            labels = np.repeat(labels, self.n_channel)
            channel_labels = np.stack([channel, labels], axis=1)
            self.targets.append(np.random.permutation(channel_labels))
        self.x_adv = x.clone()
        self.best_loss = -100 * torch.ones(self.batch, device=config.device)
        self.forward = np.zeros(self.batch, dtype=int)
        return self.forward

    def step(self):
        self.refine(self.area.copy(), self.targets.copy(), self.level)
        self.level = min(self.level + 1, len(config.segments) - 1)
        for idx in range(self.batch):
            self.area[idx] = self.update_area.update(idx, self.level)
            labels = np.unique(self.area[idx])
            labels = labels[labels != 0]
            channel = np.tile(np.arange(self.n_channel), len(labels))
            labels = np.repeat(labels, self.n_channel)
            channel_labels = np.stack([channel, labels], axis=1)
            self.targets[idx] = np.random.permutation(channel_labels)
        return self.x_adv, self.forward

    def refine(self, area, targets, level):
        n_targets = np.array([len(t) for t in targets])
        upper_loss = [[] for _ in range(self.batch)]
        lower_loss = [[] for _ in range(self.batch)]
        for t in range(n_targets.max()):
            x_upper = self.x_adv.clone()
            x_lower = self.x_adv.clone()
            for idx in range(self.batch):
                if n_targets[idx] <= t or self.forward[idx] >= config.step:
                    continue
                c, label = targets[idx][t]
                x_upper[idx, c, area[idx] == label] = self.upper[
                    idx, c, area[idx] == label
                ]
                x_lower[idx, c, area[idx] == label] = self.lower[
                    idx, c, area[idx] == label
                ]
            pred = self.model(x_upper).softmax(dim=1)
            _upper_loss = self.criterion(pred, self.y)
            pred = self.model(x_lower).softmax(dim=1)
            _lower_loss = self.criterion(pred, self.y)
            for idx in range(self.batch):
                if n_targets[idx] <= t or self.forward[idx] >= config.step:
                    continue
                upper_loss[idx].append(_upper_loss[idx].item())
                lower_loss[idx].append(_lower_loss[idx].item())
                if level == 0:
                    self.forward[idx] += 2
                else:
                    self.forward[idx] += 1
            pbar.debug(
                t + 1,
                n_targets.max(),
                f"{level = }",
                f"{self.forward.mean() = :.2f}",
            )
            if np.logical_or(n_targets <= t, self.forward >= config.step).all():
                logger.debug("")
                break

        lower_loss = [torch.tensor(loss) for loss in lower_loss]
        upper_loss = [torch.tensor(loss) for loss in upper_loss]
        loss_storage = [
            torch.stack([lower_loss[idx], upper_loss[idx]]) for idx in range(self.batch)
        ]
        u_is_better = [loss.argmax(dim=0).to(bool) for loss in loss_storage]
        loss_storage = [loss.max(dim=0)[0] for loss in loss_storage]
        indices = [loss.argsort(descending=True) for loss in loss_storage]
        n_indices = np.array([len(idx) for idx in indices])
        level += 1
        for t in range(n_indices.max()):
            next_targets = targets.copy()
            next_area = area.copy()
            for idx in range(self.batch):
                if len(indices[idx]) > t:
                    loss = loss_storage[idx][indices[idx][t]]
                    update = (loss >= self.best_loss[idx]).item()
                    upper_update = u_is_better[idx][indices[idx][t]].item() & update
                    lower_update = ~u_is_better[idx][indices[idx][t]].item() & update
                    c, label = targets[idx][indices[idx][t]]
                    if upper_update:
                        self.x_adv[idx, c, area[idx] == label] = self.upper[
                            idx, c, area[idx] == label
                        ]
                    elif lower_update:
                        self.x_adv[idx, c, area[idx] == label] = self.lower[
                            idx, c, area[idx] == label
                        ]
                    if (
                        self.forward[idx] < config.step
                        and level < len(config.segments) - 1
                    ):
                        next_area[idx] = self.update_area.update(idx, level)
                        pair = np.stack(
                            [area[idx].reshape(-1), next_area[idx].reshape(-1)]
                        ).T
                        pair = np.unique(pair, axis=0)
                        c, label = targets[idx][indices[idx][t]]
                        labels = pair[pair[:, 0] == label][:, 1]
                        channel = np.ones_like(labels) * c
                        _target = np.stack([channel, labels], axis=1)
                        next_targets[idx] = np.random.permutation(_target)
                else:
                    next_targets[idx] = np.array([])
            if self.forward.min() < config.step and level < len(config.segments) - 1:
                self.refine(next_area.copy(), next_targets.copy(), level)
