import numpy as np
import torch
from torch import Tensor

from utils import config_parser

config = config_parser()


class UpdateMethod:
    def __init__(self):
        if config.initial_point not in (
            "random",
            "lower",
            "upper",
        ):
            raise NotImplementedError(config.initial_point)
        if config.update_method not in (
            "greedy_local_search",
            "accelerated_local_search",
            "refine_search",
            "uniform_distribution",
        ):
            raise NotImplementedError(config.update_method)

    def initialize(self, x: Tensor, y: Tensor, lower: Tensor, upper: Tensor):
        batch, n_chanel = x.shape[:2]
        if config.initial_point == "random":
            is_upper = torch.randint(0, 2, x.shape, dtype=torch.bool)
            x_adv = torch.where(is_upper, upper, lower)
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            forward = 1
            return is_upper, x_adv, loss, forward

        elif config.initial_point == "lower":
            is_upper = torch.zeros_like(x, dtype=torch.bool)
            x_adv = lower.clone()
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            forward = 1

            targets = []
            for idx in range(batch):
                chanel = np.tile(np.arange(n_chanel), n_superpixel[idx])
                labels = np.repeat(range(1, n_superpixel[idx] + 1), n_chanel)
                _target = np.stack([chanel, labels], axis=1)
                np.random.shuffle(_target)
                targets.append(_target)
            n_targets = np.array([len(target) for target in targets])
            checkpoint = (config.init_checkpoint * n_targets + 1).round().astype(int)
            pre_checkpoint = np.ones_like(batch)
            return is_upper, x_adv, loss, forward

        elif config.initial_point == "upper":
            is_upper = torch.ones_like(x, dtype=torch.bool)
            x_adv = upper.clone()
            pred = self.model(x_adv).softmax(1)
            loss = self.criterion(pred, y)
            forward = 1
            return is_upper, x_adv, loss, forward

        else:
            raise ValueError(config.initial_point)

    def update(self):
        if config.update_method == "greedy_local_search":
            pass

        elif config.update_method == "accelerated_local_search":
            pass

        elif config.update_method == "refine_search":
            pass

        elif config.update_method == "uniform_distribution":
            pass

        else:
            raise ValueError(config.update_method)
