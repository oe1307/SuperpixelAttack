from typing import List

import numpy as np
import torch

from utils import config_parser, setup_logger

from .initial_point import InitialPoint

logger = setup_logger(__name__)
config = config_parser()


class UpdateMethod(InitialPoint):
    def __init__(self):
        super().__init__()

        if config.update_area not in (
            "superpixel",
            "random_square",
            "divisional_square",
        ):
            raise NotImplementedError(config.update_area)

        if config.update_method not in (
            "greedy_local_search",
            "accelerated_local_search",
            "refine_search",
            "uniform_distribution",
        ):
            raise NotImplementedError(config.update_method)

    def step(self, update_area: np.ndarray, targets: List[np.ndarray]):
        if config.update_method == "greedy_local_search":
            assert False

        elif config.update_method == "accelerated_local_search":
            assert False

        elif config.update_method == "refine_search":
            assert False

        elif config.update_method == "uniform_distribution":
            if config.update_area == "superpixel" and config.channel_wise:
                for idx in range(self.batch):
                    c, label = targets[idx][0]
                    rand = torch.rand_like(
                        self.x_adv[idx, c, update_area[idx] == label]
                    )
                    rand = (2 * rand - 1) * config.epsilon
                    self.x_adv[idx, c, update_area[idx] == label] += rand
                self.x_adv = self.x_adv.clamp(self.lower, self.upper)
            elif config.update_area == "superpixel":
                for idx in range(self.batch):
                    label = targets[idx][0]
                    rand = torch.rand_like(
                        self.x_adv[idx, :, update_area[idx] == label]
                    )
                    rand = (2 * rand - 1) * config.epsilon
                    self.x_adv[idx, :, update_area[idx] == label] += rand
                self.x_adv = self.x_adv.clamp(self.lower, self.upper)
            elif config.update_area == "random_square" and config.channel_wise:
                for idx in range(self.batch):
                    c = targets[idx][0]
                    rand = torch.rand_like(self.x_adv[idx, c, update_area[idx]])
                    rand = (2 * rand - 1) * config.epsilon
                    self.x_adv[idx, c, update_area[idx]] += rand
                self.x_adv = self.x_adv.clamp(self.lower, self.upper)
            elif config.update_area == "random_square":
                self.x_adv = self.x_adv.permute(0, 2, 3, 1)
                rand = 2 * torch.rand_like(self.x_adv[update_area]) - 1
                self.x_adv[update_area] += rand * config.epsilon
                self.x_adv = self.x_adv.permute(0, 3, 1, 2)
                self.x_adv = self.x_adv.clamp(self.lower, self.upper)
            pred = self.model(self.x_adv).softmax(dim=1)
            self.loss = self.criterion(pred, self.y)
            self.forward += 1
            update = self.loss >= self.best_loss
            self.is_upper_best[update] = self.is_upper[update]
            self.x_best[update] = self.x_adv[update]
            self.best_loss[update] = self.loss[update]

        else:
            raise ValueError(config.update_method)

        return self.x_best, self.forward
