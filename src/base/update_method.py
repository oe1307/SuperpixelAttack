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

        if config.update_method not in (
            "greedy_local_search",
            "accelerated_local_search",
            "refine_search",
            "uniform_distribution",
        ):
            raise NotImplementedError(config.update_method)

    def step(self, update_area: np.ndarray, targets: List[np.ndarray]):
        batch = np.arange(self.x_adv.shape[0])

        if config.update_method == "greedy_local_search":
            self.is_upper = self.is_upper_best.clone()
            # TODO: batch処理
            for idx in batch:
                c, label = targets[idx][0]
                self.is_upper[idx, c, update_area[idx] == label] = ~self.is_upper[
                    idx, c, update_area[idx] == label
                ]
            self.x_adv = torch.where(self.is_upper, self.upper, self.lower)
            pred = self.model(self.x_adv).softmax(dim=1)
            self.loss = self.criterion(pred, self.y)
            self.forward += 1
            update = self.loss >= self.best_loss
            self.is_upper_best[update] = self.is_upper[update]
            self.x_best[update] = self.x_adv[update]
            self.best_loss[update] = self.loss[update]

        elif config.update_method == "accelerated_local_search":
            pass

        elif config.update_method == "refine_search":
            pass

        elif config.update_method == "uniform_distribution":
            labels = [np.unique(targets[idx][:, 1]) for idx in batch]
            n_labels = np.array([len(label) for label in labels])
            for label in range(1, n_labels.max() + 1):
                self.x_adv = self.x_adv.permute(0, 2, 3, 1)
                for idx in batch:
                    rand = (
                        2 * torch.rand_like(self.x_adv[idx, update_area[idx] == label])
                        - 1
                    ) * config.epsilon
                    self.x_adv[idx, update_area[idx] == label] = (
                        self.x_adv[idx, update_area[idx] == label] + rand
                    )
                self.x_adv = self.x_adv.permute(0, 3, 1, 2).clamp(
                    self.lower, self.upper
                )
                pred = self.model(self.x_adv).softmax(dim=1)
                self.loss = self.criterion(pred, self.y)
                update = self.loss >= self.best_loss
                update = np.logical_and(update.cpu().numpy(), label <= n_labels)
                self.is_upper_best[update] = self.is_upper[update]
                self.x_best[update] = self.x_adv[update]
                self.best_loss[update] = self.loss[update]
            self.forward += n_labels
            # FIXME: here
            targets = [np.zeros((1, 1)) for _ in batch]
            for idx in batch:
                targets[idx] = np.delete(targets[idx], 0, axis=0)

        else:
            raise ValueError(config.update_method)

        return self.x_best, self.forward, targets
