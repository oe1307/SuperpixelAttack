import numpy as np
import torch

from utils import config_parser, setup_logger

from .initial_point import InitialPoint

logger = setup_logger(__name__)
config = config_parser()


class UpdateMethod(InitialPoint):
    def __init__(self):
        super().__init__()

    def step(self, update_area: np.ndarray):
        batch, n_chanel = self.x_best.shape[:2]
        is_upper = self.is_upper_best.clone()

        if config.update_method == "greedy_local_search":
            # TODO: batch処理
            for idx in range(batch):
                if self.forward[idx] >= self.checkpoint[idx]:
                    assert self.targets[idx].shape[0] == 0
                    n_update_area = update_area[idx].max()
                    chanel = np.tile(np.arange(n_chanel), n_update_area)
                    labels = np.repeat(range(1, n_update_area + 1), n_chanel)
                    self.targets[idx] = np.stack([chanel, labels], axis=1)
                    np.random.shuffle(self.targets[idx])
                    self.checkpoint[idx] += n_update_area * n_chanel
                c, label = self.targets[idx][0]
                self.targets[idx] = np.delete(self.targets[idx], 0, axis=0)
                is_upper[idx, c, update_area[idx] == label] = ~is_upper[
                    idx, c, update_area[idx] == label
                ]
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            self.forward += 1
            update = loss >= self.best_loss
            self.x_best[update] = x_adv[update]
            self.best_loss[update] = loss[update]
            self.is_upper_best[update] = is_upper[update]

        elif config.update_method == "accelerated_local_search":
            pass

        elif config.update_method == "refine_search":
            pass

        elif config.update_method == "uniform_distribution":
            for idx in range(batch):
                if self.forward[idx] >= self.checkpoint[idx]:
                    assert self.targets[idx].shape[0] == 0
                    n_update_area = update_area[idx].max()
                    chanel = np.tile(np.arange(n_chanel), n_update_area)
                    labels = np.repeat(range(1, n_update_area + 1), n_chanel)
                    self.targets[idx] = np.stack([chanel, labels], axis=1)
                    np.random.shuffle(self.targets[idx])
                    self.checkpoint[idx] += n_update_area * n_chanel
                c, label = self.targets[idx][0]
                self.targets[idx] = np.delete(self.targets[idx], 0, axis=0)
                distribution = (
                    torch.rand(x_adv.shape[1:], device=config.device) * 2 - 1
                ) * config.epsilon
                x_adv[idx, c, update_area[idx] == label] += distribution
            x_adv = x_adv.clamp(self.lower, self.upper)
            x_adv = torch.where(is_upper, self.upper, self.lower)
            pred = self.model(x_adv).softmax(dim=1)
            loss = self.criterion(pred, self.y)
            self.forward += 1
            update = loss >= self.best_loss
            self.x_best[update] = x_adv[update]
            self.best_loss[update] = loss[update]
            self.is_upper_best[update] = is_upper[update]

        else:
            raise ValueError(config.update_method)

        return self.x_best, self.forward, self.checkpoint
