import heapq

import numpy as np
import torch

from utils import config_parser

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
            breakpoint()
            if config.update_area == "superpixel" and config.channel_wise:
                # insert
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.best_loss[update] = loss[update]

                # deletion
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.best_loss[update] = loss[update]

            elif config.update_area == "superpixel":
                # insert
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.best_loss[update] = loss[update]

                # deletion
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.best_loss[update] = loss[update]

            elif config.update_area == "split_square" and config.channel_wise:
                # insert
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.best_loss[update] = loss[update]

                # deletion
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.best_loss[update] = loss[update]

            elif config.update_area == "split_square":
                # insert
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.best_loss[update] = loss[update]

                # deletion
                update = loss > self.best_loss
                self.is_upper_best[update] = is_upper[update]
                self.best_loss[update] = loss[update]

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
