import math

import torch
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser, pbar, setup_logger

from .UpdateArea import set_update_area
from .UpdateMethod import set_update_method

logger = setup_logger(__name__)
config = config_parser()


class ProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.step
        self.criterion = get_criterion()
        self.update_area = set_update_area()
        self.update_method = set_update_method()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        self.update_method.set(self.model, self.criterion)

        x_adv_all = []
        n_images = x_all.shape[0]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for b in range(n_batch):
            start = b * self.model.batch_size
            end = min((b + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()

            # initialize
            forward = self.update_method.initialize(x, y, lower, upper)
            update_area, targets = self.update_area.initialize(x, forward)
            pbar.debug(forward.min(), config.step, "forward")

            # search
            while forward.min() < config.step:
                x_best, forward, targets = self.update_method.step(update_area, targets)
                update_area, targets = self.update_area.next(forward, targets)
                pbar.debug(forward.min(), config.step, "forward")

            x_adv_all.append(x_best)
        x_adv_all = torch.concat(x_adv_all)
        return x_adv_all
