import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, UpdateArea, UpdateMethod, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class ProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.steps
        self.criterion = get_criterion()
        self.update_area = UpdateArea()
        self.update_method = UpdateMethod()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_images = x_all.shape[0]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for b in range(n_batch):
            start = b * self.model.batch_size
            end = min((b + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            batch = np.arange(x.shape[0])
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()

            # initialize
            update_area = self.update_area.initialize(x)
            self.update_method.initialize(x, y, update_area, lower, upper)

            # search
            while (self.update_method.forward < config.steps).any():
                x_best = self.update_method.update()

            x_adv_all.append(x_best)
        x_adv_all = torch.concat(x_adv_all)
        return x_adv_all
