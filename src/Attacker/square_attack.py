import math

import torch
import torchattacks
from torch import Tensor
from yaspin import yaspin

from base import Attacker
from utils import config_parser, setup_logger, pbar

logger = setup_logger(__name__)
config = config_parser()


class SquareAttack(Attacker):
    def __init__(self):
        config.n_forward = config.restart * config.steps

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        attacker = torchattacks.Square(
            self.model,
            config.norm,
            config.epsilon,
            config.steps,
            config.restart,
            config.p_init,
            seed=config.seed,
        )
        x_adv_all = []
        n_images = x_all.shape[0]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            pbar.debug(i + 1, n_batch, "batch")
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            x_adv = attacker.perturb(x, y)
            x_adv_all.append(x_adv)
        x_adv_all = torch.cat(x_adv_all, dim=0)
        return x_adv_all
