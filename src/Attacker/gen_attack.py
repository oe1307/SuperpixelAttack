import math

import torch
from advertorch.attacks import LinfGenAttack
from torch import Tensor

from base import Attacker
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class GenAttack(Attacker):
    def __init__(self):
        super().__init__()
        config.n_forward = config.steps * config.population

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        torch.use_deterministic_algorithms(False)  # for advertorch
        attacker = LinfGenAttack(
            self.model,
            config.epsilon,
            nb_samples=config.population,
            nb_iter=config.steps,
            tau=config.tau,
            alpha_init=config.alpha_init,
            rho_init=config.rho_init,
            decay=config.decay,
        )

        x_adv_all = []
        n_images = x_all.shape[0]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            pbar.debug(i + 1, n_batch, "batch")
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            with torch.cuda.device(config.device):
                torch.set_default_tensor_type(torch.cuda.FloatTensor)  # use cuda
                x_adv = attacker.perturb(x)
                torch.set_default_tensor_type(torch.FloatTensor)
            x_adv_all.append(x_adv)
        x_adv_all = torch.cat(x_adv_all, dim=0)
        return x_adv_all
