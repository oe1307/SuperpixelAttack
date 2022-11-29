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
        # remove misclassification images
        n_images = x_all.shape[0]
        clean_acc = torch.zeros(n_images, dtype=torch.bool)
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            logits = self.model(x).clone()
            clean_acc[start:end] = logits.argmax(dim=1) == y

        torch.use_deterministic_algorithms(False)  # for advertorch
        attacker = LinfGenAttack(
            self.model,
            config.epsilon,
            # hyperparameter
            nb_samples=config.population,
            nb_iter=config.steps,
            tau=config.tau,
            alpha_init=config.alpha_init,
            rho_init=config.rho_init,
            decay=config.decay,
        )
        x_adv_all = x_all.clone()
        n_batch = math.ceil(clean_acc.sum() / self.model.batch_size)
        for i in range(n_batch):
            pbar.debug(i + 1, n_batch, "batch")
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[clean_acc][start:end]
            y = y_all[clean_acc][start:end]

            with torch.cuda.device(config.device):
                torch.set_default_tensor_type(torch.cuda.FloatTensor)  # use cuda
                x_adv = attacker.perturb(x)
                torch.set_default_tensor_type(torch.FloatTensor)
            x_adv_all[clean_acc][start:end] = x_adv
        return x_adv_all
