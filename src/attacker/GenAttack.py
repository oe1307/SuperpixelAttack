import math

import torch

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class GenAttacker(Attacker):
    """GenAttacker"""

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def _attack(self, x, y):
        population = x.unsqueeze(1).repeat(1, config.population, 1, 1, 1)
        upper = (population + config.epsilon).clamp(0, 1).clone()
        lower = (population - config.epsilon).clamp(0, 1).clone()

        # Create initial generation
        probability = config.probability
        mutation = config.mutation
        is_upper = torch.randint_like(population, 0, 2, dtype=torch.bool)
        population = torch.where(is_upper, upper, lower)
        logits = []
        for i in range(config.population):
            logits.append(self.robust_acc(population[:, i], y)[1])
        logits = torch.stack(logits, dim=0)
        logits_sorted, idx_sorted = logits.sort(dim=2, descending=True)

        for iter in range(config.iteration):
            # find the elite members
            acc = idx_sorted[:, :, 0] == y
            target_logits = torch.where(
                acc, logits_sorted[:, :, 1], logits_sorted[:, :, 0]
            )
            logits_sum = logits_sorted.sum(dim=2)
            assert (target_logits > 0).all()
            fitness = torch.log(target_logits + 1e-30) - torch.log(
                logits_sum - target_logits + 1e-30
            )
            x_adv_idx = fitness.argmax(dim=0)
            breakpoint()
            x_adv = population[x_adv]

            # compute selection probability

            # update parameters

            # compute fitness
            self.compute_fitness(population)

        return x_adv
