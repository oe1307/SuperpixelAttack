import foolbox as fb
import torch
from torch import Tensor
from yaspin import yaspin

from Base import Attacker
from Utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class GenAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = config.steps * config.population

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        model = fb.PyTorchModel(self.model, bounds=(0, 1), device=config.device)

        # set target class
        logits = self.model(x).detach().clone()
        idx_sorted = logits.sort(dim=1, descending=True)[1]
        target = torch.where(idx_sorted[:, 0] == y, idx_sorted[:, 1], idx_sorted[:, 0])
        criterion = fb.criteria.TargetedMisclassification(target)

        # attack
        attack = fb.attacks.GenAttack(
            steps=config.steps,
            population=config.population,
            mutation_probability=config.mutation_probability,
            mutation_range=config.mutation_range,
            sampling_temperature=config.sampling_temperature,
        )
        with yaspin(text="Attacking...", color="cyan"):
            x_adv = attack(model, x, criterion, epsilons=[config.epsilon])[1][0]

        return x_adv