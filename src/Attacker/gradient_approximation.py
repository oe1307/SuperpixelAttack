import advertorch
import torch
from torch import Tensor
from yaspin import yaspin

from Base import Attacker
from Utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class GradientApproximation(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = config.sampling * config.steps

    @torch.enable_grad()
    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        attack = advertorch.attacks.NESAttack(
            self.model,
            eps=config.epsilon,
            nb_samples=config.sampling,
            fd_eta=config.estimation_step_size,
            nb_iter=config.steps,
            eps_iter=config.step_size,
            rand_init=config.random_initialize,
            clip_min=0.0,
            clip_max=1.0,
            targeted=False,
        )
        with yaspin(text="Attacking...", color="cyan"):
            x_adv = attack.perturb(x, y)
        torch.cuda.empty_cache()
        return x_adv
