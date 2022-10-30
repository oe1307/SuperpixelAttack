import advertorch
from torch import Tensor

from Base import Attacker
from Utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class AdvertorchGenAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = config.steps * config.population

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        attacker = advertorch.attacks.GenAttack(
            # predict,
            # eps: float,
            # order,
            # loss_fn=None,
            # nb_samples=100,
            # nb_iter=40,
            # tau=0.1,
            # alpha_init=0.4,
            # rho_init=0.5,
            # decay=0.9,
            # clip_min=0., clip_max=1.,
            # targeted: bool = False
        )
        raise NotImplementedError(attacker)
