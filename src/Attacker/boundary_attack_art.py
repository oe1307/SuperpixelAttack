import art
from torch import Tensor

from Base import Attacker
from Utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class ArtBoundaryAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = 0

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        attacker = art.attacks.evasion.BoundaryAttack(
            # estimator: "CLASSIFIER_TYPE",
            # batch_size: int = 64,
            # targeted: bool = True,
            # delta: float = 0.01,
            # epsilon: float = 0.01,
            # step_adapt: float = 0.667,
            # max_iter: int = 5000,
            # num_trial: int = 25,
            # sample_size: int = 20,
            # init_size: int = 100,
            # min_epsilon: float = 0.0,
            # verbose: bool = True,
        )
        raise NotImplementedError(attacker)
