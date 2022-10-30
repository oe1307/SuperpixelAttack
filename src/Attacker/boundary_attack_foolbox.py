import foolbox as fb
from torch import Tensor

from Base import Attacker
from Utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class FoolboxBoundaryAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = 0

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        attacker = fb.attacks.BoundaryAttack(
            # init_attack: Optional[MinimizationAttack] = None,
            # steps: int = 25000,
            # spherical_step: float = 1e-2,
            # source_step: float = 1e-2,
            # source_step_convergance: float = 1e-7,
            # step_adaptation: float = 1.5,
            # tensorboard: Union[Literal[False], None, str] = False,
            # update_stats_every_k: int = 10,
        )
        raise NotImplementedError(attacker)
