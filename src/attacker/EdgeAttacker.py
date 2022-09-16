from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class EdgeAttacker(Attacker):
    """Edge Attacker"""

    def __init__(self):
        super.__init__()

    def _attack(self, x: Tensor, y: Tensor):
        # upper = (x + config.epsilon).clamp(0, 1).detach().clone()
        # lower = (x - config.epsilon).clamp(0, 1).detach().clone()

        x_adv = x.clone()
        return x_adv
