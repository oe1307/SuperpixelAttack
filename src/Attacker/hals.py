from torch import Tensor

from Base import Attacker
from Utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class HALS(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = config.steps

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError
