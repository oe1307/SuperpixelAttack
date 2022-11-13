import numpy as np
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class ProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.forward

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        # TODO: batch処理
        for x, y in zip(x_all, y_all):
            x_adv = x.clone()

        return x_adv
