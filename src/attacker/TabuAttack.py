from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class TabuSearch(Attacker):
    def __init__(self):
        super.__init__()

    def _attack(self, x: Tensor, y: Tensor):
        raise NotImplementedError("Tabu Search is not implemented yet.")
