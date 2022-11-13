from base import Attacker
from utils import config_parser

from .square_attack import SquareAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "SquareAttack":
        return SquareAttack()
    else:
        raise NotImplementedError(config.attacker)
