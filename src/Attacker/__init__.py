from base import Attacker
from utils import config_parser

from .proposed_method import ProposedMethod
from .square_attack import SquareAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "SquareAttack":
        return SquareAttack()
    elif config.attacker == "ProposedMethod":
        return ProposedMethod()
    else:
        raise NotImplementedError(config.attacker)
