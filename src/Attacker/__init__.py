from base import Attacker
from utils import config_parser

from .proposed_base import BaseProposedMethod
from .proposed_color import ColorProposedMethod
from .square_attack import SquareAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "SquareAttack":
        return SquareAttack()
    elif config.attacker == "BaseProposedMethod":
        return BaseProposedMethod()
    elif config.attacker == "ColorProposedMethod":
        return ColorProposedMethod()
    else:
        raise NotImplementedError(config.attacker)
