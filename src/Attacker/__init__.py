from base import Attacker
from utils import config_parser

from .gen_attack import GenAttack
from .hals import HALS
from .proposed_base import BaseProposedMethod
from .proposed_color import ColorProposedMethod
from .proposed_tabu import TabuProposedMethod
from .square_attack import SquareAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "GenAttack":
        return GenAttack()
    elif config.attacker == "HALS":
        return HALS()
    elif config.attacker == "SquareAttack":
        return SquareAttack()
    elif config.attacker == "BaseProposedMethod":
        return BaseProposedMethod()
    elif config.attacker == "ColorProposedMethod":
        return ColorProposedMethod()
    elif config.attacker == "TabuProposedMethod":
        return TabuProposedMethod()
    else:
        raise NotImplementedError(config.attacker)
