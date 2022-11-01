from Base import Attacker
from Utils import config_parser

from .boundary_attack import BoundaryAttack
from .gen_attack import GenAttack
from .gradient_approximation import GradientApproximation
from .hals import HALS
from .square_attack import SquareAttack
from .tabu_attack import TabuAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "BoundaryAttack":
        return BoundaryAttack()
    elif config.attacker == "GenAttack":
        return GenAttack()
    elif config.attacker == "GradientApproximation":
        return GradientApproximation()
    elif config.attacker == "HALS":
        return HALS()
    elif config.attacker == "SquareAttack":
        return SquareAttack()
    elif config.attacker == "TabuAttack":
        return TabuAttack()
    else:
        raise NotImplementedError(config.attacker)
