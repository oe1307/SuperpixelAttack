from Base import Attacker
from Utils import config_parser

from .boundary_attack_art import ArtBoundaryAttack
from .boundary_attack_foolbox import FoolboxBoundaryAttack
from .gen_attack_advertorch import AdvertorchGenAttack
from .gen_attack_foolbox import FoolboxGenAttacker
from .gradient_estimation import GradientEstimation
from .hals import HALS
from .square_attack import SquareAttack
from .tabu_attack import TabuAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "ArtBoundaryAttack":
        return ArtBoundaryAttack()
    elif config.attacker == "FoolboxBoundaryAttack":
        return FoolboxBoundaryAttack()
    elif config.attacker == "AdvertorchGenAttack":
        return AdvertorchGenAttack()
    elif config.attacker == "FoolboxGenAttack":
        return FoolboxGenAttacker()
    elif config.attacker == "GradientEstimation":
        return GradientEstimation()
    elif config.attacker == "HALS":
        return HALS()
    elif config.attacker == "SquareAttack":
        return SquareAttack()
    elif config.attacker == "TabuAttack":
        return TabuAttack()
    else:
        raise NotImplementedError(config.attacker)
