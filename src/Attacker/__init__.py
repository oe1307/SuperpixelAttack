from base import Attacker
from utils import config_parser

from .advanced_local_search import AdvancedLocalSearch
from .gen_attack import GenAttack
from .hals import HALS
from .local_search import LocalSearch
from .only_boundary import BoundaryLocalSearch
from .plus_boundary import BoundaryPlus
from .saliency_attack import SaliencyAttack
from .square_attack import SquareAttack
from .tabu_search import TabuSearch

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "AdvancedLocalSearch":
        return AdvancedLocalSearch()
    elif config.attacker == "GenAttack":
        return GenAttack()
    elif config.attacker == "HALS":
        return HALS()
    elif config.attacker == "LocalSearch":
        return LocalSearch()
    elif config.attacker == "BoundaryLocalSearch":
        return BoundaryLocalSearch()
    elif config.attacker == "BoundaryPlus":
        return BoundaryPlus()
    elif config.attacker == "SaliencyAttack":
        return SaliencyAttack()
    elif config.attacker == "SquareAttack":
        return SquareAttack()
    elif config.attacker == "TabuSearch":
        return TabuSearch()
    else:
        raise NotImplementedError(config.attacker)
