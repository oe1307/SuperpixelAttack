from base import Attacker
from utils import config_parser, setup_logger

from .advanced_local_search import AdvancedLocalSearch
from .gen_attack import GenAttack
from .hals import HALS
from .local_search import LocalSearch
from .only_boundary import BoundaryLocalSearch
from .plus_boundary import BoundaryPlus
from .saliency_attack import SaliencyAttack
from .square_attack import SquareAttack
from .tabu_search import TabuSearch

logger = setup_logger(__name__)
config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "AdvancedLocalSearch":
        attacker = AdvancedLocalSearch()
    elif config.attacker == "GenAttack":
        attacker = GenAttack()
    elif config.attacker == "HALS":
        attacker = HALS()
    elif config.attacker == "LocalSearch":
        attacker = LocalSearch()
    elif config.attacker == "BoundaryLocalSearch":
        attacker = BoundaryLocalSearch()
    elif config.attacker == "BoundaryPlus":
        attacker = BoundaryPlus()
    elif config.attacker == "SaliencyAttack":
        attacker = SaliencyAttack()
    elif config.attacker == "SquareAttack":
        attacker = SquareAttack()
    elif config.attacker == "TabuSearch":
        attacker = TabuSearch()
    else:
        raise NotImplementedError(config.attacker)
    logger.debug("Set attacker")
    return attacker
