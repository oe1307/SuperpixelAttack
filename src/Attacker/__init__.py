from base import Attacker
from utils import config_parser

from .gen_attack import GenAttack
from .hals import HALS
from .local_search_improved import ImprovedLocalSearchProposedMethod
from .proposed_local_search import LocalSearchProposedMethod
from .proposed_only_boundary import BoundaryProposedMethod
from .proposed_plus_boundary import BoundaryPlusProposedMethod
from .proposed_tabu_search import TabuSearchProposedMethod
from .saliency_attack import SaliencyAttack
from .square_attack import SquareAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "GenAttack":
        return GenAttack()
    elif config.attacker == "HALS":
        return HALS()
    elif config.attacker == "ImprovedLocalSearchMethod":
        return ImprovedLocalSearchProposedMethod()
    elif config.attacker == "LocalSearchProposedMethod":
        return LocalSearchProposedMethod()
    elif config.attacker == "BoundaryProposedMethod":
        return BoundaryProposedMethod()
    elif config.attacker == "BoundaryPlusProposedMethod":
        return BoundaryPlusProposedMethod()
    elif config.attacker == "TabuSearchProposedMethod":
        return TabuSearchProposedMethod()
    elif config.attacker == "SquareAttack":
        return SquareAttack()
    elif config.attacker == "SaliencyAttack":
        return SaliencyAttack()
    else:
        raise NotImplementedError(config.attacker)
