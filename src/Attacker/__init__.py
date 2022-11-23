from base import Attacker
from utils import config_parser

from .gen_attack import GenAttack
from .hals import HALS
from .proposed_attention import AttentionProposedMethod
from .proposed_local_search import LocalSearchProposedMethod
from .square_attack import SquareAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "GenAttack":
        return GenAttack()
    elif config.attacker == "HALS":
        return HALS()
    elif config.attacker == "SquareAttack":
        return SquareAttack()
    elif config.attacker == "AttentionProposedMethod":
        return AttentionProposedMethod()
    elif config.attacker == "LocalSearchProposedMethod":
        return LocalSearchProposedMethod()
    else:
        raise NotImplementedError(config.attacker)
