from base import Attacker
from utils import config_parser, setup_logger

from .gen_attack import GenAttack
from .parsimonious_attack import ParsimoniousAttack
from .proposed_method import ProposedMethod
from .proposed_method_with_removal import ProposedMethodWithRemoval
from .saliency_attack import SaliencyAttack
from .square_attack import SquareAttack

logger = setup_logger(__name__)
config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "ProposedMethod":
        attacker = ProposedMethod()
    elif config.attacker == "ProposedMethodWithRemoval":
        attacker = ProposedMethodWithRemoval()
    elif config.attacker == "GenAttack":
        attacker = GenAttack()
    elif config.attacker == "ParsimoniousAttack":
        attacker = ParsimoniousAttack()
    elif config.attacker == "SaliencyAttack":
        attacker = SaliencyAttack()
    elif config.attacker == "SquareAttack":
        attacker = SquareAttack()
    else:
        raise NotImplementedError(config.attacker)
    logger.debug("Set attacker")
    return attacker
