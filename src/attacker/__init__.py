from utils import config_parser

from .APGD import APGDAttacker
from .LazyGreedy import LazyGreedyAttacker


def get_attacker():
    config = config_parser.config

    if config.attacker == "LazyGreedy":
        attacker = LazyGreedyAttacker()
    elif config.attacker == "APGD":
        attacker = APGDAttacker()
    else:
        raise NotImplementedError(f"Attacker {config.attacker} is not implemented.")

    return attacker
