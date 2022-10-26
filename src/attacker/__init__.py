from utils import config_parser

from .gen_attack import GenAttacker

# proposed method

config = config_parser()


def get_attacker():
    if config.attacker == "GenAttacker":
        attacker = GenAttacker()
    else:
        raise NotImplementedError(config.attacker)
    return attacker
