from utils import config_parser

from .gen_attack_foolbox import FoolboxGenAttacker

# proposed method

config = config_parser()


def get_attacker():
    if config.attacker == "FoolboxGenAttacker":
        attacker = FoolboxGenAttacker()
    else:
        raise NotImplementedError(config.attacker)
    return attacker
