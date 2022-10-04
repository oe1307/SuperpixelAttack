from utils import config_parser

from .AutoPGD import AutoPGD
from .GenAttack import GenAttacker
from .GenAttack2 import GenAttacker2
from .HALS import HALS
from .PGD import PGD

config = config_parser()


def get_attacker():
    if config.attacker == "AutoPGD":
        attacker = AutoPGD()
    elif config.attacker == "GenAttack":
        attacker = GenAttacker()
    elif config.attacker == "GenAttack2":
        attacker = GenAttacker2()
    elif config.attacker == "HALS":
        attacker = HALS()
    elif config.attacker == "PGD":
        attacker = PGD()
    else:
        raise NotImplementedError(f"Attacker {config.attacker} is not implemented.")

    return attacker
