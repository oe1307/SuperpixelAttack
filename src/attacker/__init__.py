from utils import config_parser

from .auto_pgd import AutoPGD
from .gen_attack import GenAttacker
from .gen_attack2 import GenAttacker2
from .hals import HALS
from .pgd import PGD
from .square_attack import SquareAttack

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
    elif config.attacker == "SquareAttack":
        attacker = SquareAttack()
    else:
        raise NotImplementedError(f"Attacker {config.attacker} is not implemented.")

    return attacker
