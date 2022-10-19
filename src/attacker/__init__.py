from utils import config_parser

from .auto_pgd import AutoPGD
from .fgsm import FGSM
from .gen_attack import GenAttacker
from .gen_attack2 import GenAttacker2
from .hals import HALS
from .pgd import PGD
from .square_attack2 import SquareAttack2

# TabuAttack
from .TabuAttack.local_search import LocalSearch
from .TabuAttack.method1 import TabuAttack1
from .TabuAttack.method2 import TabuAttack2
from .TabuAttack.method3 import TabuAttack3
from .TabuAttack.method5 import TabuAttack5

config = config_parser()


def get_attacker():
    if config.attacker == "AutoPGD":
        attacker = AutoPGD()
    elif config.attacker == "FGSM":
        attacker = FGSM()
    elif config.attacker == "GenAttack":
        attacker = GenAttacker()
    elif config.attacker == "GenAttack2":
        attacker = GenAttacker2()
    elif config.attacker == "HALS":
        attacker = HALS()
    elif config.attacker == "PGD":
        attacker = PGD()
    elif config.attacker == "SquareAttack2":
        attacker = SquareAttack2()
    elif config.attacker == "LocalSearch":
        attacker = LocalSearch()
    elif config.attacker == "TabuAttack1":
        attacker = TabuAttack1()
    elif config.attacker == "TabuAttack2":
        attacker = TabuAttack2()
    elif config.attacker == "TabuAttack3":
        attacker = TabuAttack3()
    elif config.attacker == "TabuAttack5":
        attacker = TabuAttack5()
    else:
        raise NotImplementedError(f"Attacker {config.attacker} is not implemented.")

    return attacker
