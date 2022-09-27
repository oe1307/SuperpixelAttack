from utils import config_parser

from ._GenAttack import _GenAttacker
from .AutoPGD import AutoPGD_Attacker
from .GenAttack import GenAttacker
from .HALS import HALS_Attacker
from .PGD import PGD_Attacker

config = config_parser()


def get_attacker():

    if config.attacker == "_GenAttack":
        attacker = _GenAttacker()
    elif config.attacker == "AutoPGD":
        attacker = AutoPGD_Attacker()
    elif config.attacker == "GenAttack":
        attacker = GenAttacker()
    elif config.attacker == "HALS":
        attacker = HALS_Attacker()
    elif config.attacker == "PGD":
        attacker = PGD_Attacker()
    else:
        raise NotImplementedError(f"Attacker {config.attacker} is not implemented.")

    return attacker
