from utils import config_parser

from .APGD import APGD_Attacker
from .HALS import HALS_Attacker
from .PGD import PGD_Attacker

config = config_parser()


def get_attacker():

    if config.attacker == "HALS":
        attacker = HALS_Attacker()
    elif config.attacker == "APGD":
        attacker = APGD_Attacker()
    elif config.attacker == "PGD":
        attacker = PGD_Attacker()
    else:
        raise NotImplementedError(f"Attacker {config.attacker} is not implemented.")

    return attacker
