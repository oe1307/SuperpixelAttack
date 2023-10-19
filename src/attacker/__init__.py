from base import Attacker
from utils import config_parser

from .acc_sign_hunter import AccSignHunter
from .gen_attack import GenAttack
from .geo_da import GeoDA
from .parsimonious_attack import ParsimoniousAttack
from .saliency_attack import SaliencyAttack
from .sign_hunter import SignHunter
from .square_attack import SquareAttack
from .superpixel_attack import SuperpixelAttack

config = config_parser()


def get_attacker() -> Attacker:
    if config.attacker == "SuperpixelAttack":
        attacker = SuperpixelAttack()
    elif config.attacker == "ParsimoniousAttack":
        attacker = ParsimoniousAttack()
    elif config.attacker == "SquareAttack":
        attacker = SquareAttack()
    elif config.attacker == "SignHunter":
        attacker = SignHunter()
    elif config.attacker == "AccSignHunter":
        attacker = AccSignHunter()
    elif config.attacker == "GenAttack":
        attacker = GenAttack()
    elif config.attacker == "GeoDA":
        attacker = GeoDA()
    elif config.attacker == "SaliencyAttack":
        attacker = SaliencyAttack()
    else:
        raise NotImplementedError(config.attacker)
    return attacker
