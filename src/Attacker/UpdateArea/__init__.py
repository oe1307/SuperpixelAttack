from utils import config_parser

from .random_square import RandomSquare
from .saliency_map import SaliencyMap
from .split_square import SplitSquare
from .superpixel import Superpixel

config = config_parser()


def set_update_area():
    if config.update_area == "superpixel":
        update_area = Superpixel()
    elif config.update_area == "split_square":
        update_area = SplitSquare()
    elif config.update_area == "saliency_map":
        update_area = SaliencyMap()
    elif config.update_area == "random_square":
        update_area = RandomSquare()
    else:
        raise NotImplementedError(config.update_area)
    return update_area
