from torch import Tensor

from utils import config_parser

from .superpixel import SuperpixelManager

config = config_parser()


class UpdateArea:
    def __init__(self):
        pass

    def initialize(self, x: Tensor):
        if config.update_area == "superpixel":
            superpixel_manager = SuperpixelManager()
            superpixel = superpixel_manager.cal_superpixel()
            n_superpixel = superpixel.max(axis=(1, 2))
            return superpixel, n_superpixel
        elif config.update_area == "random_square":
            pass
        elif config.update_area == "divisional_square":
            pass
        else:
            raise NotImplementedError(config.update_area)

    def update(self):
        pass
