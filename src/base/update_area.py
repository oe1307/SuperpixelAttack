import numpy as np
from torch import Tensor

from utils import config_parser

from .superpixel import SuperpixelManager

config = config_parser()


class UpdateArea:
    def __init__(self):
        if config.update_area == "superpixel":
            self.superpixel_manager = SuperpixelManager()

        elif config.update_area == "random_square":
            pass

        elif config.update_area == "divisional_square":
            pass

        else:
            raise NotImplementedError(config.update_area)

    def initialize(self, x: Tensor):
        batch = np.arange(x.shape[0])

        if config.update_area == "superpixel":
            self.superpixel = self.superpixel_manager.cal_superpixel(x)
            self.level = np.zeros_like(batch)
            update_area = self.superpixel[batch, self.level]

        elif config.update_area == "random_square":
            pass

        elif config.update_area == "divisional_square":
            pass

        else:
            raise NotImplementedError(config.update_area)

        return update_area

    def update(self):
        if config.update_area == "superpixel":
            pass

        elif config.update_area == "random_square":
            pass

        elif config.update_area == "divisional_square":
            pass

        else:
            raise NotImplementedError(config.update_area)
