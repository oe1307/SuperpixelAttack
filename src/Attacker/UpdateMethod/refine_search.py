import numpy as np
import torch

from utils import config_parser, setup_logger

from .base_method import BaseMethod

logger = setup_logger(__name__)
config = config_parser()


class RefineSearch(BaseMethod):
    def __init__(self):
        super().__init__()
        if config.update_area == "random_square":
            raise NotImplementedError("HALS does not support random_square")

    def step(self, update_area: np.ndarray, targets):
        # 先にすべての分割をここで手に入れる

        k_init = config.k_init
        split_level = 1
        if config.update_area == "saliency_map":
            self.saliency_detection = torch.from_numpy(update_area != -1)
        while self.forward.min() < config.step:
            self.refine(update_area, k_init, split_level)
            if k_init > 1:
                assert k_init % 2 == 0
                k_init //= 2
        self.forward = np.ones(self.batch) * config.step
        return self.x_best, self.forward, targets

    def refine(self, update_area, k, split_level):
        if config.update_area == "superpixel" and config.channel_wise:
            assert False, "Not implemented"
        elif config.update_area == "superpixel":
            assert False, "Not implemented"
        elif config.update_area == "split_square" and config.channel_wise:
            assert False, "Not implemented"
        elif config.update_area == "split_square":
            assert False, "Not implemented"
        elif config.update_area == "saliency_map" and config.channel_wise:
            assert False, "Not implemented"
        elif config.update_area == "saliency_map":
            assert False, "Not implemented"
