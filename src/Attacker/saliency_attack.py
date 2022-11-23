import math

import torch
from torch import Tensor

from base import Attacker, SODModel, get_criterion
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SaliencyAttack(Attacker):
    def __init__(self):
        super().__init__()
        if config.dataset != "imagenet":
            raise ValueError("Saliency Attack is only for ImageNet")
        self.criterion = get_criterion()
        config.n_forward = config.steps

        # saliency model
        self.saliency_model = SODModel()
        chkpt = torch.load(config.saliency_weight, map_location=config.device)
        self.saliency_model.load_state_dict(chkpt["model"])
        self.saliency_model.to(config.device)
        self.saliency_model.eval()

    def _attack(self, x_all: Tensor, y_all: Tensor):
        x_adv_all = []
        n_images = x_all.shape[0]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            upper = (x + config.epsilon).clamp(0, 1)
            lower = (x - config.epsilon).clamp(0, 1)
        del y, upper, lower, x_adv_all
        raise NotImplementedError
