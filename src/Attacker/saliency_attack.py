import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, SODModel, get_criterion
from utils import config_parser, setup_logger, change_level

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
        weights = torch.load(config.saliency_weight)
        self.saliency_model.load_state_dict(weights["model"])
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
            batch, c, h, w = x.shape
            self.split = config.initial_split
            saliency_map = self.saliency_model(x)[0].round()
            is_upper = torch.zeros(
                (batch, c, h // self.split, w // self.split),
                dtype=bool,
                device=config.device,
            )
            self.forward = torch.zeros(self.batch, device=config.device)

            while True:
                self.refine()
                if self.split > 1:
                    is_upper = is_upper.repeat([1, 1, 2, 2])
                    if self.split % 2 == 1:
                        logger.critical(f"self.split is not even: {self.split}")
                    self.split //= 2

        del y, upper, lower, x_adv_all
        raise NotImplementedError

    def visualize(self, saliency_map: Tensor):
        change_level("matplotlib", 40)
        from matplotlib import pyplot as plt

        plt.subplots(figsize=(6, 6))
        plt.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.imshow(saliency_map[0, 0].cpu().numpy(), cmap="gray")
        plt.savefig("saliency_map.png")
        quit()
