import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, SODModel, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SaliencyAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.criterion = get_criterion()
        config.n_forward = config.steps

        # saliency model
        self.saliency_model = SODModel()
        weights = torch.load(config.saliency_weight)
        self.saliency_model.load_state_dict(weights["model"])
        self.saliency_model.to(config.device)
        self.saliency_model.eval()

    def _attack(self, x_all: Tensor, y_all: Tensor):

        # make saliency map
        saliency_map = []
        n_images, c, h, w = x_all.shape
        n_batch = math.ceil(n_images / config.saliency_batch_size)
        for i in range(n_batch):
            pbar.debug(i + 1, n_batch, "saliency map")
            start = i * config.saliency_batch_size
            end = min((i + 1) * config.saliency_batch_size, n_images)
            x = x_all[start:end]
            saliency_map.append(self.saliency_model(x)[0].round()[:, 0])
        saliency_map = torch.cat(saliency_map, dim=0)

        x_adv_all = []
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()
            split = config.initial_split

            # initialize
            init_block = (n_images, c, h // split, w // split)
            is_upper = torch.zeros(init_block, dtype=torch.bool, device=config.device)
            h = np.repeat(np.arange(init_block[2]), init_block[3])
            w = np.tile(np.arange(init_block[3]), init_block[2])
            targets = np.stack([h, w], axis=1)
            breakpoint()

        self.forward = np.ones(n_images)

        while True:
            is_upper, loss, is_upper_best, best_loss = self.refine(
                is_upper, loss, is_upper_best, best_loss
            )
            if self.forward.min() >= config.steps:
                break
            elif self.split > 1:
                is_upper = is_upper.repeat([1, 1, 2, 2])
                if self.split % 2 == 1:
                    logger.critical(f"self.split is not even: {self.split}")
                self.split //= 2

    def cal_loss(self, is_upper_all: Tensor) -> Tensor:
        n_images = is_upper_all.shape[0]
        loss = torch.zeros(n_images, device=config.device)
        num_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(num_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            upper = self.upper[start:end]
            lower = self.lower[start:end]
            is_upper = is_upper_all[start:end].repeat([1, 1, self.split, self.split])
            x_adv = torch.where(is_upper, upper, lower)
            y = self.y[start:end]
            pred = self.model(x_adv).softmax(dim=1)
            loss[start:end] = self.criterion(pred, y)
        return loss
