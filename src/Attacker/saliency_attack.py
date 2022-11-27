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
        x_adv_all = []
        n_images, n_chanel, height, width = x_all.shape
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            self.x_adv = x_all[start:end].clone()
            self.y = y_all[start:end]
            self.upper = (self.x_adv + config.epsilon).clamp(0, 1).clone()
            self.lower = (self.x_adv - config.epsilon).clamp(0, 1).clone()
            self.saliency_map = self.saliency_model(self.x_adv)[0].round()
            self.saliency_map = self.saliency_map.bool()

            k_init = config.k_init
            split_level = 1
            block = np.array([(c, 0, height, 0, width) for c in range(n_chanel)])
            self.forward = np.zeros(self.x_adv.shape[0])

            # main loop
            while True:
                self.refine(block, k_init, split_level)
                if self.forward >= config.steps:
                    break
                elif k_init > 1:
                    assert k_init % 2 == 0
                    k_init //= 2
            x_adv_all.append(self.x_adv)
        x_adv_all = torch.concat(x_adv_all)
        return x_adv_all

    def refine(self, search_block, k, split_level):
        forward = self.forward.copy()
        if split_level == 1:

            split_blocks = []
            for block in search_block:
                split_blocks.append(self.split(block, k))
            split_blocks = np.concatenate(split_blocks)
            np.random.shuffle(split_blocks)

            upper_loss, lower_loss = [], []
            for block in split_blocks:
                _block = torch.zeros_like(self.x_adv, dtype=torch.bool)
                _block[:, block[0], block[1] : block[2], block[3] : block[4]] = True
                _block = _block & self.saliency_map
                x_adv = torch.where(_block, self.upper, self.x_adv)
                pred = self.model(x_adv).softmax(dim=0)
                upper_loss.append(self.criterion(pred, self.y))
                forward += (_block.sum(dim=(1, 2, 3)) != 0).cpu().numpy()
                x_adv = torch.where(_block, self.lower, self.x_adv)
                pred = self.model(x_adv).softmax(dim=0)
                lower_loss.append(self.criterion(pred, self.y))
                forward += (_block.sum(dim=(1, 2, 3)) != 0).cpu().numpy()
                if forward.min() >= config.steps:
                    break
            upper_loss, lower_loss = torch.stack(upper_loss), torch.stack(lower_loss)
            loss, u_is_better = torch.stack([lower_loss, upper_loss]).max(dim=0)
            indices = loss.argsort(dim=0, descending=True)
            for index in indices:
                breakpoint()
        else:
            split_blocks = self.split(search_block, k)
            for block in split_blocks:
                _block = torch.zeros_like(self.x_adv, dtype=torch.bool)
                _block[:, block[0], block[1] : block[2], block[3] : block[4]] = True
                _block = _block & self.saliency_map
                x_adv = torch.where(_block, self.upper, self.x_adv)
                pred = self.model(x_adv).softmax(dim=1)
                upper_loss.append(self.criterion(pred, self.y))
                forward += (_block.sum(dim=(1, 2, 3)) != 0).cpu().numpy()
                x_adv = torch.where(_block, self.lower, self.x_adv)
                pred = self.model(x_adv).softmax(dim=1)
                lower_loss.append(self.criterion(pred, self.y))
                forward += (_block.sum(dim=(1, 2, 3)) != 0).cpu().numpy()
                if forward.min() >= config.steps:
                    break
            upper_loss, lower_loss = torch.stack(upper_loss), torch.stack(lower_loss)
            loss, u_is_better = torch.stack([lower_loss, upper_loss]).max(dim=0)
            indices = loss.argsort(descending=True)


    def split(self, block: np.ndarray, k):
        n_blocks = ((block[2] - block[1]) // k, (block[4] - block[3]) // k)
        x1 = np.linspace(block[1], block[2] - k, n_blocks[0], dtype=int)
        x2 = np.linspace(block[1] + k, block[2], n_blocks[0], dtype=int)
        x = np.stack([np.repeat(x1, n_blocks[1]), np.repeat(x2, n_blocks[1])]).T
        y1 = np.linspace(block[3], block[4] - k, n_blocks[1], dtype=int)
        y2 = np.linspace(block[3] + k, block[4], n_blocks[1], dtype=int)
        y = np.stack([np.tile(y1, n_blocks[0]), np.tile(y2, n_blocks[0])]).T
        c = np.repeat(block[0], n_blocks[0] * n_blocks[1])
        split_block = np.stack([c, x[:, 0], x[:, 1], y[:, 0], y[:, 1]], axis=1)
        np.random.shuffle(split_block)
        return split_block
