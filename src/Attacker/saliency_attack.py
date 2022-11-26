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
        saliency_map_all = []
        n_images, n_chanel, height, width = x_all.shape
        n_batch = math.ceil(n_images / config.saliency_batch_size)
        for i in range(n_batch):
            pbar.debug(i + 1, n_batch, "saliency map")
            start = i * config.saliency_batch_size
            end = min((i + 1) * config.saliency_batch_size, n_images)
            x = x_all[start:end]
            saliency_map = self.saliency_model(x)[0].round()
            saliency_map_all.append(saliency_map)
        saliency_map_all = torch.concat(saliency_map_all)

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
            saliency_map = saliency_map_all[start:end].to(torch.bool)
            pred = self.model(x).softmax(dim=1)
            loss = self.criterion(pred, y)
            forward = np.ones(x.shape[0])

            # initialize
            init_block = (n_images, n_chanel, height // split, width // split)
            is_upper = torch.zeros(init_block, dtype=torch.bool, device=config.device)
            _h = np.repeat(np.arange(init_block[2]), init_block[3])
            _w = np.tile(np.arange(init_block[3]), init_block[2])
            targets = np.stack([_h, _w], axis=1)
            c = np.tile(np.arange(n_chanel), targets.shape[0])
            targets = np.repeat(targets, n_chanel, axis=0)
            targets = np.stack([c, targets[:, 0], targets[:, 1]], axis=1)
            for c, h, w in targets:
                _is_upper = is_upper.clone()
                assert (~(_is_upper[:, c, h, w])).all().item()
                _is_upper[:, c, h, w] = True
                _is_upper = _is_upper.repeat_interleave(split, dim=2)
                _is_upper = _is_upper.repeat_interleave(split, dim=3)
                x_adv = torch.where(_is_upper, upper, lower)
                x_adv = torch.where(saliency_map, x_adv, x)
                pred = self.model(x_adv).softmax(dim=1)
                _loss = self.criterion(pred, y)
                update = _loss > loss
                is_upper[update] = _is_upper[update]
                loss[update] = _loss[update]
            from matplotlib import pyplot as plt

            plt.imshow(saliency_map[0, 0].cpu().numpy().astype(np.uint8))
            plt.show()
            plt.savefig("test.png")
            plt.imshow(x_adv[0].cpu().numpy().transpose(1, 2, 0))
            plt.savefig("test2.png")

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
