import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from skimage.segmentation import mark_boundaries, slic
from torch import Tensor

from base import Attacker, get_criterion
from utils import change_level, config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class TabuInitializedProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.step
        self.criterion = get_criterion()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_images = x_all.shape[0]
        n_chanel = x_all.shape[1]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for b in range(n_batch):
            start = b * self.model.batch_size
            end = min((b + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            batch = np.arange(x.shape[0])
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()
            forward = np.zeros_like(batch)

            # calculate various roughness superpixel
            with ThreadPoolExecutor(config.thread) as executor:
                futures = [
                    executor.submit(self.cal_superpixel, x[idx]) for idx in batch
                ]
            superpixel_storage = [future.result() for future in futures]
            superpixel_storage = np.array(superpixel_storage)

            # initialize
            superpixel_level = np.zeros_like(batch)
            superpixel = superpixel_storage[batch, superpixel_level]
            n_superpixel = superpixel.max(axis=(1, 2))
            chanel = np.tile(np.arange(n_chanel), n_superpixel.max())
            labels = np.repeat(range(1, n_superpixel.max() + 1), n_chanel)

            # search upper
            upper_loss = []
            for c, label in zip(chanel, labels):
                x_adv = x.permute(1, 0, 2, 3).clone()
                _upper = upper.permute(1, 0, 2, 3).clone()
                x_adv[c, superpixel == label] = _upper[c, superpixel == label]
                pred = self.model(x_adv.permute(1, 0, 2, 3)).softmax(dim=1)
                upper_loss.append(self.criterion(pred, y).clone())
            upper_loss = torch.stack(upper_loss, dim=1)
            forward += n_chanel * n_superpixel

            # search lower
            lower_loss = []
            for c, label in zip(chanel, labels):
                x_adv = x.permute(1, 0, 2, 3).clone()
                _lower = lower.permute(1, 0, 2, 3).clone()
                x_adv[c, superpixel == label] = _lower[c, superpixel == label]
                pred = self.model(x_adv.permute(1, 0, 2, 3)).softmax(dim=1)
                lower_loss.append(self.criterion(pred, y).clone())
            lower_loss = torch.stack(lower_loss, dim=1)
            forward += n_chanel * n_superpixel

            # give init x_adv
            loss, u_is_better = torch.stack([lower_loss, upper_loss]).argmax(dim=0)
            u_is_better = u_is_better.to(torch.bool)
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            for idx in batch:
                for c, label, u in zip(chanel, labels, u_is_better[idx]):
                    is_upper_best[idx, c, superpixel[idx] == label] = u
            x_best = torch.where(is_upper_best, upper, lower)
            pred = self.model(x_best).softmax(dim=1)
            best_loss = self.criterion(pred, y).clone()
            forward += 1
            tabu_list = np.zeros((batch.shape[0], n_chanel, labels.shape[0]))
            assert forward.max() < config.checkpoints[0]

            while True:
                if sum([(forward == c).sum() for c in config.checkpoints]) > 0:
                    for c in config.checkpoints:
                        superpixel_level += forward == c
                    superpixel = superpixel_storage[batch, superpixel_level]
                    n_superpixel = superpixel.max(axis=(1, 2))

                # fast fit
                is_upper = []
                for idx in batch:
                    _is_upper = is_upper_best[idx].clone()
                    _best_loss = -10
                    for c, label in zip(chanel, labels):
                        if tabu_list[idx, c, label - 1] > 0:
                            tabu_list[idx, c, label - 1] -= 1
                            continue

                pbar.debug(forward.min(), config.step, "forward", f"batch: {b}")
                if forward.min() >= config.step:
                    break

            x_adv_all.append(x_best)
        x_adv_all = torch.concat(x_adv_all)
        return x_adv_all

    def cal_superpixel(self, x):
        superpixel_storage = []
        for n_segments in config.segments:
            img = (x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            superpixel = slic(img, n_segments=n_segments)
            superpixel_storage.append(superpixel)
        return superpixel_storage
