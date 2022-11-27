import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from skimage.segmentation import slic
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class ImprovedLocalSearchProposedMethod(Attacker):
    def __init__(self):
        assert type(config.steps) == int
        config.n_forward = config.steps
        self.criterion = get_criterion()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_images, n_chanel = x_all.shape[:2]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for b in range(n_batch):
            start = b * self.model.batch_size
            end = min((b + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            batch = np.arange(x.shape[0])
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()

            # calculate various roughness superpixel
            with ThreadPoolExecutor(config.thread) as executor:
                futures = [
                    executor.submit(self.cal_superpixel, x[idx], idx, batch.max() + 1)
                    for idx in batch
                ]
            superpixel_storage = [future.result() for future in futures]
            superpixel_storage = np.array(superpixel_storage)
            level = np.zeros_like(batch)
            superpixel = superpixel_storage[batch, level]
            n_superpixel = superpixel.max(axis=(1, 2))

            # initialize
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            x_best = lower.clone()
            pred = self.model(x_best).softmax(1)
            best_loss = self.criterion(pred, y)
            forward = 1

            targets = []
            for idx in batch:
                chanel = np.tile(np.arange(n_chanel), n_superpixel[idx])
                labels = np.repeat(range(1, n_superpixel[idx] + 1), n_chanel)
                _target = np.stack([chanel, labels], axis=1)
                np.random.shuffle(_target)
                targets.append(_target)
            checkpoint = config.checkpoint * n_superpixel
            pre_checkpoint = np.zeros_like(batch)

            # local search
            searched = [[] for _ in batch]
            loss_storage = []
            while True:
                is_upper = is_upper_best.clone()
                for idx in batch:
                    if forward == checkpoint[idx]:
                        level[idx] = min(level[idx] + 1, len(config.segments) - 1)
                        superpixel[idx] = superpixel_storage[idx, level[idx]]
                        n_superpixel[idx] = superpixel[idx].max()
                        chanel = np.tile(np.arange(n_chanel), n_superpixel[idx])
                        labels = np.repeat(range(1, n_superpixel[idx] + 1), n_chanel)
                        targets[idx] = np.stack([chanel, labels], axis=1)
                        np.random.shuffle(targets[idx])
                        pre_checkpoint[idx] = checkpoint[idx] - 1
                        checkpoint[idx] += config.checkpoint * n_superpixel[idx]
                        searched[idx] = []
                    if targets[idx].shape[0] == 0:
                        target_order = np.array(loss_storage)
                        target_order = target_order[pre_checkpoint[idx] :, idx]
                        target_order = np.argsort(target_order)[::-1]
                        assert target_order.shape[0] == np.array(searched[idx]).shape[0]
                        targets[idx] = np.array(searched[idx])[target_order]
                        searched[idx] = []
                    c, label = targets[idx][0]
                    searched[idx].append((c, label))
                    targets[idx] = np.delete(targets[idx], 0, axis=0)
                    is_upper[idx, c, superpixel[idx] == label] = ~is_upper[
                        idx, c, superpixel[idx] == label
                    ]
                x_adv = torch.where(is_upper, upper, lower)
                pred = self.model(x_adv).softmax(dim=1)
                loss = self.criterion(pred, y)
                loss_storage.append(loss.cpu().numpy())
                update = loss >= best_loss
                x_best[update] = x_adv[update]
                best_loss[update] = loss[update]
                is_upper_best[update] = is_upper[update]
                forward += 1

                pbar.debug(forward, config.steps, "forward", f"batch: {b}")
                if forward == config.steps:
                    break

            x_adv_all.append(x_best)
        x_adv_all = torch.concat(x_adv_all)
        return x_adv_all

    def cal_superpixel(self, x, idx, total):
        pbar.debug(idx + 1, total, "cal_superpixel")
        superpixel_storage = []
        for n_segments in config.segments:
            img = (x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            superpixel = slic(img, n_segments=n_segments)
            superpixel_storage.append(superpixel)
        return superpixel_storage
