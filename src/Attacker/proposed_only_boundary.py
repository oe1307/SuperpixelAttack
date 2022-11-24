import itertools as it
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from skimage.segmentation import find_boundaries, slic
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class BoundaryProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.step
        self.criterion = get_criterion()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_images = x_all.shape[0]
        n_chanel = x_all.shape[1]
        h, w = x_all.shape[2:]
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
                    executor.submit(self.cal_superpixel, x[idx]) for idx in batch
                ]
            superpixel_storage = [future.result() for future in futures]
            superpixel_storage = np.array(superpixel_storage)
            superpixel_level = np.zeros_like(batch)
            superpixel = superpixel_storage[batch, superpixel_level]
            n_superpixel = superpixel.max(axis=(1, 2))

            # calculate boundary box
            with ThreadPoolExecutor(config.thread) as executor:
                futures = [
                    executor.submit(self.cal_boundary_box, superpixel_storage[idx])
                    for idx in batch
                ]
            boundary_box_storage = [future.result() for future in futures]
            boundary_box = [box[0] for box in boundary_box_storage]
            n_boundary = np.array([boundary_box[idx].shape[0] for idx in batch])

            # initialize
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            x_best = lower.clone()
            pred = self.model(x_best).softmax(1)
            best_loss = self.criterion(pred, y)
            forward = np.ones_like(batch)

            targets = []
            for idx in batch:
                chanel = np.tile(np.arange(n_chanel), n_superpixel[idx])
                labels = np.repeat(range(1, n_superpixel[idx] + 1), n_chanel)
                target = np.stack([chanel, labels], axis=1)
                np.random.shuffle(target)
                targets.append(target)
            checkpoint = 3 * n_superpixel

            for _ in range(checkpoint.max()):
                is_upper = is_upper_best.clone()
                for idx in batch:
                    if forward[idx] >= checkpoint[idx]:
                        continue
                    c, label = targets[idx][0]
                    targets[idx] = np.delete(targets[idx], 0, axis=0)
                    is_upper[idx, c, superpixel[idx] == label] = ~is_upper[
                        idx, c, superpixel[idx] == label
                    ]
                    forward[idx] += 1
                x_adv = torch.where(is_upper, upper, lower)
                pred = self.model(x_adv).softmax(dim=1)
                loss = self.criterion(pred, y)
                update = loss >= best_loss
                x_best[update] = x_adv[update]
                best_loss[update] = loss[update]
                is_upper_best[update] = is_upper[update]

            targets = []
            for idx in batch:
                chanel = np.tile(np.arange(n_chanel), n_boundary[idx])
                ids = np.repeat(range(n_boundary[idx]), n_chanel)
                target = np.stack([chanel, ids], axis=1)
                np.random.shuffle(target)
                targets.append(target)
            checkpoint = forward + 3 * n_boundary

            # local search
            while True:
                is_upper = is_upper_best.clone()
                for idx in batch:
                    if forward[idx] >= config.n_forward:
                        continue
                    if forward[idx] == checkpoint[idx] or targets[idx].shape[0] == 0:
                        superpixel_level[idx] += 1
                        if superpixel_level[idx] >= len(config.segments):
                            logger.warning("Reach maximum superpixel level")
                            superpixel_level[idx] = len(config.segments) - 1
                        boundary_box[idx] = boundary_box_storage[idx][superpixel_level[idx]]
                        n_boundary[idx] = boundary_box[idx].shape[0]
                        chanel = np.tile(np.arange(n_chanel), n_boundary[idx])
                        ids = np.repeat(range(n_boundary[idx]), n_chanel)
                        target = np.stack([chanel, ids], axis=1)
                        np.random.shuffle(target)
                        targets[idx] = target
                        checkpoint[idx] += 3 * n_boundary[idx]
                    if targets[idx].shape[0] == 0:
                        continue
                    c, box_id = targets[idx][0]
                    targets[idx] = np.delete(targets[idx], 0, axis=0)
                    is_upper[idx, c, boundary_box[idx][box_id]] = ~is_upper[
                        idx, c, boundary_box[idx][box_id]
                    ]
                x_adv = torch.where(is_upper, upper, lower)
                pred = self.model(x_adv).softmax(dim=1)
                loss = self.criterion(pred, y)
                update = loss >= best_loss
                x_best[update] = x_adv[update]
                best_loss[update] = loss[update]
                is_upper_best[update] = is_upper[update]
                forward += 1

                pbar.debug(forward.min(), config.step, "forward", f"batch: {b}")
                if forward.min() == config.step:
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

    def cal_boundary_box(self, superpixel_storage):
        w, h = superpixel_storage.shape[1:]
        boundary_box_storage = []
        for level in range(len(config.segments)):
            box = []
            superpixel = superpixel_storage[level]
            tmp = np.stack([superpixel[1:, 1:], superpixel[:-1, :-1]]).reshape(2, -1)
            tmp = np.unique(np.sort(tmp, axis=0).reshape(2, -1), axis=1)
            tmp = tmp[np.tile((tmp[0] != tmp[1]), 2).reshape(2, -1)]
            candidate = tmp.reshape(2, -1).T
            for label1, label2 in candidate:
                boundary = np.logical_and(
                    find_boundaries(superpixel == label1),
                    find_boundaries(superpixel == label2),
                )
                rows = np.repeat(np.any(boundary, axis=1), w).reshape((h, w))
                cols = np.tile(np.any(boundary, axis=0), h).reshape((h, w))
                boundary_box = np.logical_and(rows, cols)
                box.append(boundary_box)
            boundary_box_storage.append(np.array(box))
        return boundary_box_storage
