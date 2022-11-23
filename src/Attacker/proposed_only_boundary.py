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
            forward = n_chanel * n_superpixel

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

            u_is_better = torch.stack([lower_loss, upper_loss]).argmax(dim=0).to(bool)
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            for idx in batch:
                for c, label, u in zip(chanel, labels, u_is_better[idx]):
                    is_upper_best[idx, c, superpixel[idx] == label] = u
            x_best = torch.where(is_upper_best, upper, lower)
            pred = self.model(x_best).softmax(dim=1)
            best_loss = self.criterion(pred, y).clone()
            forward += 1

            boundary_boxes = [[] for _ in batch]
            for idx in batch:
                candidate = it.combinations(range(1, n_superpixel[idx] + 1), 2)
                for label1, label2 in candidate:
                    boundary = np.logical_and(
                        find_boundaries(superpixel[idx] == label1),
                        find_boundaries(superpixel[idx] == label2),
                    )
                    rows = np.repeat(np.any(boundary, axis=1), w)
                    rows = rows.reshape(x.shape[2:])
                    cols = np.tile(np.any(boundary, axis=0), h)
                    cols = cols.reshape(x.shape[2:])
                    boundary_box = np.logical_and(rows, cols)
                    if boundary_box.sum() > 0:
                        boundary_boxes[idx].append(boundary_box)
            n_boundary = np.array([len(boundary) for boundary in boundary_boxes])

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
                    if forward[idx] == checkpoint[idx] or boundary_boxes[idx] == []:
                        superpixel_level[idx] += 1
                        superpixel[idx] = superpixel_storage[idx, superpixel_level[idx]]
                        n_superpixel[idx] = superpixel[idx].max()
                        candidate = it.combinations(range(1, n_superpixel[idx] + 1), 2)
                        boundary_boxes[idx] = []
                        for label1, label2 in candidate:
                            boundary = np.logical_and(
                                find_boundaries(superpixel[idx] == label1),
                                find_boundaries(superpixel[idx] == label2),
                            )
                            rows = np.repeat(np.any(boundary, axis=1), w)
                            rows = rows.reshape(x.shape[2:])
                            cols = np.tile(np.any(boundary, axis=0), h)
                            cols = cols.reshape(x.shape[2:])
                            boundary_box = np.logical_and(rows, cols)
                            if boundary_box.sum() > 0:
                                boundary_boxes[idx].append(boundary_box)
                        n_boundary[idx] = len(boundary_boxes[idx])
                        chanel = np.tile(np.arange(n_chanel), n_boundary[idx])
                        ids = np.repeat(range(n_boundary[idx]), n_chanel)
                        target = np.stack([chanel, ids], axis=1)
                        np.random.shuffle(target)
                        targets[idx] = target
                        checkpoint[idx] += 3 * n_boundary[idx]
                    c, box_id = targets[idx][0]
                    targets[idx] = np.delete(targets[idx], 0, axis=0)
                    is_upper[idx, c, boundary_boxes[idx][box_id]] = ~is_upper[
                        idx, c, boundary_boxes[idx][box_id]
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
