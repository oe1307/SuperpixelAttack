import itertools as it
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from skimage.segmentation import find_boundaries, mark_boundaries, slic
from torch import Tensor

from base import Attacker, get_criterion
from utils import change_level, config_parser, pbar, setup_logger

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
            forward = np.zeros_like(batch)

            # calculate various roughness superpixel
            with ThreadPoolExecutor(config.thread) as executor:
                futures = [
                    executor.submit(self.cal_superpixel, x[idx]) for idx in batch
                ]
            superpixel_storage = [future.result() for future in futures]
            superpixel_storage = np.array(superpixel_storage)

            # initialize
            superpixel_level = 0
            superpixel = superpixel_storage[:, superpixel_level]
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
            u_is_better = torch.stack([lower_loss, upper_loss]).argmax(dim=0).to(bool)
            del upper_loss, lower_loss
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            for idx in batch:
                for c, label, u in zip(chanel, labels, u_is_better[idx]):
                    is_upper_best[idx, c, superpixel[idx] == label] = u
            x_best = torch.where(is_upper_best, upper, lower)
            pred = self.model(x_best).softmax(dim=1)
            best_loss = self.criterion(pred, y).clone()
            forward += 1

            while True:
                # list up boundary box
                boundary_boxes, idx_storage = [], []
                for idx in batch:
                    candidate = it.combinations(range(1, n_superpixel[idx] + 1), 2)
                    for label1, label2 in candidate:
                        boundary = np.logical_and(
                            find_boundaries(superpixel[idx] == label1),
                            find_boundaries(superpixel[idx] == label2),
                        )
                        if boundary.sum() > 0 and forward[idx] < config.n_forward:
                            rows = np.repeat(np.any(boundary, axis=1), w)
                            rows = rows.reshape(x.shape[2:])
                            cols = np.tile(np.any(boundary, axis=0), h)
                            cols = cols.reshape(x.shape[2:])
                            boundary_boxes.append(np.logical_and(rows, cols))
                            idx_storage.append(idx)
                            forward[idx] += 1
                boundary_boxes = np.array(boundary_boxes)
                boundary_boxes = torch.from_numpy(boundary_boxes).to(config.device)
                idx_storage = np.array(idx_storage)

                n_batch = math.ceil(idx_storage.shape[0] / self.model.batch_size)
                loss_storage = []
                for c in range(n_chanel):
                    _loss_storage = []
                    for b in range(n_batch):
                        start = b * self.model.batch_size
                        end = min((b + 1) * self.model.batch_size, idx_storage.shape[0])
                        boundary_box = boundary_boxes[start:end]
                        idx = idx_storage[start:end]
                        is_upper = is_upper_best[idx].clone()
                        is_upper[:, c] = torch.where(
                            boundary_box, ~is_upper[:, c], is_upper[:, c]
                        )
                        x_adv = torch.where(is_upper, upper[idx], lower[idx])
                        pred = self.model(x_adv).softmax(dim=1)
                        _loss_storage.append(self.criterion(pred, y[idx]))
                    loss_storage.append(torch.cat(_loss_storage, dim=0))
                loss_storage = torch.stack(loss_storage, dim=1)

                # update one boundary box
                rise = []
                for idx in batch:
                    if (idx_storage == idx).sum() == 0:
                        rise.append(np.zeros(1))
                        continue
                    search = np.where(idx_storage == idx)[0]
                    loss = loss_storage[search]
                    boundary_box = boundary_boxes[search]
                    rise.append(loss > best_loss[idx])
                    _best_loss, box_id = loss.max(dim=0)
                    _best_loss, c = _best_loss.max(dim=0)
                    best_boundary_box = boundary_box[box_id[c]]
                    if _best_loss > best_loss[idx]:
                        is_upper_best[idx] = torch.where(
                            best_boundary_box,
                            ~is_upper_best[idx, c],
                            is_upper_best[idx, c],
                        )
                    best_loss[idx] = torch.max(best_loss[idx], _best_loss)
                x_best = torch.where(is_upper_best, upper, lower)
                assert x_best.shape == x.shape
                pbar.debug(forward.min(), config.step, "forward", f"batch: {b}")
                if forward.min() >= config.step:
                    break

                # updated multi boundary box
                for idx, _rise in enumerate(rise):
                    if _rise.sum().item() > 1:
                        update = np.stack(np.where(_rise.cpu().numpy())).T
                        search = np.where(idx_storage == idx)[0]
                        boundary_box = boundary_boxes[search]
                        assert search.shape[0] == rise[idx].shape[0]
                        for box_id, c in update:
                            try:
                                is_upper[idx, c] = torch.where(
                                    boundary_box[box_id],
                                    ~is_upper[idx, c],
                                    is_upper[idx, c],
                                )
                            except:
                                breakpoint()
                        forward[idx] += 1

                pbar.debug(forward.min(), config.step, "forward", f"batch: {b}")
                if forward.min() >= config.step:
                    break

                superpixel_level = min(superpixel_level + 1, len(config.segments) - 1)
                superpixel = superpixel_storage[:, superpixel_level]
                logger.debug(f"{superpixel_level = }")

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
