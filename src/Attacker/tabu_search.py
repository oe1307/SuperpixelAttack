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


class TabuSearch(Attacker):
    """
    extract next superpixel,
    and search many times next superpixel using tabu search
    """

    def __init__(self):
        assert type(config.steps) == int
        assert config.additional_search in (
            "best",
            "worst",
            "impacter",
            "non_impacter",
            "random",
        )
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
            n_targets = superpixel.max(axis=(1, 2))

            # initialize
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            x_best = lower.clone()
            pred = self.model(x_best).softmax(1)
            best_loss = self.criterion(pred, y)
            forward = 1

            targets = []
            for idx in batch:
                chanel = np.tile(np.arange(n_chanel), n_targets[idx])
                labels = np.repeat(range(1, n_targets[idx] + 1), n_chanel)
                _target = np.stack([chanel, labels], axis=1)
                np.random.shuffle(_target)
                targets.append(_target)
            checkpoint = config.init_checkpoint * n_targets + 1
            pre_checkpoint = np.ones_like(batch)

            # local search
            searched = [[] for _ in batch]
            loss_storage = []
            best_loss_storage = [best_loss.cpu().numpy()]
            while True:
                is_upper = is_upper_best.clone()
                for idx in batch:
                    if forward >= checkpoint[idx]:
                        # extract target pixel
                        _loss = np.array(loss_storage)
                        _loss = _loss[pre_checkpoint[idx] - 1 :, idx]
                        _best_loss = np.array(best_loss_storage)
                        _best_loss = _best_loss[pre_checkpoint[idx] - 1 : -1, idx]
                        diff = _loss - _best_loss
                        if config.attention_pixel == "best":
                            target_order = np.argsort(diff)[::-1]
                        elif config.attention_pixel == "worst":
                            target_order = np.argsort(diff)
                        elif config.attention_pixel == "impacter":
                            target_order = np.argsort(np.abs(diff))[::-1]
                        elif config.attention_pixel == "non_impacter":
                            target_order = np.argsort(np.abs(diff))
                        elif config.attention_pixel == "random":
                            target_order = np.arange(len(diff))
                            np.random.shuffle(target_order)
                        assert target_order.shape[0] == np.array(searched[idx]).shape[0]
                        target_order = target_order[: target_order.shape[0] * 2 // 3]
                        # TODO: doubled searched
                        attention_pixel = np.array(searched[idx])[target_order]

                        level[idx] = min(level[idx] + 1, len(config.segments) - 1)
                        next_superpixel = superpixel_storage[idx, level[idx]].copy()
                        label_pair = np.stack(
                            [superpixel[idx].reshape(-1), next_superpixel.reshape(-1)]
                        )
                        pair, count = np.unique(
                            label_pair.T, axis=0, return_counts=True
                        )
                        targets[idx] = []
                        for c, label in attention_pixel:
                            _pair = pair[pair[:, 0] == label]
                            _count = count[pair[:, 0] == label]
                            _target = _pair[_count >= np.average(_count)]
                            _target[:, 0] = c
                            targets[idx].append(_target)
                        targets[idx] = np.concatenate(targets[idx])
                        np.random.shuffle(targets[idx])
                        n_targets[idx] = targets[idx].shape[0]
                        pre_checkpoint[idx] = checkpoint[idx]
                        checkpoint[idx] += int(n_targets[idx] * config.checkpoint)
                        superpixel[idx] = next_superpixel.copy()
                        searched[idx] = []
                    if targets[idx].shape[0] == 0:
                        # decide additional search pixel
                        _loss = np.array(loss_storage)
                        _loss = _loss[pre_checkpoint[idx] - 1 :, idx]
                        _best_loss = np.array(best_loss_storage)
                        _best_loss = _best_loss[pre_checkpoint[idx] - 1 : -1, idx]
                        diff = _loss - _best_loss
                        if config.additional_search == "best":
                            target_order = np.argsort(diff)[::-1]
                        elif config.additional_search == "worst":
                            target_order = np.argsort(diff)
                        elif config.additional_search == "impacter":
                            target_order = np.argsort(np.abs(diff))[::-1]
                        elif config.additional_search == "non_impacter":
                            target_order = np.argsort(np.abs(diff))
                        elif config.additional_search == "random":
                            target_order = np.arange(len(diff))
                            np.random.shuffle(target_order)
                        assert target_order.shape[0] == np.array(searched[idx]).shape[0]
                        targets[idx] = np.array(searched[idx])[target_order]
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
                best_loss_storage.append(best_loss.cpu().numpy())
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
