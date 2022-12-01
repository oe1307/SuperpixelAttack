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


class AdvancedLocalSearch(Attacker):
    """
    extract next superpixel,
    but search many times next superpixel
    """

    def __init__(self):
        self.check_param()
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
            superpixel_storage = np.array([future.result() for future in futures])
            level = np.zeros_like(batch)
            superpixel = superpixel_storage[batch, level]

            # initialize
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            x_best = lower.clone()
            pred = self.model(x_best).softmax(1)
            best_loss = self.criterion(pred, y)

            targets = []
            n_superpixel = superpixel.max(axis=(1, 2))
            for idx in batch:
                chanel = np.tile(np.arange(n_chanel), n_superpixel[idx])
                labels = np.repeat(range(1, n_superpixel[idx] + 1), n_chanel)
                _target = np.stack([chanel, labels], axis=1)
                np.random.shuffle(_target)
                targets.append(_target)
            n_targets = np.array([len(target) for target in targets])
            checkpoint = (config.init_checkpoint * n_targets + 1).round().astype(int)
            pre_checkpoint = np.ones_like(batch)

            # local search
            searched = [[] for _ in batch]
            loss_storage = []
            best_loss_storage = [best_loss.cpu().numpy()]
            for forward in range(1, config.steps + 1):
                is_upper = is_upper_best.clone()
                for idx in batch:
                    if forward >= checkpoint[idx]:
                        # decide attention pixel
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
                        ratio = int(target_order.shape[0] * config.ratio) + 1
                        attention_pixel = np.array(searched[idx])[target_order[:ratio]]
                        # TODO: doubled searched

                        # extract target pixel
                        level[idx] = min(level[idx] + 1, len(config.segments) - 1)
                        pre_superpixel = superpixel[idx].copy()
                        superpixel[idx] = superpixel_storage[idx, level[idx]]
                        label_pair = np.stack(
                            [pre_superpixel.reshape(-1), superpixel[idx].reshape(-1)]
                        ).T
                        pair, count = np.unique(label_pair, axis=0, return_counts=True)
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
                        elif config.additional_search == "old":
                            target_order = np.arange(len(diff))
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
                pbar.debug(forward, config.steps, "forward", f"batch: {b}")

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

    def check_param(self):
        assert type(config.steps) == int
        assert config.attention_pixel in (
            "best",
            "worst",
            "impacter",
            "non_impacter",
            "random",
        )
        assert config.additional_search in (
            "best",
            "worst",
            "impacter",
            "non_impacter",
            "random",
            "old",
        )
        assert type(config.init_checkpoint) == float
        assert type(config.checkpoint) == float