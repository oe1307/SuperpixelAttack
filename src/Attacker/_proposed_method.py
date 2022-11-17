import itertools as it
import sys

import numpy as np
import torch
from skimage.segmentation import mark_boundaries, slic
from torch import Tensor
from torch.nn import functional as F

from base import Attacker, get_criterion
from utils import change_level, config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class ProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.step
        self.criterion = get_criterion()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_chanel = x_all.shape[1]
        # TODO: batch処理
        for idx, (x, y) in enumerate(zip(x_all, y_all)):
            x = x.to(config.device)
            y = y.to(config.device)
            upper = (x + config.epsilon).clamp(0, 1)
            lower = (x - config.epsilon).clamp(0, 1)
            pred = F.softmax(self.model(x.unsqueeze(0)), dim=1)
            base_loss = self.criterion(pred, y).item()
            forward = 1

            # 複数の荒さのsuperpixelをあらかじめ計算
            superpixel_storage = []
            for i, n_segments in enumerate(config.segments):
                pbar(i + 1, len(config.segments), "superpixel")
                img = (x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                superpixel = slic(img, n_segments=n_segments)
                superpixel_storage.append(superpixel)

            # initialize
            superpixel_level = 0
            superpixel = superpixel_storage[superpixel_level]
            n_superpixel = superpixel.max()
            _y = y.repeat(n_chanel * n_superpixel).clone()
            candidate = list(it.product(range(n_chanel), range(1, n_superpixel + 1)))

            # search upper
            x_adv = x.repeat(n_chanel * n_superpixel, 1, 1, 1)
            for n, (c, label) in enumerate(candidate):
                x_adv[n, c, superpixel == label] = upper[c, superpixel == label]
            pred = F.softmax(self.model(x_adv), dim=1)
            upper_loss = self.criterion(pred, _y)
            forward += len(candidate)

            # search lower
            x_adv = x.repeat(n_chanel * n_superpixel, 1, 1, 1)
            for n, (c, label) in enumerate(candidate):
                x_adv[n, c, superpixel == label] = lower[c, superpixel == label]
            pred = F.softmax(self.model(x_adv), dim=1)
            lower_loss = self.criterion(pred, _y)
            forward += len(candidate)

            # make attention map
            loss, u_is_better = torch.stack([lower_loss, upper_loss]).max(dim=0)
            loss = loss - base_loss
            u_is_better = u_is_better.bool()
            attention_map = [
                (superpixel_level, c, label, u.item(), _loss.item())
                for (c, label), u, _loss in zip(candidate, u_is_better, loss)
            ]

            # give init x_adv
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            for (c, label), u in zip(candidate, u_is_better):
                is_upper_best[c, superpixel == label] = u
            x_best = torch.where(is_upper_best, upper, lower)
            pred = F.softmax(self.model(x_best.unsqueeze(0)), dim=1)
            best_loss = self.criterion(pred, y).item()
            forward += 1
            # self.visualize(superpixel, x_best)  # 可視化

            while True:
                attention_map.sort(key=lambda k: k[4], reverse=True)

                # divide most attention area
                superpixel_level, c, label, u, _ = attention_map.pop(0)
                superpixel = superpixel_storage[superpixel_level]
                attention = superpixel == label
                next_level = min(superpixel_level + 1, len(config.segments) - 1)
                next_superpixel = superpixel_storage[next_level]
                n_superpixel = next_superpixel.max()
                target = []
                for target_label in range(1, n_superpixel + 1):
                    target_pixel = next_superpixel == target_label
                    intersection = np.logical_and(attention, target_pixel)
                    if intersection.sum() >= target_pixel.sum() / 2:
                        target.append(target_label)

                # search target
                if len(target) == 0:
                    logger.warning("\ntarget is empty")
                    continue
                is_upper = is_upper_best.repeat(len(target), 1, 1, 1)
                for n, target_label in enumerate(target):
                    is_upper[n, c, next_superpixel == target_label] = not u
                x_adv = torch.where(is_upper, upper, lower)
                pred = F.softmax(self.model(x_adv), dim=1)
                _y = y.repeat(len(target)).clone()
                loss = self.criterion(pred, _y)
                forward += len(target)
                _best_loss, _best_idx = loss.max(dim=0)
                _loss = loss - best_loss

                # update one superpixel
                update = 0
                if _best_loss > best_loss:
                    update = 1
                    best_loss = _best_loss.item()
                    is_upper_best = is_upper[_best_idx].clone()
                    x_best = x_adv[_best_idx].clone()
                pbar(forward, config.step, "forward", f"{idx =}")
                if forward >= config.step:
                    break

                # updated multi superpixel
                if (_loss > 0).sum().item() > 1:
                    is_upper = is_upper_best.clone()
                    for target_loss, target_label in zip(_loss, target):
                        if (target_loss > 0).item():
                            is_upper[c, next_superpixel == target_label] = not u
                    x_adv = torch.where(is_upper, upper, lower)
                    pred = F.softmax(self.model(x_adv.unsqueeze(0)), dim=1)
                    loss = self.criterion(pred, y).item()
                    forward += 1
                    if loss > best_loss:
                        update = 2
                        best_loss = loss
                        is_upper_best = is_upper.clone()
                        x_best = x_adv.clone()

                if update == 0:  # not updated
                    for loss, label in zip(_loss, target):
                        attention_map.append((next_level, c, label, u, loss.item()))
                elif update == 1:  # updated one superpixel
                    for i, (loss, label) in enumerate(zip(_loss, target)):
                        if i == _best_idx:
                            attention_map.append(
                                (next_level, c, label, not u, loss.item())
                            )
                        else:
                            attention_map.append((next_level, c, label, u, loss.item()))

                elif update == 2:  # updated multi superpixel
                    for loss, label in zip(_loss, target):
                        if (loss > 0).item():
                            attention_map.append(
                                (next_level, c, label, not u, loss.item())
                            )
                        else:
                            attention_map.append((next_level, c, label, u, loss.item()))

                pbar(forward, config.step, "forward", f"{idx =}")
                if forward >= config.step:
                    break
            x_adv_all.append(x_best)
        x_adv_all = torch.stack(x_adv_all)
        return x_adv_all

    def visualize(self, superpixel: np.ndarray, x: Tensor):
        change_level("matplotlib", 40)
        from matplotlib import pyplot as plt

        assert x.dim() == 3
        x = x.cpu().numpy().transpose(1, 2, 0)

        plt.subplots(figsize=(6, 6))
        plt.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.imshow(mark_boundaries(x, superpixel))
        plt.savefig(f"{config.savedir}/superpixel.png")
        plt.close()

        plt.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.imshow(superpixel)
        plt.colorbar()
        plt.savefig(f"{config.savedir}/label.png")
        plt.close()

        plt.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.imshow(x)
        plt.savefig(f"{config.savedir}/x_best.png")
        plt.close()
        sys.exit(0)
