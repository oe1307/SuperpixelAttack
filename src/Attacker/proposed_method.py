import itertools as it
import math
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
        n_images = x_all.shape[0]
        n_chanel = x_all.shape[1]
        chanel = np.arange(n_chanel)
        n_batch = math.ceil(n_images / self.model.batch_size)
        for b in range(n_batch):
            start = b * self.model.batch_size
            end = min((b + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            batch = np.arange(x.shape[0])
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()
            pred = F.softmax(self.model(x), dim=1)
            base_loss = self.criterion(pred, y)
            forward = np.ones_like(y.cpu())

            # calculate various roughness superpixel
            # TODO: cpuの並列処理
            superpixel_storage = []
            for idx in batch:
                pbar(idx + 1, batch.max() + 1, "superpixel", f"batch: {b}")
                _superpixel_storage = []
                for idx, n_segments in enumerate(config.segments):
                    img = (x[idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(
                        np.uint8
                    )
                    superpixel = slic(img, n_segments=n_segments)
                    _superpixel_storage.append(superpixel)
                superpixel_storage.append(_superpixel_storage)
            superpixel_storage = np.array(superpixel_storage)

            # initialize
            superpixel = superpixel_storage[:, 0]
            n_superpixel = superpixel.max(axis=(1, 2))
            candidate = list(it.product(chanel, range(1, n_superpixel.max() + 1)))

            # search upper
            upper_loss = []
            for c, label in candidate:
                x_adv = x.permute(1, 0, 2, 3).clone()
                _upper = upper.permute(1, 0, 2, 3).clone()
                x_adv[c, superpixel == label] = _upper[c, superpixel == label]
                pred = F.softmax(self.model(x_adv.permute(1, 0, 2, 3)), dim=1)
                upper_loss.append(self.criterion(pred, y).clone())
            upper_loss = torch.stack(upper_loss, dim=1)
            forward += n_chanel * n_superpixel

            # search lower
            lower_loss = []
            for c, label in candidate:
                x_adv = x.permute(1, 0, 2, 3).clone()
                _lower = lower.permute(1, 0, 2, 3).clone()
                x_adv[c, superpixel == label] = _lower[c, superpixel == label]
                pred = F.softmax(self.model(x_adv.permute(1, 0, 2, 3)), dim=1)
                lower_loss.append(self.criterion(pred, y).clone())
            lower_loss = torch.stack(lower_loss, dim=1)
            forward += n_chanel * n_superpixel

            # make attention map
            loss, u_is_better = torch.stack([lower_loss, upper_loss]).max(dim=0)
            rise = loss - base_loss.unsqueeze(1)
            attention_map = []
            for idx in batch:
                _loss = rise[idx, : n_superpixel[idx] * n_chanel]
                u = u_is_better[idx, : n_superpixel[idx] * n_chanel]
                _candidate = it.product(chanel, range(1, n_superpixel[idx] + 1))
                attention_map.append(
                    [
                        [0, c, label, u.item(), _loss.item()]
                        for (c, label), u, _loss in zip(
                            _candidate, u_is_better[idx], rise[idx]
                        )
                    ]
                )

            # give init x_adv
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            for idx, _attention_map in enumerate(attention_map):
                for _, c, label, u, _ in _attention_map:
                    is_upper_best[idx, c, superpixel[idx] == label] = u
            x_best = torch.where(is_upper_best, upper, lower)
            pred = F.softmax(self.model(x_best), dim=1)
            best_loss = self.criterion(pred, y).clone()
            forward += 1

            while True:
                # sort attention map
                for _attention_map in attention_map:
                    _attention_map.sort(key=lambda k: k[4], reverse=True)

                # divide most attention area
                target, attention = [], []
                for idx, _attention_map in enumerate(attention_map):
                    level, c, label, u, _ = attention_map[idx].pop(0)
                    attention.append([level, c, u])
                    superpixel = superpixel_storage[idx, level]
                    next_level = min(level + 1, len(config.segments) - 1)
                    next_superpixel = superpixel_storage[idx, next_level]
                    n_superpixel = next_superpixel.max()
                    _target = []
                    for target_label in range(1, n_superpixel + 1):
                        target_pixel = next_superpixel == target_label
                        intersection = np.logical_and(superpixel == label, target_pixel)
                        if intersection.sum() >= target_pixel.sum() / 2:
                            _target.append(target_label)
                    target.append(_target)
                n_target = np.array([len(_target) for _target in target])
                for idx, _target in enumerate(target):  # for batch processing
                    target[idx] += [-1] * (n_target.max() - len(_target))
                target, attention = np.array(target), np.array(attention)

                # search target
                next_level = (attention[:, 0] + 1).clip(0, len(config.segments) - 1)
                next_superpixel = superpixel_storage[batch, next_level]
                is_upper, loss = [], []
                for t in range(n_target.max()):
                    _is_upper = is_upper_best.clone()
                    c = attention[:, 1]
                    t_pixel = next_superpixel == target[:, t].reshape(-1, 1, 1)
                    for idx in batch:
                        _is_upper[idx, c[idx], t_pixel[idx]] = not attention[idx, 2]
                    is_upper.append(_is_upper)
                    x_adv = torch.where(_is_upper, upper, lower)
                    pred = F.softmax(self.model(x_adv), dim=1)
                    loss.append(self.criterion(pred, y))
                is_upper, loss = torch.stack(is_upper, dim=0), torch.stack(loss, dim=1)
                _best_loss, best_target = loss.max(dim=1)
                forward += (target != -1).sum(axis=1)

                # update one superpixel
                update = _best_loss > best_loss
                n_target = torch.from_numpy(n_target).to(config.device)
                update = torch.logical_and(update, best_target < n_target)
                best_loss = torch.where(update, _best_loss, best_loss)
                _is_upper_best = is_upper[best_target, batch]
                _update = update.view(-1, 1, 1, 1)
                is_upper_best = torch.where(_update, _is_upper_best, is_upper_best)
                x_best = torch.where(is_upper_best, upper, lower)
                pbar(forward.min(), config.step, "forward", f"batch: {b}")
                if forward.min() >= config.step:
                    break

                # updated multi superpixel
                rise = loss - base_loss.unsqueeze(1)
                search_multi_superpixel = (rise > 0).sum(dim=1) > 1
                is_upper = is_upper_best.clone()
                for idx, (_target, _loss) in enumerate(zip(target, loss)):
                    c = attention[idx, 1]
                    for label, L in zip(_target, _loss):
                        next_pixel = next_superpixel[idx]
                        u = not attention[idx, 2] if L > 0 else attention[idx, 2]
                        is_upper[idx, c, next_pixel == label] = u
                x_adv = torch.where(is_upper, upper, lower)
                pred = F.softmax(self.model(x_adv), dim=1)
                loss = self.criterion(pred, y)
                forward += search_multi_superpixel.cpu().numpy()
                update = update.to(torch.uint8) + (loss > best_loss * 2)
                best_loss = torch.where(update > 1, loss, best_loss)
                is_upper_best = torch.where(
                    (update > 1).view(-1, 1, 1, 1), is_upper, is_upper_best
                )
                x_best = torch.where(is_upper_best, upper, lower)

                # update attention map
                for idx in batch:
                    level = next_level[idx]
                    c = attention[idx, 1]
                    u = attention[idx, 2]
                    _rise = rise[idx, : n_target[idx]].cpu().numpy()
                    _target = target[: n_target[idx]]
                    if update[idx] == 0:  # not updated
                        for r, label in zip(_rise, _target):
                            attention_map[idx].append((level, c, label, u, r))
                    elif update[idx] == 1:  # updated one superpixel
                        for target, (r, label) in enumerate(zip(_rise, _target)):
                            is_best_target = target == best_target[idx].item()
                            _u = not u if is_best_target else u
                            attention_map[idx].append((level, c, label, _u, r))
                    else:
                        for r, label in zip(_rise, _target):
                            attention_map[idx].append((level, c, label, not u, r))

                pbar(forward.min(), config.step, "forward", f"batch: {b}")
                if forward.min() >= config.step:
                    break

            x_adv_all.append(x_best)
        x_adv_all = torch.concat(x_adv_all)
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
