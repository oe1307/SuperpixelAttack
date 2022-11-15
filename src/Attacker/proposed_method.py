import bisect
import itertools as it
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser, setup_logger, pbar, change_level

logger = setup_logger(__name__)
config = config_parser()


class ProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.forward
        self.criterion = get_criterion()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_images = x_all.shape[0]
        n_chanel = x_all.shape[1]
        n_batch = math.ceil(n_images / self.model.batch_size)
        loss_storage = torch.zeros(n_images, config.forward + 1)
        best_loss_storage = torch.zeros(n_images, config.forward + 1)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, config.n_examples)
            x = x_all[start:end].to(config.device)
            y = y_all[start:end].to(config.device)
            upper = (x + config.epsilon).clamp(0, 1)
            lower = (x - config.epsilon).clamp(0, 1)

            # superpixel
            labels, n_labels, slics = [], [], []
            for n, _x in enumerate(x):
                pbar(n + 1, x.shape[0], f"batch {i} superpixel")
                _x = (_x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                converted = cv2.cvtColor(_x, cv2.COLOR_RGB2HSV_FULL)
                slic = cv2.ximgproc.createSuperpixelSLIC(
                    converted, cv2.ximgproc.MSLIC, config.region_size, config.ruler
                )
                slic.iterate(config.num_iterations)
                slic.enforceLabelConnectivity(config.min_element_size)
                slics.append(slic)
                label = slic.getLabels()
                labels.append(label)
                n_label = slic.getNumberOfSuperpixels()
                n_labels.append(n_label)
            # self.superpixel_visualize(x[0], slics[0])  # 可視化

            # initialize
            is_upper_best = []
            for idx in range(x.shape[0]):
                is_upper = torch.zeros_like(x[0], dtype=torch.bool)
                for label in range(n_labels[idx]):
                    # RGB
                    is_upper[0, labels[idx] == label] = torch.randint(
                        2, (1,), dtype=torch.bool
                    )
                    is_upper[1, labels[idx] == label] = torch.randint(
                        2, (1,), dtype=torch.bool
                    )
                    is_upper[2, labels[idx] == label] = torch.randint(
                        2, (1,), dtype=torch.bool
                    )
                is_upper_best.append(is_upper)
            is_upper_best = torch.stack(is_upper_best)
            x_best = torch.where(is_upper_best, upper, lower)
            pred = F.softmax(self.model(x_best), dim=1)
            best_loss = self.criterion(pred, y)
            loss_storage[start:end, 0] = best_loss
            best_loss_storage[start:end, 0] = best_loss
            # self.imshow(labels[0], x_best[0])  # 可視化

            # loop: greedy
            for step in range(config.forward):
                pbar(step + 1, config.forward, f"batch {i} step")
                is_upper = is_upper_best.clone()
                for idx in range(x.shape[0]):
                    candidate = np.array(
                        list(it.product(range(n_chanel), range(n_labels[idx])))
                    )
                    n_flip = int(self.flip_size_manager(step) * len(candidate))
                    flips = np.random.choice(len(candidate), n_flip, replace=False)
                    flips = candidate[flips]
                    for c, label in flips:
                        is_upper[idx, c, labels[idx] == label] = ~is_upper[
                            idx, c, labels[idx] == label
                        ]
                x_adv = torch.where(is_upper, upper, lower)
                pred = F.softmax(self.model(x_adv), dim=1)
                loss = self.criterion(pred, y)
                is_upper_best = torch.where(
                    (loss > best_loss).view(-1, 1, 1, 1), is_upper, is_upper_best
                )
                x_best = torch.where(
                    (loss > best_loss).view(-1, 1, 1, 1), x_adv, x_best
                )
                best_loss = torch.where(loss > best_loss, loss, best_loss)
                loss_storage[start:end, step + 1] = loss
                best_loss_storage[start:end, step + 1] = best_loss
                # self.imshow(labels[0], x_best[0])  # 可視化
            x_adv_all.append(x_best)
        x_adv_all = torch.cat(x_adv_all, dim=0)
        return x_adv_all

    def flip_size_manager(self, step):
        i_p = step / config.forward
        checkpoint = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
        p_ratio = [0.5**i for i in range(len(checkpoint) + 1)]
        i_ratio = bisect.bisect_left(checkpoint, i_p)
        return config.p_init * p_ratio[i_ratio]

    def superpixel_visualize(self, x, slic):
        change_level("matplotlib", 40)
        from matplotlib import pyplot as plt

        superpixel = x.cpu().numpy().transpose(1, 2, 0)
        contour_mask = slic.getLabelContourMask(False)
        superpixel[contour_mask == 255] = (0, 255, 255)
        plt.subplots(figsize=(6, 6))
        plt.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.imshow(superpixel)
        plt.savefig(f"{config.savedir}/superpixel.png")
        plt.close()
        quit()

    def imshow(self, labels: np.ndarray, x: Tensor):
        change_level("matplotlib", 40)
        from matplotlib import pyplot as plt

        plt.imshow(labels)
        plt.savefig(f"{config.savedir}/labels.png")

        assert x.dim() == 3
        plt.imshow(x.cpu().numpy().transpose(1, 2, 0))
        plt.savefig(f"{config.savedir}/x_best.png")
        quit()
