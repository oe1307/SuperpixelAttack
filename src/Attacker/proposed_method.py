import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser, setup_logger, pbar

logger = setup_logger(__name__)
config = config_parser()


class ProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.forward
        self.criterion = get_criterion()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_images = x_all.shape[0]
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
            labels, n_labels = [], []
            for n, _x in enumerate(x):
                pbar(n + 1, x.shape[0], f"batch {i} superpixel")
                _x = (_x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                converted = cv2.cvtColor(_x, cv2.COLOR_RGB2HSV_FULL)
                slic = cv2.ximgproc.createSuperpixelSLIC(
                    converted, cv2.ximgproc.MSLIC, config.region_size, config.ruler
                )
                slic.iterate(config.num_iterations)
                slic.enforceLabelConnectivity(config.min_element_size)
                label = slic.getLabels()
                labels.append(label)
                n_label = slic.getNumberOfSuperpixels()
                n_labels.append(n_label)
                # self.visualize(_x, slic)  # 可視化

            # initialize
            is_upper = []
            for idx in range(x.shape[0]):
                _is_upper = torch.zeros_like(x[0], dtype=torch.bool)
                for label in range(n_labels[idx]):
                    # RGB
                    _is_upper[0, labels == label] = torch.randint(
                        2, (1,), dtype=torch.bool
                    )
                    _is_upper[1, labels == label] = torch.randint(
                        2, (1,), dtype=torch.bool
                    )
                    _is_upper[2, labels == label] = torch.randint(
                        2, (1,), dtype=torch.bool
                    )
                is_upper.append(_is_upper)
            is_upper = torch.stack(is_upper)
            x_best = torch.where(is_upper, upper, lower)
            pred = F.softmax(self.model(x_best), dim=1)
            best_loss = self.criterion(pred, y)
            loss_storage[start:end, 0] = best_loss
            best_loss_storage[start:end, 0] = best_loss

            # loop: greedy
            for step in range(config.forward):
                pbar(step + 1, config.forward, f"batch {i} step")
                for idx in range(x.shape[0]):
                    c, label = np.random.randint(x.shape[1]), np.random.randint(
                        n_labels[idx]
                    )
                    is_upper[idx, c, labels == label] = ~is_upper[
                        idx, c, labels == label
                    ]
                x_adv = torch.where(is_upper, upper, lower)
                pred = F.softmax(self.model(x_adv), dim=1)
                loss = self.criterion(pred, y)
                best_loss = torch.where(loss > best_loss, loss, best_loss)
                loss_storage[start:end, step + 1] = loss
                best_loss_storage[start:end, step + 1] = best_loss
                x_best = torch.where(
                    (loss > best_loss).view(-1, 1, 1, 1), x_adv, x_best
                )
            x_adv_all.append(x_best)
        x_adv_all = torch.cat(x_adv_all, dim=0)
        return x_adv_all

    def visualize(self, _x, slic):
        from matplotlib import pyplot as plt

        superpixel = _x.copy()
        contour_mask = slic.getLabelContourMask(False)
        superpixel[contour_mask == 255] = (0, 0, 255)
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
        breakpoint()
