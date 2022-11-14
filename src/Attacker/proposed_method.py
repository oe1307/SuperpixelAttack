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
        # TODO: batch処理
        index = -1
        x_adv_all = []
        n_images = x_all.shape[0]
        loss_storage = torch.zeros(n_images, config.forward)
        best_loss_storage = torch.zeros(n_images, config.forward)
        for x, y in zip(x_all, y_all):
            index += 1
            pbar(index + 1, n_images, "images")
            upper = (x + config.epsilon).clamp(0, 1)
            lower = (x - config.epsilon).clamp(0, 1)

            # superpixel
            _x = (x.clone().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            converted = cv2.cvtColor(_x, cv2.COLOR_RGB2HSV_FULL)
            slic = cv2.ximgproc.createSuperpixelSLIC(
                converted, cv2.ximgproc.MSLIC, config.region_size, config.ruler
            )
            slic.iterate(config.num_iterations)
            slic.enforceLabelConnectivity(config.min_element_size)
            labels = slic.getLabels()
            n_labels = slic.getNumberOfSuperpixels()
            # self.visualize(_x, slic)  # 可視化

            # initialize
            is_upper = torch.zeros_like(x, dtype=torch.bool)
            _is_upper = torch.randint(
                0, 2, (n_labels, x.shape[0]), dtype=torch.bool, device=config.device
            )
            for label in range(n_labels):
                # RGB
                is_upper[0, labels == label] = _is_upper[label, 0]
                is_upper[1, labels == label] = _is_upper[label, 1]
                is_upper[2, labels == label] = _is_upper[label, 2]
            x_best = torch.where(is_upper, upper, lower)
            pred = F.softmax(self.model(x_best.unsqueeze(0)), dim=1)
            best_loss = self.criterion(pred, y)

            # roop: greedy
            for step in range(config.forward):
                c, label = np.random.randint(x.shape[0]), np.random.randint(n_labels)
                is_upper[c, labels == label] = ~is_upper[c, labels == label]
                x_adv = torch.where(is_upper, upper, lower)
                pred = F.softmax(self.model(x_adv.unsqueeze(0)), dim=1)
                loss = self.criterion(pred, y)
                loss_storage[index, step] = loss.clone()
                if loss > best_loss:
                    best_loss = loss.clone()
                    x_best = x_adv.clone()
                best_loss_storage[index, step] = best_loss.clone()

            x_adv_all.append(x_adv)
        np.save(f"{config.savedir}/loss.npy", loss_storage.cpu().numpy())
        np.save(f"{config.savedir}/best_loss.npy", best_loss_storage.cpu().numpy())
        x_adv_all = torch.stack(x_adv_all)
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
