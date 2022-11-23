import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from skimage.segmentation import mark_boundaries, slic
from torch import Tensor

from base import Attacker, get_criterion
from utils import change_level, config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class LocalSearchProposedMethod(Attacker):
    def __init__(self):
        config.n_forward = config.step
        self.criterion = get_criterion()

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        n_images = x_all.shape[0]
        n_chanel = x_all.shape[1]
        n_batch = math.ceil(n_images / self.model.batch_size)
        for b in range(n_batch):
            start = b * self.model.batch_size
            end = min((b + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            y = y_all[start:end]
            batch = np.arange(x.shape[0])
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()

            # initialize
            is_upper_best = torch.zeros_like(x, dtype=torch.bool)
            x_best = lower.clone()
            pred = self.model(x_best).softmax(1)
            best_loss = self.criterion(pred, y)
            forward = np.ones_like(batch)

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

            # local search
            while True:
                # update superpixel
                superpixel_level += forward == 3 * n_superpixel
                superpixel = superpixel_storage[batch, superpixel_level]
                n_superpixel = superpixel.max(axis=(1, 2))

                is_upper = []
                for idx in batch:

                    _is_upper = is_upper_best[idx].clone()
                    _best_loss = -10
                    for c, label in zip(chanel, labels):
                        if tabu_list[idx, c, label - 1] > 0:
                            tabu_list[idx, c, label - 1] -= 1
                            continue

                pbar.debug(forward.min(), config.step, "forward", f"batch: {b}")
                if forward.min() >= config.step:
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
