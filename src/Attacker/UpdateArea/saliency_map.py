import math

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms as T

from base import SODModel
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SaliencyMap:
    def __init__(self):
        self.saliency_model = SODModel()
        self.saliency_transform = T.Resize(256)
        weights = torch.load("../storage/model/saliency/saliency_weight.pth")
        self.saliency_model.load_state_dict(weights["model"])
        self.saliency_model.to(config.device)
        self.saliency_model.eval()

    def initialize(self, x: Tensor, forward: np.ndarray):
        self.batch, self.n_channel, self.height, self.width = x.shape

        # saliency_map
        self.k_init = config.k_init
        threshold = config.threshold
        self.saliency_detection = []
        n_saliency_batch = math.ceil(self.batch / config.saliency_batch)
        for i in range(n_saliency_batch):
            pbar.debug(i + 1, n_saliency_batch, "saliency map")
            start = i * config.saliency_batch
            end = min((i + 1) * config.saliency_batch, self.batch)
            img = torch.stack(
                [self.saliency_transform(x_idx) for x_idx in x[start:end]]
            )
            saliency_map = self.saliency_model(img)[0].cpu()
            saliency_map = [T.Resize(self.height)(m).numpy()[0] for m in saliency_map]
            self.saliency_detection.append(np.array(saliency_map) >= threshold)
        self.saliency_detection = np.concatenate(self.saliency_detection, axis=0)
        detected_pixels = self.saliency_detection.sum(axis=(1, 2))
        not_detected = detected_pixels <= (self.height // self.k_init) * (
            self.width // self.k_init
        )
        while not_detected.sum() > 0:
            logger.warning(f"{threshold=} -> {not_detected.sum()} images not detected")
            threshold /= 2
            saliency_map = self.saliency_model(x[not_detected])[0].cpu().numpy()
            self.saliency_detection[not_detected] = saliency_map >= threshold
            detected_pixels = self.saliency_detection.sum(axis=(1, 2))
            not_detected = detected_pixels <= (self.height // self.k_init) * (
                self.width // self.k_init
            )

        # split_square
        self.split = config.initial_split
        assert self.height % self.split == 0
        h = self.height // self.split
        assert self.width % self.split == 0
        w = self.width // self.split
        self.update_area = np.arange(h * w).reshape(1, h, w)
        self.update_area = np.repeat(self.update_area, self.batch, axis=0)
        self.update_area = np.repeat(self.update_area, self.split, axis=1)
        self.update_area = np.repeat(self.update_area, self.split, axis=2)

        # intersection
        self.update_area[~self.saliency_detection] = -1
        self.targets = []
        for idx in range(self.batch):
            labels = np.unique(self.update_area[idx])
            if config.channel_wise:
                channel = np.tile(np.arange(self.n_channel), len(labels[labels != -1]))
                labels = np.repeat(labels[labels != -1], self.n_channel)
                _targets = np.stack([channel, labels], axis=1)
                self.targets.append(np.random.permutation(_targets))
            else:
                self.targets.append(np.random.permutation(labels[labels != -1]))
        return self.update_area, self.targets

    def next(self, forward: np.ndarray):
        if self.targets.shape[0] == 1:
            breakpoint()
        else:
            self.targets = np.delete(self.targets, 0)
        return self.update_area, self.targets
