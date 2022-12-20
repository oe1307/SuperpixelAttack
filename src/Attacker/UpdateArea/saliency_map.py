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
        if config.update_method != "adaptive_search":
            raise ValueError("Update area is only available for adaptive search.")
        self.saliency_model = SODModel().to(config.device)
        self.saliency_transform = T.Resize(256)
        weights = torch.load("../storage/model/saliency/saliency_weight.pth")
        self.saliency_model.load_state_dict(weights["model"])
        self.saliency_model.eval()

    def initialize(self, x: Tensor, level: np.ndarray):
        self.batch, self.n_channel, self.height, self.width = x.shape
        self.saliency_map(x)

        update_area = []
        for idx in range(self.batch):
            k_int = max(config.k_int // 2 ** level[idx], 1)
            assert self.height % k_int == 0
            h = self.height // k_int
            assert self.width % k_int == 0
            w = self.width // k_int
            square = np.arange(1, h * w + 1).reshape(h, w)
            square = np.repeat(square, k_int, axis=0)
            square = np.repeat(square, k_int, axis=1)
            update_area.append(square)
        update_area = np.stack(update_area)
        assert update_area.shape == (self.batch, self.height, self.width)
        update_area[~self.saliency_detection] = 0
        return update_area

    def update(self, idx: int, level: np.ndarray):
        k_int = max(config.k_int // 2**level, 1)
        assert self.height % k_int == 0
        h = self.height // k_int
        assert self.width % k_int == 0
        w = self.width // k_int
        update_area = np.arange(1, h * w + 1).reshape(h, w)
        update_area = np.repeat(update_area, k_int, axis=0)
        update_area = np.repeat(update_area, k_int, axis=1)
        assert update_area.shape == (self.height, self.width)
        update_area[~self.saliency_detection[idx]] = 0
        return update_area

    def saliency_map(self, x: Tensor):
        threshold = config.threshold
        self.saliency_detection = []
        n_saliency_batch = math.ceil(self.batch / config.saliency_batch)
        for i in range(n_saliency_batch):
            pbar.debug(i + 1, n_saliency_batch, "saliency map")
            start = i * config.saliency_batch
            end = min((i + 1) * config.saliency_batch, self.batch)
            img = torch.stack(
                [self.saliency_transform(x_idx) for x_idx in x[start:end]]
            ).to(config.device)
            saliency_map = self.saliency_model(img)[0]
            saliency_map = [T.Resize(self.height)(m)[0] for m in saliency_map]
            saliency_map = torch.stack(saliency_map)
            self.saliency_detection.append(saliency_map >= threshold)
        self.saliency_detection = torch.cat(self.saliency_detection)
        detected_pixels = self.saliency_detection.sum(axis=(1, 2))
        not_detected = detected_pixels <= (self.height // config.k_int) * (
            self.width // config.k_int
        )
        while not_detected.sum() > 0:
            logger.warning(f"{threshold=} -> {not_detected.sum()} images not detected")
            threshold /= 2
            saliency_map = self.saliency_model(x[not_detected])[0]
            self.saliency_detection[not_detected] = saliency_map >= threshold
            detected_pixels = self.saliency_detection.sum(axis=(1, 2))
            not_detected = detected_pixels <= (self.height // config.k_int) * (
                self.width // config.k_int
            )
        self.saliency_detection = self.saliency_detection.cpu().numpy()
