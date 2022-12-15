import math

import numpy as np
import torch
from torchvision import transforms as T

from base import SODModel
from utils import config_parser, pbar, setup_logger

from .base_remover import Remover

logger = setup_logger(__name__)
config = config_parser()


class SaliencyRemover(Remover):
    def __init__(self, update_area, update_method):
        super().__init__(update_area, update_method)
        self.saliency_model = SODModel()
        self.saliency_transform = T.Resize(256)
        weights = torch.load("../storage/model/saliency/saliency_weight.pth")
        self.saliency_model.load_state_dict(weights["model"])
        self.saliency_model.to(config.device)
        self.saliency_model.eval()

    def initialize(self, update_area, targets, forward):
        threshold = config.threshold
        self.saliency_detection = []
        n_saliency_batch = math.ceil(self.update_area.batch / config.saliency_batch)
        for j in range(n_saliency_batch):
            pbar.debug(j + 1, n_saliency_batch, "saliency map")
            start = j * config.saliency_batch
            end = min((j + 1) * config.saliency_batch, self.update_area.batch)
            img = torch.stack(
                [
                    self.saliency_transform(x_idx)
                    for x_idx in self.update_method.x_best[start:end]
                ]
            )
            saliency_map = self.saliency_model(img)[0].cpu()
            saliency_map = [
                T.Resize(self.update_area.height)(m).numpy()[0] for m in saliency_map
            ]
            self.saliency_detection.append(np.array(saliency_map) >= threshold)
        self.saliency_detection = np.concatenate(self.saliency_detection, axis=0)
        detected_pixels = self.saliency_detection.sum(axis=(1, 2))
        not_detected = detected_pixels <= (self.update_area.height // config.k_init) * (
            self.update_area.width // config.k_init
        )
        while not_detected.sum() > 0:
            logger.warning(f"{threshold=} -> {not_detected.sum()} images not detected")
            threshold /= 2
            saliency_map = (
                self.saliency_model(self.update_method.x_best[not_detected])[0]
                .cpu()
                .numpy()
            )
            self.saliency_detection[not_detected] = saliency_map >= threshold
            detected_pixels = self.saliency_detection.sum(axis=(1, 2))
            not_detected = detected_pixels <= (
                self.update_area.height // config.k_init
            ) * (self.update_area.width // config.k_init)
        return targets
