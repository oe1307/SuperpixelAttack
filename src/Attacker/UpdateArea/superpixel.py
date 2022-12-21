from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage.segmentation import slic
from torch import Tensor

from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class Superpixel:
    def __init__(self):
        pass

    def initialize(self, x: Tensor, level: np.ndarray):
        batch = x.shape[0]
        self.superpixel = self.cal_superpixel(x)
        update_areas = self.superpixel[np.arange(batch), level]
        return update_areas

    def update(self, idx: int, level: int):
        if level > len(config.segments) - 1:
            logger.warning("level is out of range")
            level = len(config.segments) - 1
        update_area = self.superpixel[idx, level]
        return update_area

    def cal_superpixel(self, x):
        """calculate superpixel with multi-threading"""
        batch = np.arange(x.shape[0])
        if config.thread == 1:
            superpixel = np.array(
                [self._cal_superpixel(x[idx], idx, batch.max() + 1) for idx in batch]
            )
        else:
            with ThreadPoolExecutor(config.thread) as executor:
                futures = [
                    executor.submit(self._cal_superpixel, x[idx], idx, batch.max() + 1)
                    for idx in batch
                ]
            superpixel = np.array([future.result() for future in futures])
        return superpixel

    def _cal_superpixel(self, x: Tensor, idx: int, total: int) -> list:
        """calculate superpixel per an image"""
        pbar.debug(idx + 1, total, "cal_superpixel")
        superpixel_storage = []
        for n_segments in config.segments:
            img = (x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            superpixel = slic(img, n_segments=n_segments)
            superpixel_storage.append(superpixel)
        return superpixel_storage
