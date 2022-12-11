from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage.segmentation import slic
from torch import Tensor

from utils import config_parser, pbar

config = config_parser()


class SuperpixelManager:
    def __init__(self):
        pass

    def cal_superpixel(self, x):
        batch = np.arange(x.shape[0])
        if config.thread == 1:
            superpixel_storage = np.array(
                [self._cal_superpixel(x[idx], idx, batch.max() + 1) for idx in batch]
            )
        else:
            with ThreadPoolExecutor(config.thread) as executor:
                futures = [
                    executor.submit(self._cal_superpixel, x[idx], idx, batch.max() + 1)
                    for idx in batch
                ]
            superpixel_storage = np.array([future.result() for future in futures])
        level = np.zeros_like(batch)
        superpixel = superpixel_storage[batch, level]
        return superpixel

    def _cal_superpixel(self, x: Tensor, idx: int, total: int) -> list:
        """calculate superpixel per an image

        Args:
            x (Tensor): one image
            idx (int): index of image
            total (int): number of images

        Returns:
            list: various size of superpixel
        """
        pbar.debug(idx + 1, total, "cal_superpixel")
        superpixel_storage = []
        for n_segments in config.segments:
            img = (x.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            superpixel = slic(img, n_segments=n_segments)
            superpixel_storage.append(superpixel)
        return superpixel_storage
