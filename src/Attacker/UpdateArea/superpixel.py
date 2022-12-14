from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage.segmentation import slic
from torch import Tensor

from utils import config_parser, pbar

config = config_parser()


class Superpixel:
    def __init__(self):
        pass

    def initialize(self, x: Tensor, forward: np.ndarray):
        self.batch, self.n_channel = x.shape[:2]
        self.superpixel = self.cal_superpixel(x)
        self.level = np.zeros(self.batch, dtype=int)
        self.update_area = self.superpixel[np.arange(self.batch), self.level]
        n_update_area = self.update_area.max(axis=(1, 2))
        if config.channel_wise:
            targets = []
            for idx in range(self.batch):
                channel = np.tile(np.arange(self.n_channel), n_update_area[idx])
                labels = np.repeat(range(1, n_update_area[idx] + 1), self.n_channel)
                _target = np.stack([channel, labels], axis=1)
                targets.append(np.random.permutation(_target))
        else:
            targets = []
            for idx in range(self.batch):
                labels = range(1, n_update_area[idx] + 1)
                targets.append(np.random.permutation(labels))
        return self.update_area, targets

    def next(self, forward: np.ndarray, targets):
        for idx in range(self.batch):
            if targets[idx].shape[0] == 0:
                self.level[idx] = min(self.level[idx] + 1, len(config.segments) - 1)
                self.update_area[idx] = self.superpixel[idx, self.level[idx]]
                _n_update_area = self.update_area[idx].max()
                if config.channel_wise:
                    channel = np.tile(np.arange(self.n_channel), _n_update_area)
                    labels = np.repeat(range(1, _n_update_area + 1), self.n_channel)
                    _target = np.stack([channel, labels], axis=1)
                else:
                    _target = np.arange(1, _n_update_area + 1)
                targets[idx] = np.random.permutation(_target)
        return self.update_area, targets

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
