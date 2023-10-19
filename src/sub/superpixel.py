import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from skimage.segmentation import slic
from torch import Tensor

from utils import ProgressBar, config_parser, fix_seed

config = config_parser()


class Superpixel:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        config.cal_superpixel_time = 0
        assert config.alpha > 0

    def construct(self, x: Tensor):
        """calculate superpixel with multi_threading"""
        assert x.dim() == 4
        self.n_images, self.n_channel, self.height, self.width = x.shape
        self.threshold = self.height * self.width * self.threshold
        assert x.device.type == "cpu"
        self.pbar = ProgressBar(len(x), "superpixel", color="cyan")
        stopwatch = time.time()
        with ThreadPoolExecutor(config.thread) as executor:
            futures = executor.map(self._construct, x)
        config.cal_superpixel_time += time.time() - stopwatch
        self.pbar.end()
        self.storage, self.targets = zip(*futures)
        self.storage = self.cast_numpy(self.storage)
        self.targets = np.stack(self.targets, axis=0)

    def _construct(self, x_idx: Tensor):
        """calculate superpixel and targets per an image"""
        assert x_idx.dim() == 3
        image = (x_idx.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        superpixel = [np.ones((self.height, self.width), dtype=np.int64)]
        targets, n_targets = [], 0
        j = 0
        while n_targets < config.iter - 1:
            n_segments = config.segments_ratio ** (j + 1)
            _superpixel = slic(
                image,
                n_segments,
                config.alpha,
                enforce_connectivity=config.connected,
            )
            superpixel.append(_superpixel)
            superpixel, targets = self.build(superpixel, targets, j)
            n_targets = sum([len(t) for t in targets])
            j += 1
        superpixel = np.stack(superpixel, axis=0)[1:]
        targets = np.concatenate(targets, axis=0)[: config.iter - 1]
        self.pbar.step()
        return superpixel, targets

    def build(self, superpixel, targets, j):
        """stack superpixel and targets"""
        if (superpixel[-2] == superpixel[-1]).sum() >= self.threshold:
            return superpixel, targets
        labels = np.unique(superpixel[-1])
        areas = np.repeat(labels, self.n_channel)
        colors = np.tile(np.arange(self.n_channel), len(labels))
        _targets = np.stack([np.full(len(areas), j), colors, areas], axis=1)
        fix_seed(config.seed)
        targets.append(np.random.permutation(_targets))
        return superpixel, targets

    def cast_numpy(self, superpixel: list) -> NDArray:
        """resize superpixel and cast superpixel to numpy array"""
        n_superpixel = np.array([len(s) for s in superpixel])
        padding = np.max(n_superpixel) - n_superpixel
        superpixel = np.stack(
            [np.pad(s, [(0, p), (0, 0), (0, 0)]) for s, p in zip(superpixel, padding)],
            axis=0,
        )
        return superpixel
