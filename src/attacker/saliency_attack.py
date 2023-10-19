import math
import os

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torchvision import transforms as T

from base import Attacker, get_criterion
from sub import SODModel
from utils import ProgressBar, config_parser

config = config_parser()


class SaliencyAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.criterion = get_criterion()

        # saliency model
        self.saliency_model = SODModel()
        self.saliency_transform = T.Resize(256, antialias=True)
        weight_path = (
            "../storage/model/sod_model/best-model_epoch-204_mae-0.0505_loss-0.1370.pth"
        )

        assert os.path.exists(weight_path), (
            "download from"
            + "https://drive.google.com/file/d/1Sc7dgXCZjF4wVwBihmIry-Xk7wTqrJdr/view"
        )
        weights = torch.load(weight_path)
        self.saliency_model.load_state_dict(weights["model"])
        self.saliency_model = self.saliency_model.to(config.device)
        self.saliency_model.eval()

    def _attack(self, x_all: Tensor, y_all: Tensor):
        x_adv_all = []
        n_batch = math.ceil(self.n_images / config.batch_size)
        for b in range(n_batch):
            start = b * config.batch_size
            end = min((b + 1) * config.batch_size, self.n_images)
            x = x_all[start:end]
            self.y = y_all[start:end].to(config.device)
            self.upper = (x + config.epsilon).clamp(0, 1).clone()
            self.lower = (x - config.epsilon).clamp(0, 1).clone()
            self.best_loss = -100 * torch.ones(len(x)).to(config.device)
            self.forward = np.zeros(len(x), dtype=int)

            k_int = config.k_int
            threshold = config.threshold
            split_level = 1
            block = np.array([(None, 0, self.height, 0, self.width)] * len(x))

            self.saliency_detection = []
            n_sod_batch = math.ceil(len(x) / config.sod_batch)
            pbar = ProgressBar(n_sod_batch, "saliency map", color="cyan")
            for i in range(n_sod_batch):
                start = i * config.sod_batch
                end = min((i + 1) * config.sod_batch, len(x))
                image = torch.stack(
                    [self.saliency_transform(x_idx) for x_idx in x[start:end]]
                ).to(config.device)
                saliency_map = self.saliency_model(image)[0]
                resize = T.Resize(self.height, antialias=True)
                saliency_map = [resize(m)[0] for m in saliency_map]
                saliency_map = torch.stack(saliency_map).cpu()
                self.saliency_detection.append(saliency_map >= threshold)
                pbar.step()
            self.saliency_detection = torch.cat(self.saliency_detection)
            detected_pixels = self.saliency_detection.sum(axis=(1, 2))
            border_line = (self.height // k_int) * (self.width // k_int)
            not_detected = detected_pixels <= border_line
            pbar.end()
            while not_detected.sum() > 0:
                threshold /= 2
                image = x[not_detected].to(config.device)
                saliency_map = self.saliency_model(image)[0]
                self.saliency_detection[not_detected] = saliency_map >= threshold
                detected_pixels = self.saliency_detection.sum(axis=(1, 2))
                not_detected = detected_pixels <= border_line
            self.saliency_detection = torch.repeat_interleave(
                self.saliency_detection.unsqueeze(1), 3, dim=1
            )

            # refine search
            self.x_adv = x.clone()
            while True:
                self.refine(block, k_int, split_level)
                if self.forward.min() >= config.iter:
                    break
                elif k_int > 1:
                    assert k_int % 2 == 0
                    k_int //= 2
            x_adv_all.append(self.x_adv)
        x_adv_all = torch.concat(x_adv_all)
        del self.x_adv, self.upper, self.lower, self.best_loss, self.forward
        return x_adv_all

    def refine(self, search_block, k, split_level):
        if split_level == 1:
            split_blocks = []
            for c in range(self.n_channel):
                search_block[:, 0] = c
                split_blocks.append(self.split(search_block, k))
            split_blocks = np.concatenate(split_blocks)
            for blocks in split_blocks.transpose(1, 0, 2):
                np.random.shuffle(blocks)
            n_blocks = len(split_blocks)
            upper_loss = -100 * torch.ones((n_blocks, len(self.x_adv))).to(
                config.device
            )
            lower_loss = -100 * torch.ones((n_blocks, len(self.x_adv))).to(
                config.device
            )
            pbar = ProgressBar(n_blocks, f"split_level:{split_level}", color="cyan")
            for i, block in enumerate(split_blocks):
                _block = torch.zeros_like(self.x_adv, dtype=bool)
                for idx, b in enumerate(block):
                    _block[idx, b[0], b[1] : b[2], b[3] : b[4]] = True
                _block = _block & self.saliency_detection
                condition1 = self.forward < config.iter
                condition2 = (_block.sum(dim=(1, 2, 3)) != 0).cpu().numpy()
                condition = condition1 & condition2
                x_adv = torch.where(_block, self.upper, self.x_adv)
                prediction = self.model(x_adv[condition].to(config.device)).softmax(
                    dim=1
                )
                upper_loss[i, condition] = self.criterion(prediction, self.y[condition])
                self.forward += condition
                x_adv = torch.where(_block, self.lower, self.x_adv)
                prediction = self.model(x_adv[condition].to(config.device)).softmax(
                    dim=1
                )
                lower_loss[i, condition] = self.criterion(prediction, self.y[condition])
                self.forward += condition
                pbar.step()
            pbar.end()
        else:
            split_blocks = self.split(search_block, k)
            n_blocks = len(split_blocks)
            upper_loss = -100 * torch.ones((n_blocks, len(self.x_adv))).to(
                config.device
            )
            lower_loss = -100 * torch.ones((n_blocks, len(self.x_adv))).to(
                config.device
            )
            n_blocks = len(split_blocks)
            forward = self.forward.min()
            pbar = ProgressBar(
                n_blocks, f"split_level:{split_level}", f"{forward = }", color="cyan"
            )
            for i, block in enumerate(split_blocks):
                _block = torch.zeros_like(self.x_adv, dtype=torch.bool)
                for idx, b in enumerate(block):
                    _block[idx, b[0], b[1] : b[2], b[3] : b[4]] = True
                _block = _block & self.saliency_detection
                condition1 = self.forward < config.iter
                condition2 = (_block.sum(dim=(1, 2, 3)) != 0).cpu().numpy()
                condition = condition1 & condition2
                x_adv = torch.where(_block, self.upper, self.x_adv)
                prediction = self.model(x_adv[condition]).softmax(dim=1)
                upper_loss[i, condition] = self.criterion(prediction, self.y[condition])
                x_adv = torch.where(_block, self.lower, self.x_adv)
                prediction = self.model(x_adv[condition]).softmax(dim=1)
                lower_loss[i, condition] = self.criterion(prediction, self.y[condition])
                self.forward += condition
                pbar.step()
            pbar.end()

        loss_storage, u_is_better = torch.stack([lower_loss, upper_loss]).max(dim=0)
        indices = loss_storage.argsort(dim=0, descending=True)
        for index in indices:
            is_upper = u_is_better[index, np.arange(len(self.x_adv))].to(torch.bool)
            block = split_blocks[index.cpu().numpy(), np.arange(len(self.x_adv))]
            _block = torch.zeros_like(self.x_adv, dtype=torch.bool)
            for idx, b in enumerate(block):
                _block[idx, b[0], b[1] : b[2], b[3] : b[4]] = True
            _block = _block & self.saliency_detection
            loss = loss_storage[index, np.arange(len(self.x_adv))]
            update = loss >= self.best_loss
            upper_update = (update & is_upper).cpu().numpy()
            self.x_adv[upper_update] = torch.where(
                _block[upper_update],
                self.upper[upper_update],
                self.x_adv[upper_update],
            )
            lower_update = (update & ~is_upper).cpu().numpy()
            self.x_adv[lower_update] = torch.where(
                _block[lower_update],
                self.lower[lower_update],
                self.x_adv[lower_update],
            )
            self.best_loss[update] = loss[update]
            if k > 1 and self.forward.min() < config.iter:
                k //= 2
                self.refine(block, k, split_level + 1)

    def split(self, block: NDArray, k: int) -> np.ndarray:
        split_block = []
        for _block in block:
            n_blocks = ((_block[2] - _block[1]) // k, (_block[4] - _block[3]) // k)
            x1 = np.linspace(_block[1], _block[2] - k, n_blocks[0], dtype=int)
            x2 = np.linspace(_block[1] + k, _block[2], n_blocks[0], dtype=int)
            x = np.stack([np.repeat(x1, n_blocks[1]), np.repeat(x2, n_blocks[1])]).T
            y1 = np.linspace(_block[3], _block[4] - k, n_blocks[1], dtype=int)
            y2 = np.linspace(_block[3] + k, _block[4], n_blocks[1], dtype=int)
            y = np.stack([np.tile(y1, n_blocks[0]), np.tile(y2, n_blocks[0])]).T
            c = np.repeat(_block[0], n_blocks[0] * n_blocks[1])
            _split_block = np.stack([c, x[:, 0], x[:, 1], y[:, 0], y[:, 1]], axis=1)
            split_block.append(np.random.permutation(_split_block))
        split_block = np.stack(split_block, axis=1)
        return split_block
