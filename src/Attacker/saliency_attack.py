import math

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms as T

from base import Attacker, SODModel, get_criterion
from utils import config_parser, pbar, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SaliencyAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.criterion = get_criterion()
        config.n_forward = config.step

        # saliency model
        self.saliency_model = SODModel()
        self.saliency_transform = T.Resize(256)
        weights = torch.load("../storage/model/saliency/saliency_weight.pth")
        self.saliency_model.load_state_dict(weights["model"])
        self.saliency_model.to(config.device)
        self.saliency_model.eval()

    def _attack(self, x_all: Tensor, y_all: Tensor):
        x_adv_all = []
        n_images, n_channel, height, width = x_all.shape
        n_batch = math.ceil(n_images / self.model.batch_size)
        for i in range(n_batch):
            start = i * self.model.batch_size
            end = min((i + 1) * self.model.batch_size, n_images)
            x = x_all[start:end]
            self.y = y_all[start:end]
            batch = x.shape[0]
            self.upper = (x + config.epsilon).clamp(0, 1).clone()
            self.lower = (x - config.epsilon).clamp(0, 1).clone()
            self.best_loss = -100 * torch.ones(batch, device=config.device)
            self.forward = np.zeros(batch, dtype=int)

            k_int = config.k_int
            split_level = 1
            block = np.array([(None, 0, height, 0, width)] * batch)

            threshold = config.threshold
            self.saliency_detection = []
            n_saliency_batch = math.ceil(batch / config.saliency_batch)
            for j in range(n_saliency_batch):
                pbar.debug(j + 1, n_saliency_batch, "saliency map")
                start = j * config.saliency_batch
                end = min((j + 1) * config.saliency_batch, batch)
                img = torch.stack(
                    [self.saliency_transform(x_idx) for x_idx in x[start:end]]
                )
                saliency_map = self.saliency_model(img)[0]
                saliency_map = [T.Resize(height)(m)[0] for m in saliency_map]
                saliency_map = torch.stack(saliency_map)
                self.saliency_detection.append(saliency_map >= threshold)
            self.saliency_detection = torch.cat(self.saliency_detection)
            detected_pixels = self.saliency_detection.sum(axis=(1, 2))
            not_detected = detected_pixels <= (height // k_int) * (width // k_int)
            while not_detected.sum() > 0:
                logger.warning(
                    f"{threshold=} -> {not_detected.sum()} images not detected"
                )
                threshold /= 2
                saliency_map = self.saliency_model(x[not_detected])[0]
                self.saliency_detection[not_detected] = saliency_map >= threshold
                detected_pixels = self.saliency_detection.sum(axis=(1, 2))
                not_detected = detected_pixels <= (height // k_int) * (width // k_int)
            self.saliency_detection = torch.repeat_interleave(
                self.saliency_detection.unsqueeze(1), 3, dim=1
            )

            # refine search
            self.x_adv = x.clone()
            while True:
                self.refine(block, k_int, split_level)
                if self.forward.min() >= config.step:
                    break
                elif k_int > 1:
                    assert k_int % 2 == 0
                    k_int //= 2
            x_adv_all.append(self.x_adv)
        x_adv_all = torch.concat(x_adv_all)
        del self.x_adv, self.upper, self.lower, self.best_loss, self.forward
        return x_adv_all

    def refine(self, search_block, k, split_level):
        batch, n_channel = self.x_adv.shape[:2]

        if split_level == 1:
            split_blocks = []
            for c in range(n_channel):
                search_block[:, 0] = c
                split_blocks.append(self.split(search_block, k))
            split_blocks = np.concatenate(split_blocks)
            for blocks in split_blocks.transpose(1, 0, 2):
                np.random.shuffle(blocks)

            upper_loss = -100 * torch.ones(
                (split_blocks.shape[0], batch), device=config.device
            )
            lower_loss = -100 * torch.ones(
                (split_blocks.shape[0], batch), device=config.device
            )
            for i, block in enumerate(split_blocks):
                _block = torch.zeros_like(self.x_adv, dtype=bool)
                for idx, b in enumerate(block):
                    _block[idx, b[0], b[1] : b[2], b[3] : b[4]] = True
                _block = _block & self.saliency_detection
                condition1 = self.forward < config.step
                condition2 = (_block.sum(dim=(1, 2, 3)) != 0).cpu().numpy()
                condition = condition1 & condition2
                x_adv = torch.where(_block, self.upper, self.x_adv)
                pred = self.model(x_adv[condition]).softmax(dim=1)
                upper_loss[i, condition] = self.criterion(pred, self.y[condition])
                self.forward += condition
                x_adv = torch.where(_block, self.lower, self.x_adv)
                pred = self.model(x_adv[condition]).softmax(dim=1)
                lower_loss[i, condition] = self.criterion(pred, self.y[condition])
                self.forward += condition
                pbar.debug(
                    i + 1,
                    split_blocks.shape[0],
                    f"{split_level = }",
                    f"forward = {self.forward.min()}",
                )
                if self.forward.min() >= config.step:
                    logger.debug("")
                    break
        else:
            split_blocks = self.split(search_block, k)
            upper_loss = -100 * torch.ones(
                (split_blocks.shape[0], batch), device=config.device
            )
            lower_loss = -100 * torch.ones(
                (split_blocks.shape[0], batch), device=config.device
            )
            for i, block in enumerate(split_blocks):
                _block = torch.zeros_like(self.x_adv, dtype=torch.bool)
                for idx, b in enumerate(block):
                    _block[idx, b[0], b[1] : b[2], b[3] : b[4]] = True
                _block = _block & self.saliency_detection
                condition1 = self.forward < config.step
                condition2 = (_block.sum(dim=(1, 2, 3)) != 0).cpu().numpy()
                condition = condition1 & condition2
                x_adv = torch.where(_block, self.upper, self.x_adv)
                pred = self.model(x_adv[condition]).softmax(dim=1)
                upper_loss[i, condition] = self.criterion(pred, self.y[condition])
                x_adv = torch.where(_block, self.lower, self.x_adv)
                pred = self.model(x_adv[condition]).softmax(dim=1)
                lower_loss[i, condition] = self.criterion(pred, self.y[condition])
                self.forward += condition
                pbar.debug(
                    i + 1,
                    split_blocks.shape[0],
                    f"{split_level = }",
                    f"forward = {self.forward.min()}",
                )
                if self.forward.min() >= config.step:
                    logger.debug("")
                    break

        loss_storage, u_is_better = torch.stack([lower_loss, upper_loss]).max(dim=0)
        indices = loss_storage.argsort(dim=0, descending=True)
        for index in indices:
            is_upper = u_is_better[index, np.arange(batch)].to(torch.bool)
            block = split_blocks[index.cpu().numpy(), np.arange(batch)]
            _block = torch.zeros_like(self.x_adv, dtype=torch.bool)
            for idx, b in enumerate(block):
                _block[idx, b[0], b[1] : b[2], b[3] : b[4]] = True
            _block = _block & self.saliency_detection
            loss = loss_storage[index, np.arange(batch)]
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
            if k > 1 and self.forward.min() < config.step:
                k //= 2
                self.refine(block, k, split_level + 1)

    def split(self, block: np.ndarray, k: int) -> np.ndarray:
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
