import math
import time

import numpy as np
import torch
from torch import Tensor

from base import Attacker, get_criterion
from sub import Superpixel
from utils import ProgressBar, config_parser

config = config_parser()


class SuperpixelAttack(Attacker):
    def __init__(self):
        super().__init__()
        config.forward_time = 0
        self.criterion = get_criterion()
        self.superpixel = Superpixel()

    def _attack(self, data: Tensor, label: Tensor) -> Tensor:
        adv_data = []
        success_iteration = []
        n_batch = math.ceil(len(data) / config.batch_size)
        for b in range(n_batch):
            start = b * config.batch_size
            end = min((b + 1) * config.batch_size, len(data))
            x, y = data[start:end], label[start:end]
            self.batch = len(x)
            self.superpixel.construct(x)
            x, y = x.to(config.device), y.to(config.device)

            # initial point
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()
            x_adv = lower.clone()
            is_upper_best = torch.zeros_like(x, dtype=bool)
            stopwatch = time.time()
            prediction = self.model(x_adv).softmax(dim=1)
            best_loss = self.criterion(prediction, y)
            config.forward_time += time.time() - stopwatch
            success_iter = (prediction.argmax(dim=1) == y).cpu().int()

            # search
            pbar = ProgressBar(config.iter, f"batch:{b}", "iter", color="cyan", start=1)
            for self.iter in range(1, config.iter):
                is_upper = is_upper_best.clone()
                target = self.search()
                is_upper[target] = ~is_upper[target]
                x_adv = lower.clone()
                x_adv[is_upper] = upper[is_upper].clone()
                stopwatch = time.time()
                prediction = self.model(x_adv).softmax(dim=1)
                loss = self.criterion(prediction, y)
                config.forward_time += time.time() - stopwatch
                update = loss >= best_loss
                is_upper_best[update] = is_upper[update]
                best_loss[update] = loss[update]
                success_iter += (prediction.argmax(dim=1) == y).cpu().int()
                pbar.step()
            x_adv = lower.clone()
            x_adv[is_upper_best] = upper[is_upper_best].clone()
            adv_data.append(x_adv.cpu())
            success_iteration.append(success_iter)
            pbar.end()
        adv_data = torch.cat(adv_data, dim=0)
        success_iteration = torch.cat(success_iteration, dim=0)
        np.save(f"{config.savedir}/success_iter.npy", success_iteration.numpy())
        return adv_data

    def search(self) -> Tensor:
        n_targets = len(self.superpixel.storage)
        j = self.superpixel.targets[:, self.iter - 1, 0]
        color = self.superpixel.targets[:, self.iter - 1, 1]
        area = self.superpixel.targets[:, self.iter - 1, 2]

        target = np.zeros(
            (self.batch, self.n_channel, self.height, self.width), dtype=bool
        )
        area = area.repeat(self.height * self.width)
        area = area.reshape(n_targets, self.height, self.width)
        search_area = self.superpixel.storage[range(n_targets), j] == area
        target[range(n_targets), color] = search_area
        target = torch.from_numpy(target)
        return target
