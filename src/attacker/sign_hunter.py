import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, get_criterion
from utils import ProgressBar, config_parser

config = config_parser()


class SignHunter(Attacker):
    def __init__(self):
        super().__init__()
        self.criterion = get_criterion()

    def _attack(self, data: Tensor, label: Tensor):
        adv_data = []
        success_iteration = []
        n_batch = math.ceil(len(data) / config.batch_size)
        for b in range(n_batch):
            start = b * config.batch_size
            end = min((b + 1) * config.batch_size, len(data))
            x, y = data[start:end], label[start:end].to(config.device)

            # initial point
            sign = torch.ones_like(x)
            x_adv = (x + config.epsilon * sign).clamp(0, 1).to(config.device)
            prediction = self.model(x_adv).softmax(dim=1)
            best_loss = self.criterion(prediction, y).cpu()
            success_iter = (prediction.argmax(dim=1) == y).cpu().int()

            # search
            regions = self.get_regions()
            pbar = ProgressBar(config.iter, f"batch:{b}", "iter", color="cyan", start=1)
            for r in regions:
                _sign = sign.clone()
                _sign.permute(1, 2, 3, 0)[r] *= -1
                x_adv = (x + config.epsilon * _sign).clamp(0, 1).to(config.device)
                prediction = self.model(x_adv).softmax(dim=1)
                loss = self.criterion(prediction, y).cpu()
                update = loss >= best_loss
                sign[update] = _sign[update]
                best_loss[update] = loss[update]
                success_iter += (prediction.argmax(dim=1) == y).cpu().int()
                pbar.step()
            x_adv = (x + config.epsilon * sign).clamp(0, 1)
            adv_data.append(x_adv)
            success_iteration.append(success_iter)
            pbar.end()
        adv_data = torch.cat(adv_data, dim=0)
        success_iteration = torch.cat(success_iteration, dim=0)
        np.save(f"{config.savedir}/success_iter.npy", success_iteration.numpy())
        return adv_data

    def get_regions(self):
        image_size = self.n_channel * self.height * self.width
        d = image_size
        regions = [torch.ones((image_size), dtype=bool)]
        n_regions = 1
        while n_regions < config.iter - 1:
            d //= 2
            _n_regions = math.ceil(image_size / d)
            for r in range(_n_regions):
                start = r * d
                end = min((r + 1) * d, image_size)
                region = torch.zeros((image_size), dtype=bool)
                region[start:end] = True
                regions.append(region)
            n_regions += _n_regions
        regions = torch.stack(regions, dim=0)[: config.iter - 1]
        regions = regions.reshape(
            config.iter - 1, self.n_channel, self.height, self.width
        )
        return regions
