import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, get_criterion
from utils import ProgressBar, config_parser

config = config_parser()


class AccSignHunter(Attacker):
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
            sign = torch.ones_like(x, dtype=int)
            x_adv = (x + config.epsilon * sign).clamp(0, 1).to(config.device)
            prediction = self.model(x_adv).softmax(dim=1)
            best_loss = self.criterion(prediction, y).cpu()
            success_iter = (prediction.argmax(dim=1) == y).cpu().int()
            sign = sign.reshape(len(x), -1)

            # search
            iter, node, depth = 1, 0, 0
            regions = torch.tensor((0, x[0].numel())).repeat(len(x), 1, 1)
            assert len(regions) == len(x)
            reduction = torch.zeros(len(x), 1)
            assert regions.shape[:2] == reduction.shape
            pbar = ProgressBar(config.iter, f"batch:{b}", "iter", color="cyan", start=1)
            while iter < config.iter:
                _sign = sign.clone()
                for idx, (start, end) in enumerate(regions[:, node].numpy()):
                    _sign[idx, start:end] *= -1
                x_adv = (x + config.epsilon * _sign.reshape(x.shape)).clamp(0, 1)
                x_adv = x_adv.to(config.device)
                if node % 2 == 1:
                    reduction[:, node] -= reduction[:, node - 1]
                else:
                    prediction = self.model(x_adv).softmax(dim=1)
                    loss = self.criterion(prediction, y).cpu()
                    reduction[:, node] = best_loss - loss
                    iter += 1
                    pbar.step()
                update = reduction[:, node] <= 0
                sign[update] = _sign[update]
                best_loss[update] = loss[update]
                success_iter += (prediction.argmax(dim=1) == y).cpu().int()
                node += 1
                if node == 2**depth:
                    node, depth = 0, depth + 1
                    regions, reduction = self.devide(regions, reduction)
            x_adv = (x + config.epsilon * sign.reshape(x.shape)).clamp(0, 1)
            adv_data.append(x_adv)
            success_iteration.append(success_iter)
            pbar.end()
        adv_data = torch.cat(adv_data, dim=0)
        success_iteration = torch.cat(success_iteration, dim=0)
        np.save(f"{config.savedir}/success_iter.npy", success_iteration.numpy())
        return adv_data

    def devide(self, _regions, reduction):
        regions = []
        for r in _regions.permute(1, 0, 2):
            mid = (r[:, 0] + r[:, 1]) // 2
            regions.append(torch.stack((r[:, 0], mid), dim=1))
            regions.append(torch.stack((mid, r[:, 1]), dim=1))
        regions = torch.stack(regions, dim=1)
        reduction = reduction.repeat_interleave(2, 1)
        reduction, order = reduction.sort(dim=1)
        regions = regions.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))
        return regions, reduction
