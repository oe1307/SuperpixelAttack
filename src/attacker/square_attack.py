import math

import numpy as np
import torch
from torch import Tensor

from base import Attacker, get_criterion
from utils import ProgressBar, config_parser

config = config_parser()


class SquareAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.criterion = get_criterion()

    def _attack(self, data: Tensor, label: Tensor) -> Tensor:
        adv_data = []
        success_iteration = []
        n_batch = math.ceil(len(data) / config.batch_size)
        for b in range(n_batch):
            start = b * config.batch_size
            end = min((b + 1) * config.batch_size, len(data))
            x, y = data[start:end], label[start:end].to(config.device)

            # initial point
            upper = (x + config.epsilon).clamp(0, 1).to(config.device)
            lower = (x - config.epsilon).clamp(0, 1).to(config.device)
            random = 2 * torch.rand((len(x), self.n_channel, 1, self.width)) - 1
            random = torch.sign(random)
            x_best = (x + config.epsilon * random).to(config.device)
            x_best = x_best.clamp(lower, upper)
            prediction = self.model(x_best).softmax(dim=1)
            best_loss = self.criterion(prediction, y).cpu()
            success_iter = (prediction.argmax(dim=1) == y).cpu().int()

            # search
            pbar = ProgressBar(config.iter, f"batch:{b}", "iter", color="cyan", start=1)
            for self.iter in range(1, config.iter):
                delta = self.search()
                x_adv = (x_best + delta).clamp(lower, upper).to(config.device)
                prediction = self.model(x_adv).softmax(dim=1)
                loss = self.criterion(prediction, y).cpu()
                update = loss >= best_loss
                x_best[update] = x_adv[update].clone()
                best_loss[update] = loss[update]
                success_iter += (prediction.argmax(dim=1) == y).cpu().int()
                pbar.step()
            adv_data.append(x_best.cpu())
            success_iteration.append(success_iter)
            pbar.end()
        adv_data = torch.cat(adv_data, dim=0)
        success_iteration = torch.cat(success_iteration, dim=0)
        np.save(f"{config.savedir}/success_iter.npy", success_iteration.numpy())
        return adv_data

    def search(self) -> Tensor:
        p = self.p_selection(self.iter)
        s = max(int(round(math.sqrt(p * self.height * self.width))), 1)
        vh = ((self.height - s) * torch.rand(1)).long()
        vw = ((self.width - s) * torch.rand(1)).long()
        delta = torch.zeros((self.n_channel, self.height, self.width))
        random = 2 * torch.rand((self.n_channel, 1, 1)) - 1
        random = torch.sign(random)
        delta[:, vh : vh + s, vw : vw + s] = 2.0 * config.epsilon * random
        delta = delta.to(config.device)
        return delta

    def p_selection(self, iter: int):
        schedule = (0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
        schedule = np.array(schedule) * config.iter
        p = config.p_init / (2 ** (schedule <= iter).sum())
        return p
