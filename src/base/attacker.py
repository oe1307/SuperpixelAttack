import math
import time

import torch
from torch import Tensor
from torch.nn import Module

from utils import ProgressBar, config_parser, printc

config = config_parser()


class Attacker:
    @torch.no_grad()
    def attack(self, model: Module, data: Tensor, label: Tensor):
        assert not model.training
        self.model = model
        success_classify = self.classify(data, label, "original")
        target_data, target_label = data[success_classify], label[success_classify]
        self.n_images, self.n_channel, self.height, self.width = target_data.shape
        stopwatch = time.time()
        adv_data = self._attack(target_data, target_label)
        total_time = time.time() - stopwatch
        adv_data = self.check_x_adv(adv_data, target_data)
        success_classify = self.classify(adv_data, target_label, "adversarial")
        attack_success_rate = (1 - success_classify.sum() / len(data)) * 100
        message = (
            "\n"
            f"attacker = {config.attacker}\n"
            f"n_examples = {self.n_images}/{config.n_examples}\n"
            f"epsilon = {config.epsilon}\n"
            f"model = {config.model}\n"
            f"iter = {config.iter}\n"
            f"total_time = {total_time:.2f} (sec)\n"
            f"attack_success_rate = {attack_success_rate:.2f}%\n"
        )
        printc("yellow", message)
        print(message, file=open(f"{config.savedir}/summary.txt", "w"))

    def classify(self, data: Tensor, label: Tensor, dset: str):
        success_classify = []
        n_batch = math.ceil(len(data) / config.batch_size)
        pbar = ProgressBar(n_batch, f"classify {dset} images", color="cyan")
        for b in range(n_batch):
            start = b * config.batch_size
            end = min((b + 1) * config.batch_size, len(data))
            x = data[start:end].to(config.device)
            y = label[start:end].to(config.device)
            prediction = self.model(x).softmax(dim=1)
            success_classify.append(prediction.argmax(dim=1) == y)
            pbar.step()
        success_classify = torch.cat(success_classify, dim=0).cpu()
        pbar.end()
        return success_classify

    def check_x_adv(self, x_adv: Tensor, x: Tensor):
        upper = (x + config.epsilon).clamp(0, 1)
        lower = (x - config.epsilon).clamp(0, 1)
        assert (x_adv <= upper).all() and (x_adv >= lower).all()
        x_adv = x_adv.clamp(lower, upper)
        return x_adv
