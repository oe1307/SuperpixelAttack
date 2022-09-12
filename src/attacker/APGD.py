import numpy as np
import torch

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)


class APGD_Attacker(Attacker):
    def __init__(self):
        super().__init__()

    def _recorder(self):
        config = config_parser.config
        self.step_size = torch.zeros(
            (config.n_examples, config.iteration), device=config.device
        )

    def _attack(self, model, x, y, criterion, start, end):
        config = config_parser.config
        upper = (x + config.epsilon).clamp(0, 1).detach().clone()
        lower = (x - config.epsilon).clamp(0, 1).detach().clone()

        x_adv = x.detach().clone()
        x_adv.requires_grad_()
        for i in range(config.iteration):
            logger.info(f"   iteration {i + 1}")
            self.step_size_manager(i, start, end)
            logits = model(x_adv)
            loss = criterion(logits, y).sum()
            grad = torch.autograd.grad(loss, [x_adv])[0]
            step_size = self.step_size[start:end, i].view(-1, 1, 1, 1)
            x_adv = x_adv - step_size * torch.sign(grad)
            x_adv = torch.clamp(x_adv, lower, upper)
            self._robust_acc(logits, y, loss, start, end, i)
            del loss, grad

    def step_size_manager(self, i, start, end):
        config = config_parser.config
        if i == 0:
            self.step_size[:, 0] = config.step_size
            self.checkpoint = int(0.22 * config.iteration)
            self.checkpoint_interval = int(0.22 * config.iteration)
            self.checkpoint_decay = int(0.03 * config.iteration)
            self.checkpoint_min = int(0.06 * config.iteration)
            self.checker = torch.zeros(end - start)
        elif i == self.checkpoint:
            if False:
                condition1 = 1 < config.rho * self.checkpoint_interval
                condition2_1 = (
                    self.step_size[
                        start:end, self.checkpoint - self.checkpoint_interval
                    ]
                    == self.step_size[start:end, self.checkpoint]
                )
                condition2_2 = (
                    self.best_loss[
                        start:end, self.checkpoint - self.checkpoint_interval
                    ]
                    == self.best_loss[start:end, self.checkpoint]
                )
                condition2 = torch.logical_and(condition2_1, condition2_2)
                condition = torch.logical_and(condition1, condition2)

                self.step_size[start:end, i] = self.step_size[start:end, i - 1] * (
                    1 - condition * 0.5
                )
                self.checkpoint_interval -= self.checkpoint_decay
                self.checkpoint += max(self.checkpoint_interval, self.checkpoint_min)
        else:
            self.step_size[start:end, i] = self.step_size[start:end, i - 1]

    def _record(self):
        self.step_size = self.step_size.cpu().numpy()
        np.save(f"{self.savedir}/step_size.npy", self.step_size)
