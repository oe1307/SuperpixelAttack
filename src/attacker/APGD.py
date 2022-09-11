import torch
import numpy as np

from base import Attacker
from utils import setup_logger

logger = setup_logger(__name__)


class APGDAttacker(Attacker):
    def __init__(self):
        super().__init__()

    def _recorder(self):
        self.step_size = torch.zeros((self.config.n_examples, self.config.iteration))

    def _attack(self, model, x, y, criterion):
        upper = (x + self.config.epsilon).clamp(0, 1).clone().to(self.config.device)
        lower = (x - self.config.epsilon).clamp(0, 1).clone().to(self.config.device)

        for i in range(self.config.iteration):
            logger.info(f"   iteration {i + 1}")
            self.step_size_manager(i)

            loss = criterion(model(x), y)
            self.current_loss[self.start : self.end, i + 1] = loss.detach().cpu()
            self.best_loss[self.start : self.end, i + 1] = torch.max(
                self.best_loss[self.start : self.end, i], loss.detach().cpu()
            )
            self.num_forward += x.shape[0]

    def step_size_manager(self, i):
        if i == 0:
            self.step_size[:, 0] = self.config.step_size
            self.checkpoint = int(0.22 * self.config.iteration)
            self.checkpoint_interval = int(0.22 * self.config.iteration)
            self.checkpoint_decay = int(0.03 * self.config.iteration)
            self.checkpoint_min = int(0.06 * self.config.iteration)
        elif i == self.checkpoint:
            breakpoint()
            condition1 = 1 < self.config.rho * self.checkpoint_interval
            condition2 = (
                self.step_size[
                    self.start : self.end, self.checkpoint - self.checkpoint_interval
                ]
                == self.step_size[self.start : self.end, self.checkpoint]
                and self.best_loss[
                    self.start : self.end, self.checkpoint - self.checkpoint_interval
                ]
                == self.best_loss[self.start : self.end, self.checkpoint]
            )
            condition = torch.logical_and(condition1, condition2)

            self.step_size[self.start : self.end, i] = self.step_size[
                self.start : self.end, i - 1
            ] * (1 - condition * 0.5)
            self.checkpoint_interval -= self.checkpoint_decay
            self.checkpoint += max(self.checkpoint_interval, self.checkpoint_min)
        else:
            self.checker
            self.step_size[self.start : self.end, i] = self.step_size[
                self.start : self.end, i - 1
            ]

    def _record(self):
        np.save(f"{self.savedir}/step_size.npy", self.step_size)
