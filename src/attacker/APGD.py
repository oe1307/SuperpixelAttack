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
            (config.n_examples, config.iteration + 1), device=config.device
        )

    def _attack(self, x, y):
        config = config_parser.config
        upper = (x + config.epsilon).clamp(0, 1).detach().clone()
        lower = (x - config.epsilon).clamp(0, 1).detach().clone()
        self.step_size[self.start : self.end, 0] = (
            torch.ones(self.end - self.start) * config.step_size
        )
        checker = {
            "checkpoint": int(0.22 * config.iteration),
            "checkpoint_interval": int(0.22 * config.iteration),
            "checkpoint_decay": int(0.03 * config.iteration),
            "checkpoint_min": int(0.06 * config.iteration),
            "best_loss_update": torch.zeros(
                self.end - self.start, dtype=torch.uint8, device=config.device
            ),
        }

        x_adv = x.detach().clone()
        x_adv.requires_grad_()
        logits = self.model(x_adv)
        loss = self.criterion(logits, y).sum()
        self.num_forward += x_adv.shape[0]

        for iter in range(config.iteration):
            grad = torch.autograd.grad(loss, [x_adv])[0]
            step_size = self.step_size[self.start : self.end, iter].view(-1, 1, 1, 1)
            logger.debug(f"step_size: {step_size.min():.4f} - {step_size.max():.4f}")
            x_adv = x_adv - step_size * torch.sign(grad)
            x_adv = torch.clamp(x_adv, lower, upper)
            loss = self.robust_acc(x_adv, y).sum()
            checker = self.step_size_manager(iter, checker)

    @torch.inference_mode()
    def step_size_manager(self, iter, checker):
        if iter == checker["checkpoint"]:
            config = config_parser.config
            threshold = (
                config.rho
                * checker["checkpoint_interval"]
                * torch.ones(self.end - self.start, device=config.device)
            )
            condition1 = checker["best_loss_update"] < threshold

            condition2_1 = self.step_size[
                self.start : self.end,
                checker["checkpoint"] - checker["checkpoint_interval"],
            ]
            condition2_2 = (
                self.best_loss[
                    self.start : self.end,
                    checker["checkpoint"] - checker["checkpoint_interval"],
                ]
                == self.best_loss[self.start : self.end, checker["checkpoint"]]
            )
            condition2 = torch.logical_and(condition2_1, condition2_2)

            condition = torch.logical_and(condition1, condition2)
            self.step_size[self.start : self.end, iter + 1] = self.step_size[
                self.start : self.end, iter
            ] * (1 - condition * 0.5)
            checker["checkpoint_interval"] = max(
                checker["checkpoint_interval"] - checker["checkpoint_decay"],
                checker["checkpoint_min"],
            )
            checker["checkpoint"] = (
                checker["checkpoint"] + checker["checkpoint_interval"]
            )
            checker["best_loss_update"] = torch.zeros(
                self.end - self.start, dtype=torch.uint8, device=config.device
            )

        else:
            self.step_size[self.start : self.end, iter + 1] = self.step_size[
                self.start : self.end, iter
            ]
            checker["best_loss_update"] += (
                self.current_loss[self.start : self.end, iter + 1]
                < self.current_loss[self.start : self.end, iter]
            )
        return checker

    def _record(self):
        self.step_size = self.step_size.cpu().numpy()
        np.save(f"{self.savedir}/step_size.npy", self.step_size)
