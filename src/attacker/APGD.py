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
            "previous_checkpoint": 0,
            "loss_update": torch.zeros(
                self.end - self.start, dtype=torch.uint8, device=config.device
            ),
        }

        x_adv = x.clone().requires_grad_()
        _x_adv = x.detach().clone()
        loss = self.criterion(self.model(x_adv), y).sum()
        self.num_forward += x_adv.shape[0]
        grad = torch.autograd.grad(loss, [x_adv])[0]
        self.num_backward += x_adv.shape[0]
        step_size = self.step_size[self.start : self.end, 0].view(-1, 1, 1, 1)
        logger.debug(
            f"step_size ( iter=1 ) : {step_size.min():.4f} ~ {step_size.max():.4f}"
        )
        x_adv = (x_adv + step_size * torch.sign(grad)).clamp(lower, upper)
        loss = self.robust_acc(x_adv, y).sum()

        for iter in range(1, config.iteration):
            checker = self.step_size_manager(iter, checker)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            self.num_backward += x_adv.shape[0]
            step_size = self.step_size[self.start : self.end, iter].view(-1, 1, 1, 1)
            logger.debug(
                f"step_size ( iter={iter + 1} ) : {step_size.min():.4f} ~ {step_size.max():.4f}"
            )
            z = (x_adv + step_size * torch.sign(grad)).clamp(lower, upper)
            x_adv, _x_adv = (
                x_adv
                + config.alpha * (z - x_adv)
                + (1 - config.alpha) * (x_adv - _x_adv)
            ).clamp(lower, upper), x_adv.detach().clone()
            loss = self.robust_acc(x_adv, y).sum()

    @torch.inference_mode()
    def step_size_manager(self, iter, checker):
        if iter == checker["checkpoint"]:
            config = config_parser.config
            threshold = config.rho * checker["checkpoint_interval"]
            condition1 = checker["loss_update"] < threshold

            condition2_1 = (
                self.step_size[self.start : self.end, checker["previous_checkpoint"]]
                == self.step_size[self.start : self.end, iter - 1]
            )
            condition2_2 = (
                self.best_loss[self.start : self.end, checker["previous_checkpoint"]]
                == self.best_loss[self.start : self.end, iter - 1]
            )
            condition2 = torch.logical_and(condition2_1, condition2_2)

            condition = torch.logical_or(condition1, condition2)
            self.step_size[self.start : self.end, iter] = self.step_size[
                self.start : self.end, iter - 1
            ] * (1 - condition * 0.5)
            checker["checkpoint_interval"] = max(
                checker["checkpoint_interval"] - checker["checkpoint_decay"],
                checker["checkpoint_min"],
            )
            checker["previous_checkpoint"] = checker["checkpoint"]
            checker["checkpoint"] = (
                checker["checkpoint"] + checker["checkpoint_interval"]
            )
            checker["loss_update"] = torch.zeros(
                self.end - self.start, dtype=torch.uint8, device=config.device
            )

        else:
            self.step_size[self.start : self.end, iter] = self.step_size[
                self.start : self.end, iter - 1
            ]
            checker["loss_update"] += (
                self.current_loss[self.start : self.end, iter - 1]
                < self.current_loss[self.start : self.end, iter]
            )
        return checker

    def _record(self):
        self.step_size = self.step_size.cpu().numpy()
        np.save(f"{self.savedir}/step_size.npy", self.step_size)
