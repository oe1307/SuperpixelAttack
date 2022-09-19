import numpy as np
import torch

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class AutoPGD_Attacker(Attacker):
    """AutoPGD"""

    def __init__(self):
        super().__init__()

    def _recorder(self):
        self.best_loss = torch.zeros(
            (config.n_examples, config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )
        self.current_loss = torch.zeros(
            (config.n_examples, config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )
        self.step_size = torch.zeros(
            (config.n_examples, config.iteration + 1), device=config.device
        )

    def _attack(self, x, y):
        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        self.step_size[self.start : self.end, 0] = (
            torch.ones(x.shape[0]) * config.step_size
        )
        checker = {
            "checkpoint": int(0.22 * config.iteration),
            "checkpoint_interval": int(0.22 * config.iteration),
            "checkpoint_decay": int(0.03 * config.iteration),
            "checkpoint_min": int(0.06 * config.iteration),
            "previous_checkpoint": 0,
            "loss_update": torch.zeros(
                x.shape[0], dtype=torch.uint8, device=config.device
            ),
        }

        _x_adv = x.clone()
        x_adv = x.clone().requires_grad_()
        loss = self.criterion(self.model(x_adv), y).sum().clone()
        self.num_forward += x_adv.shape[0]
        grad = torch.autograd.grad(loss, [x_adv])[0].clone()
        self.num_backward += x_adv.shape[0]
        step_size = self.step_size[self.start : self.end, 0].view(-1, 1, 1, 1)
        logger.debug(
            f"step_size ( iter=1 ) : {step_size.min():.4f} ~ {step_size.max():.4f}"
        )
        x_adv = (x_adv + step_size * torch.sign(grad)).clamp(lower, upper).clone()
        del grad
        loss = self.robust_acc(x_adv, y).sum().clone()

        for iter in range(1, config.iteration):
            checker = self.step_size_manager(iter, checker)
            grad = torch.autograd.grad(loss, [x_adv])[0].clone()
            self.num_backward += x_adv.shape[0]
            step_size = self.step_size[self.start : self.end, iter].view(-1, 1, 1, 1)
            logger.debug(
                f"step_size ( iter={iter + 1} ) : "
                + f"{step_size.min():.4f} ~ {step_size.max():.4f}"
            )
            z = (
                (x_adv + step_size * torch.sign(grad))
                .clamp(lower, upper)
                .detach()
                .clone()
            )
            del grad
            x_adv, _x_adv = (
                x_adv
                + config.alpha * (z - x_adv)
                + (1 - config.alpha) * (x_adv - _x_adv)
            ).clamp(lower, upper).clone(), x_adv.detach().clone()
            assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
            loss = self.robust_acc(x_adv, y).sum().clone()
        return x_adv

    @torch.inference_mode()
    def step_size_manager(self, iter, checker):
        if iter == checker["checkpoint"]:
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
        self.num_forward = (
            self.num_forward * self.success_iter.sum() / (config.n_examples * 101)
        )
        self.num_backward = (
            self.num_backward * self.success_iter.sum() / (config.n_examples * 100)
        )

        np.save(f"{config.savedir}/step_size.npy", self.step_size)

        msg = (
            f"num_forward = {self.num_forward}\n"
            + f"num_backward = {self.num_backward}"
        )
        print(msg, file=open(config.savedir + "/summary.txt", "a"))
        logger.info(msg + "\n")
