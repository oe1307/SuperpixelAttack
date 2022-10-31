import torch
from torch import Tensor

from Base import Attacker, get_criterion
from Utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class HALS(Attacker):
    def __init__(self):
        super().__init__()
        if (config.dataset == "cifar10" and config.initial_split != 4) or (
            config.dataset == "imagenet" and config.initial_split != 32
        ):
            logger.warning(f"{config.dataset}: split = {config.initial_split}")
        self.criterion = get_criterion()
        self.num_forward = config.steps

    def _attack(self, x_all: Tensor, y_all: Tensor) -> Tensor:
        x_adv_all = []
        for x in x_all:
            self.forward = 0
            upper = (x + config.epsilon).clamp(0, 1).clone()
            lower = (x - config.epsilon).clamp(0, 1).clone()
            split = config.initial_split
            batch, c, h, w = x.shape
            is_upper = torch.zeros(
                (batch, c, h // split, w // split),
                dtype=torch.bool,
                device=config.device,
            )
            x_adv = lower.clone()
            loss = self.criterion(self.model(x_adv), y).clone()
            self.forward += 1
            breakpoint()
            x_adv_all.append(x_adv)
        x_adv_all = torch.stack(x_adv_all)
        return x_adv_all
