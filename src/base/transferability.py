import torch

from utils import config_parser, setup_logger

from .base_attacker import Attacker
from .get_model import get_model

logger = setup_logger(__name__)
config = config_parser()


class Transfer(Attacker):
    def __init__(self):
        super().__init__()

    @torch.enable_grad()
    def transfer(self, x, y):
        assert hasattr(config, "transfer_model")
        model = get_model(
            config.transfer_model_container,
            config.transfer_model,
            x.shape[0],
            model_dir="../storage/model",
        )

        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        x_adv = x.clone().requires_grad_()

        loss = self.criterion(model(x_adv), y).sum().clone()
        self.num_forward += x_adv.shape[0]
        grad = torch.autograd.grad(loss, [x_adv])[0].clone()
        self.num_backward += x_adv.shape[0]
        x_adv = (x_adv + config.epsilon * torch.sign(grad)).clamp(lower, upper).clone()
        del grad
        assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
        return x_adv.detach().clone()