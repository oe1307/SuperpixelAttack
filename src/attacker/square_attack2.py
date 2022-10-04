import torch
from art.attacks.evasion import SquareAttack
from art.estimators.classification import PyTorchClassifier
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SquareAttack2(Attacker):
    def __init__(self):
        super().__init__()

    def recorder(self):
        super().recorder()
        self.best_loss = torch.zeros(
            (config.n_examples, 2),
            dtype=torch.float16,
            device=config.device,
        )
        self.current_loss = torch.zeros(
            (config.n_examples, 2),
            dtype=torch.float16,
            device=config.device,
        )

    @torch.inference_mode()
    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        if config.device != 0:
            logger.warning("adversarial robustness toolbox uses GPU 0")
        model = PyTorchClassifier(
            self.model,
            self.criterion,
            x.shape[1:],
            config.num_classes,
            clip_values=(0, 1),
        )
        attack = SquareAttack(
            model,
            max_iter=config.iteration,
            eps=config.epsilon,
            p_init=config.p_init,
            nb_restarts=config.nb_restarts,
            batch_size=x.shape[0],
            verbose=config.debug,
        )
        x_adv = attack.generate(x.cpu().numpy(), y.cpu().numpy())
        x_adv = torch.from_numpy(x_adv).to(config.device)
        assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
        self.model = self.model.to(config.device)
        self.robust_acc(x_adv, y)
        return x_adv
