import art
import torch
from art.estimators.classification import PyTorchClassifier
from torch import Tensor
from yaspin import yaspin

from Base import Attacker, get_criterion
from Utils import change_level, config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class BoundaryAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = config.steps

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        change_level("art", 40)
        criterion = get_criterion()
        model = PyTorchClassifier(
            self.model,
            criterion,
            input_shape=x.shape[1:],
            nb_classes=config.num_classes,
        )

        # set target class
        logits = self.model(x).detach().clone()
        idx_sorted = logits.sort(dim=1, descending=True)[1]
        target = torch.where(idx_sorted[:, 0] == y, idx_sorted[:, 1], idx_sorted[:, 0])

        attack = art.attacks.evasion.BoundaryAttack(
            estimator=model,
            batch_size=x.shape[0],
            targeted=True,
            delta=config.orthogonal_step_size,
            epsilon=config.target_step_size,
            step_adapt=config.step_adaptation,
            max_iter=config.steps,
            num_trial=config.num_trial,
            sample_size=config.sample_size,
            init_size=config.init_size,
            min_epsilon=0.0,
            verbose=False,
        )
        with yaspin(text="Attacking...", color="cyan"):
            x_adv = attack.generate(x.cpu().numpy(), target.cpu().numpy())

        x_adv = torch.from_numpy(x_adv).to(config.device)
        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        x_adv = x_adv.clamp(lower, upper)
        return x_adv
