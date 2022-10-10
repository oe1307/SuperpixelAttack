import foolbox as fb
import torch
from yaspin import yaspin
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class GenAttacker2(Attacker):
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
        """Generate adversarial examples

        Args:
            x (Tensor): input
            y (Tensor): label

        Parameters:
            epsilon (float): perturbation bound
            alpha (float): mutation range
            rho (float): mutation probability
            population (int): population size
            num_iter (int): number of iterations
            tau (float): temperature

        Returns:
            Tensor: adversarial examples
        """
        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()
        model = fb.PyTorchModel(self.model, bounds=(0, 1), device=config.device)

        logits = self.model(x).detach().clone()
        idx_sorted = logits.sort(dim=1, descending=True)[1]
        target = torch.where(idx_sorted[:, 0] == y, idx_sorted[:, 1], idx_sorted[:, 0])
        criterion = fb.criteria.TargetedMisclassification(target)

        attack = fb.attacks.GenAttack(
            steps=config.steps,
            population=config.population,
            mutation_probability=config.mutation_probability,
            mutation_range=config.mutation_range,
            sampling_temperature=config.sampling_temperature,
        )
        with yaspin(text="Attacking...", color="cyan"):
            x_adv = attack(model, x, criterion, epsilons=[config.epsilon])[1][0]
        self.num_forward += x.shape[0] * (config.population * config.steps - 1)
        self.robust_acc(x_adv, y)
        assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
        return x_adv
