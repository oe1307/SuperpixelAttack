import numpy as np
import torch
from torch import Tensor

from base import Attacker
from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class GenAttacker(Attacker):
    def __init__(self):
        super().__init__()

    def recorder(self):
        super().recorder()
        self.best_loss = torch.zeros(
            (config.n_examples, config.population * config.iteration + 1),
            dtype=torch.float16,
            device=config.device,
        )
        self.current_loss = torch.zeros(
            (config.n_examples, config.population * config.iteration + 1),
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
        assert 0 <= config.rho <= 1
        rho = torch.ones(x.shape[0], device=config.device) * config.rho
        rho_min = torch.ones(x.shape[0], device=config.device) * config.rho_min
        alpha = (
            torch.ones(x.shape[0], dtype=torch.float16, device=config.device)
            * config.alpha
        )
        alpha_min = (
            torch.ones(x.shape[0], dtype=torch.float16, device=config.device)
            * config.alpha_min
        )
        num_plateaus = torch.zeros(x.shape[0], dtype=torch.int8, device=config.device)
        upper = (x + config.epsilon).clamp(0, 1).clone()
        lower = (x - config.epsilon).clamp(0, 1).clone()

        # Create initial generation
        population = []
        for _ in range(config.population):
            is_upper = torch.randint_like(x, 0, 2, dtype=torch.bool)
            x_adv = torch.where(is_upper, upper, lower)
            population.append(x_adv)
        population = torch.stack(population)

        for iter in range(config.iteration):
            fitness = []
            for x_adv in population:
                fitness.append(self.robust_acc(x_adv, y))
            fitness = torch.stack(fitness)

            next_population = []
            # find the elite members
            elite_idx = fitness.argmax(dim=0)
            next_population.append(population[elite_idx, torch.arange(x.shape[0])])

            # compute selection probability
            prob = torch.softmax(fitness / config.temperature, dim=0).cpu().numpy()
            for _ in range(config.population - 1):
                parent_idx = np.array(
                    [
                        np.random.choice(config.population, p=prob[:, i], size=2)
                        for i in range(x.shape[0])
                    ]
                )
                parent1 = population[parent_idx[:, 0], torch.arange(x.shape[0])]
                parent1_fitness = fitness[parent_idx[:, 0], torch.arange(x.shape[0])]
                parent2 = population[parent_idx[:, 1], torch.arange(x.shape[0])]
                parent2_fitness = fitness[parent_idx[:, 1], torch.arange(x.shape[0])]
                crossover_prob = parent1_fitness / (parent1_fitness + parent2_fitness)
                crossover_prob = crossover_prob.view(-1, 1, 1, 1).repeat(
                    1, x.shape[1], x.shape[2], x.shape[3]
                )
                child_idx = torch.rand_like(parent1) < crossover_prob
                child = torch.where(child_idx, parent1, parent2)

                # Apply mutations and clipping
                mutation_direction = (
                    2
                    * torch.randint_like(
                        x, 0, 2, dtype=torch.int8, device=config.device
                    )
                    - 1
                )
                next_population.append(
                    (
                        child
                        + (torch.bernoulli(rho) * alpha * config.epsilon).view(
                            -1, 1, 1, 1
                        )
                        * mutation_direction
                    ).clamp(lower, upper)
                )

            # update parameters
            population = torch.stack(next_population)
            del next_population
            rho = torch.max(rho_min, 0.5 * (0.9**num_plateaus))
            alpha = torch.max(alpha_min, 0.4 * (0.9**num_plateaus))

        x_adv = population[0]
        assert torch.all(x_adv <= upper + 1e-6) and torch.all(x_adv >= lower - 1e-6)
        return x_adv

    def record(self):
        super().record()
        self.num_forward = (
            self.num_forward
            * self.success_iter.sum()
            / (config.n_examples * (config.population * config.iteration + 1))
        ).to(torch.int32)
        msg = f"num_forward = {self.num_forward}\n" + "num_backward = 0"
        print(msg, file=open(config.savedir + "/summary.txt", "a"))
        logger.info(msg + "\n")
