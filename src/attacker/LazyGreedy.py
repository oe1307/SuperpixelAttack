import torch

from base import Attacker
from utils import setup_logger

logger = setup_logger(__name__)


class LazyGreedyAttacker(Attacker):
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def _attack(self, model, x, y, criterion):
        upper = (x + self.config.epsilon).clamp(0, 1).clone().to(self.config.device)
        lower = (x - self.config.epsilon).clamp(0, 1).clone().to(self.config.device)

        for i in range(self.config.iteration):
            logger.debug(f"iteration {i}")
            logits = model(x)
            loss = criterion(logits, y)
            self.current_loss[self.start : self.end, i + 1] = loss.detach().cpu()
            self.best_loss[self.start : self.end, i + 1] = torch.max(
                self.best_loss[self.start : self.end, i], loss.detach().cpu()
            )
            self.num_forward += x.shape[0]
