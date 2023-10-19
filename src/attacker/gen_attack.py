import torch
from advertorch.attacks import LinfGenAttack
from torch import Tensor

from base import Attacker
from utils import ProgressBar, config_parser

config = config_parser()


class GenAttack(Attacker):
    def __init__(self):
        super().__init__()
        torch.use_deterministic_algorithms(False)  # for advertorch

    def _attack(self, data: Tensor, label: Tensor):
        attacker = LinfGenAttack(self.model, config.epsilon)
        torch.set_default_device(config.device)
        adv_data = []
        pbar = ProgressBar(len(data), "idx", color="cyan")
        for idx in range(len(data)):
            x = data[idx].unsqueeze(0).to(config.device)
            x_adv = attacker.perturb(x).cpu()
            adv_data.append(x_adv)
            pbar.step()
        adv_data = torch.cat(adv_data)
        torch.set_default_tensor_type(torch.FloatTensor)
        pbar.end()
        return adv_data
