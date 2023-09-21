import os
from glob import glob

import art
import torch
from art.attacks.evasion import GeoDA as _GeoDA
from art.estimators.classification import PyTorchClassifier
from torch import Tensor

from base import Attacker, get_criterion
from utils import config_parser

config = config_parser()


class GeoDA(Attacker):
    def __init__(self):
        assert config.iter > 250, "GeoDA needs at least 251 iterations"
        art.config._folder = "../../storage/art"
        self.n_class = 1000
        self.norm = "inf"
        self.criterion = get_criterion()

    def _attack(self, data: Tensor, label: Tensor):
        _data = data.numpy()
        art_model = PyTorchClassifier(
            self.model, self.criterion, data.shape[1:], self.n_class
        )
        attacker = _GeoDA(art_model, config.batch_size, self.norm, max_iter=config.iter)
        adv_data = attacker.generate(_data)
        adv_data = torch.from_numpy(adv_data)
        for dct_file in glob("2d_dct_basis_*.npy"):
            os.remove(dct_file)
        upper = (data + config.epsilon).clamp(0, 1)
        lower = (data - config.epsilon).clamp(0, 1)
        adv_data = adv_data.clamp(lower, upper)
        return adv_data
