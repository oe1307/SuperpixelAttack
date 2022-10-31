import art
import numpy as np
import torch
from art.estimators.classification import PyTorchClassifier
from torch import Tensor
from yaspin import yaspin

from Base import Attacker, get_criterion
from Utils import change_level, config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


class SquareAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = config.steps

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        criterion = get_criterion()
        model = PyTorchClassifier(
            self.model,
            criterion,
            input_shape=x.shape[1:],
            nb_classes=config.num_classes,
        )

        attacker = art.attacks.evasion.SquareAttack(
            estimator=model,
            norm=np.inf,
            # adv_criterion: = None,
            # loss: Union[Callable[[np.ndarray, np.ndarray], np.ndarray], None] = None,
            # max_iter: int = 100,
            # eps: float = 0.3,
            # p_init: float = 0.8,
            # nb_restarts: int = 1,
            # batch_size: int = 128,
            # verbose: bool = True,
        )
        raise NotImplementedError(attacker)
