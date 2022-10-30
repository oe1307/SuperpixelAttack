import art
from torch import Tensor

from Base import Attacker


class SquareAttack(Attacker):
    def __init__(self):
        super().__init__()
        self.num_forward = 0

    def _attack(self, x: Tensor, y: Tensor) -> Tensor:
        attacker = art.attacks.evasion.SquareAttack(
            # estimator: "CLASSIFIER_TYPE",
            # norm: Union[int, float, str] = np.inf,
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
