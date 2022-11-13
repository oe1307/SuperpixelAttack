from typing import Callable

from torch import Tensor

from utils import config_parser

config = config_parser()


def get_criterion() -> Callable[[Tensor, Tensor], Tensor]:
    raise NotImplementedError
