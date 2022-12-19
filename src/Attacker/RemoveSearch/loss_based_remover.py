from utils import config_parser

from .base_method import BaseMethod

config = config_parser()


class LossBasedRemover(BaseMethod):
    def __init__(self):
        super().__init__()
