from base import Attacker
from utils import setup_logger

logger = setup_logger(__name__)


class APGD(Attacker):
    def __init__(self):
        super().__init__()

    def _attack(self, model, data, label):
        pass
