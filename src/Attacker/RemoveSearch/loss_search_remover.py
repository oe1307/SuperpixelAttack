from .base_remover import Remover


class LossSearchRemover(Remover):
    def __init__(self):
        super().__init__()

    def remove(update_area, targets, forward):
        return targets
