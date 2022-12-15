class Remover:
    def __init__(self, update_area, update_method):
        self.update_area = update_area
        self.update_method = update_method

    def initialize(self, update_area, targets, forward):
        return targets

    def remove(self, update_area, targets, forward):
        return targets
