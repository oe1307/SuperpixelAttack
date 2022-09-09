from torch.nn import CrossEntropyLoss

from utils import config_parser


def criterion():
    config = config_parser.config

    if config.criterion == "cw":
        return CWLoss()
    elif config.criterion == "ce":
        return CrossEntropyLoss(reduction="none")
    elif config.criterion == "dlr":
        return DLRLoss()
    else:
        raise NotImplementedError


class CWLoss:
    def __init__(self):
        self.name = "cw_loss"

    def forward(self, logits, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        z_y = x[np.arange(x.shape[0]), y]
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
        value_true_maximum = z_y - max_zi
        loss = (-1.0) * value_true_maximum
        if output_target_label:
            target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
            return loss.reshape(-1), target
        else:
            return loss.reshape(-1)


class DLRLoss:
    def __init__(self):
        self.name = "dlr_loss"

    def forward(self, logits, y):
        return dlr_loss(logits, y)
