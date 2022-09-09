import torch
from robustbench.data import PREPROCESSINGS, CustomImageFolder, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel


def load_imagenet(n_examples, transforms_test):
    """Load ImageNet data."""
    assert n_examples <= 5000
    prepr = get_preprocessing(
        BenchmarkDataset(self.config.dataset),
        ThreatModel(threat_model),
        model_name,
        PREPROCESSINGS,
    )
    dataset = CustomImageFolder("../storage/imagenet/val", transform=transforms_test)

    x_test = list()
    y_test = list()

    for index in range(n_examples):
        x, y, _ = dataset.__getitem__(index)
        x_test.append(x.unsqueeze(0))
        y_test.append(y)

    x_test = torch.vstack(x_test)
    y_test = torch.tensor(y_test).type(torch.long)

    return x_test, y_test
