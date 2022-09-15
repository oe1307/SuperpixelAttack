import torch
from robustbench.data import CustomImageFolder, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils import config_parser

config = config_parser()


def load_imagenet(model_name):
    """Load ImageNet data."""
    assert config.n_examples <= 5000
    transform = get_preprocessing(
        BenchmarkDataset.imagenet, ThreatModel(config.norm), model_name, None
    )
    dataset = CustomImageFolder("../storage/data/imagenet/val", transform=transform)
    img = list()
    label = list()
    for index in range(config.n_examples):
        x, y = dataset.__getitem__(index)[:2]
        img.append(x.unsqueeze(0))
        label.append(y)
    img = torch.vstack(img)
    label = torch.tensor(label)
    return img, label
