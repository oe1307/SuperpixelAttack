import warnings

import torch
from robustbench import load_model
from robustbench.data import get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils import config_parser

config = config_parser()


def get_model(root="../storage/model"):
    torch.hub.set_dir(root)
    dataset = BenchmarkDataset.imagenet
    norm = ThreatModel(config.norm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_model(config.model, root, dataset, norm=norm)
    model = model.eval().to(config.device)
    transform = get_preprocessing(dataset, norm, config.model, None)
    return model, transform
