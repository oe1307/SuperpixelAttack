import os
from typing import Union

from robustbench.data import CustomImageFolder, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from torch import Tensor
from torch.utils.data import DataLoader

from utils import config_parser, setup_logger

logger = setup_logger(__name__)
config = config_parser()


def load_imagenet(
    model_name: str, data_dir: str = "../storage/data"
) -> Union[Tensor, Tensor]:
    """Load ImageNet data"""
    config.n_classes = 1000
    data_dir = os.path.join(data_dir, "imagenet/val")
    if not os.path.exists(data_dir):
        raise FileNotFoundError("please download imagenet dataset")
    norm = ThreatModel(config.norm)
    transform = get_preprocessing(BenchmarkDataset.imagenet, norm, model_name, None)
    dataset = CustomImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, config.n_examples, shuffle=False, num_workers=config.thread
    )
    logger.debug("Loading ImageNet data...")
    img, label = next(iter(dataloader))[:2]
    logger.debug("Loaded ImageNet data")
    img = img.to(config.device)
    label = label.to(config.device)
    return img, label
