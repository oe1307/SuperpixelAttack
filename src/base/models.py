from typing import Callable, Union

import robustbench
import torchvision
from robustbench.data import get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from torch.nn import Module
from torchvision import transforms as T

from utils import config_parser, counter, setup_logger

logger = setup_logger(__name__)
config = config_parser()


def get_model(model_dir="../storage/model") -> Union[Module, Callable]:

    if config.model_name in (
        "resnet50",
        "vgg16_bn",
    ):
        model = torchvision.models.get_model(config.model_name, weights="DEFAULT")
        transform = get_transform(config.model_name)

    elif config.model_name in (
        "Wong2020Fast",
        "Engstrom2019Robustness",
        "Salman2020Do_R18",
        "Salman2020Do_R50",
        "Salman2020Do_50_2",
    ):
        model = robustbench.load_model(config.model_name, model_dir, "imagenet")
        transform = get_preprocessing(
            BenchmarkDataset.imagenet, ThreatModel(config.norm), config.model_name, None
        )

    else:
        raise NotImplementedError(config.model_name)

    model = model.to(config.device)
    model.eval()
    config.batch_size = config.model[config.model_name]
    model.forward = counter(model.forward)
    logger.debug("Loaded model")
    return model, transform


def get_transform():
    if config.model_name == "resnet50":
        transform = T.Compose([T.Resize(232), T.CenterCrop(224), T.ToTensor()])
    elif config.model_name == "vgg16_bn":
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    else:
        raise NotImplementedError(config.model_name)
    return transform
