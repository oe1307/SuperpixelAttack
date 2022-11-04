from robustbench import load_model
from torch.nn import Module
from torchvision import models

from Utils import config_parser, counter

config = config_parser()


def get_model(
    model_container: str, model_name: str, batch_size: int, model_dir: str
) -> Module:
    """Get model from robustbench and set batch size."""
    if model_container == "robustbench":
        model = load_model(model_name, model_dir, config.dataset)
    elif model_container == "pytorch":
        model = load_model_from_pytorch(model_name, model_dir)
    model = model.to(config.device)
    model.eval()
    model.name = model_name
    model.batch_size = batch_size
    model.forward = counter(model.forward)
    return model


def load_model_from_pytorch(model_name, model_dir) -> Module:
    assert config.dataset == "imagenet"
    if model_name == "inception_v3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    return model
