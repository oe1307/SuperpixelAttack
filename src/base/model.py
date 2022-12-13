import robustbench
import torchvision
from torch.nn import Module

from utils import config_parser, counter, setup_logger

logger = setup_logger(__name__)
config = config_parser()


def get_model(
    model_container: str,
    model_name: str,
    batch_size: int,
    model_dir: str = "../storage/model",
) -> Module:

    if model_container == "torchvision":
        model = torchvision.models.get_model(model_name, weights="DEFAULT")
        preprocessing = get_prepocessing(model_name)
    elif model_container == "robustbench":
        model = robustbench.load_model(model_name, model_dir, "imagenet")
        preprocessing = None
    else:
        raise NotImplementedError(model_container)

    model = model.to(config.device)
    model.eval()
    model.name, model.batch_size = model_name, batch_size
    model.forward = counter(model.forward)
    logger.debug("Loaded model")
    return model, preprocessing


def get_prepocessing(model_name: str):
    if model_name == "inception_v3":
        return torchvision.models.Inception_V3_Weights.IMAGENET1K_V1.transforms()
    elif model_name == "resnet50":
        return torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    else:
        raise NotImplementedError(model_name)
