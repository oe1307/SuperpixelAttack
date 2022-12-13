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
        preprocessing = model._transform_input
        model.transform_input = False
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
