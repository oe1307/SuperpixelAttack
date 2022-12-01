from robustbench import load_model
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
    """Get model from robustbench and set batch size."""
    if model_container == "robustbench":
        model = load_model(model_name, model_dir, config.dataset)
    model = model.to(config.device)
    model.eval()
    model.name = model_name
    model.batch_size = batch_size
    model.forward = counter(model.forward)
    logger.debug("Loaded model")
    return model
