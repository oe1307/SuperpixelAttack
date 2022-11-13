from robustbench import load_model
from torch.nn import Module

from utils import config_parser, counter

config = config_parser()


def get_model(
    model_container: str, model_name: str, batch_size: int, model_dir: str
) -> Module:
    """Get model from robustbench and set batch size."""
    model = load_model(model_name, model_dir, config.dataset)

    breakpoint()  # logit -> precision score

    model = model.to(config.device)
    model.eval()
    model.name = model_name
    model.batch_size = batch_size
    model.forward = counter(model.forward)
    return model