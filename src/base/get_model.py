from robustbench import load_model

from utils import config_parser

config = config_parser()


def get_model(model_container: str, model_name: str, batch_size: int, model_dir: str):
    """Get model from robustbench and set batch size."""
    if model_container == "robustbench":
        model = load_model(model_name, model_dir, config.dataset).to(config.device)
    model.eval()
    model.name = model_name
    model.batch_size = batch_size
    return model
