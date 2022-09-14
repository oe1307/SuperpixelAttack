from robustbench import load_model

from utils import config_parser


def get_model(model_container: str, model_name: str, batch_size: int):
    """Get model from robustbench and set batch size."""
    if model_container == "robustbench":
        model = robustbench_model(model_name)
    model.name = model_name
    model.batch_size = batch_size
    return model


def robustbench_model(model_name):
    """Get model from robustbench"""
    config = config_parser.config
    model = load_model(
        model_name, model_dir="../storage/model", dataset=config.dataset
    ).to(config.device)
    model.eval()
    return model
