from robustbench import load_model

from utils import config_parser


def get_model(model_name, batch_size):
    config = config_parser.config
    model = load_model(
        model_name, model_dir="../storage/model", dataset=config.dataset
    ).to(config.device)
    model.name = model_name
    model.batch_size = batch_size
    model.eval()
    return model
