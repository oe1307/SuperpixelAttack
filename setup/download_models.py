import robustbench
import torchvision

pytorch_models = (
    "resnet50",
    "vgg16_bn",
)

robustbench_models = (
    "Wong2020Fast",
    "Engstrom2019Robustness",
    "Salman2020Do_R18",
    "Salman2020Do_R50",
    "Salman2020Do_50_2",
)

for model in pytorch_models:
    print(model)
    torchvision.models.get_model(model, weights="DEFAULT")

for model in robustbench_models:
    print(model)
    robustbench.load_model(model, "../storage/model", "imagenet")
