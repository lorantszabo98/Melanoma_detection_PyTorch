import torch
import torchvision
from torch import nn


def model_selector(model, number_of_classes):
    model_name = model.__class__.__name__

    if model_name == 'ResNet':
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)

    elif model_name == 'MobileNetV3':
        model.classifier[-1] = nn.Linear(1280, number_of_classes)

    elif model_name == 'MobileNetV2':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, number_of_classes)

    elif model_name == 'EfficientNet':
        model.classifier[1] = nn.Linear(1280, number_of_classes)

    # If you want to add a new model type, you can do it here

    else:
        print(f'[ERROR]: {model_name} is not a valid model name. Please check the model initialization!')
