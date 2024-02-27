import os
import torch


def save_model(save_directory, model,  number_of_epochs, additional_text='', augmentation=''):
    os.makedirs(save_directory, exist_ok=True)
    model_name = model.__class__.__name__

    torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}{additional_text}_model_epochs_{number_of_epochs}{augmentation}.pth"))


def load_model(load_directory, model, number_of_epochs, additional_text='', augmentation=''):
    model_name = model.__class__.__name__

    model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}{additional_text}_model_epochs_{number_of_epochs}{augmentation}.pth"), map_location=torch.device('cpu')))