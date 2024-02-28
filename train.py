import os
import torch
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.dataset import get_dataloaders
from utils.saving_loading_models import save_model
import torchvision.models as models
import torch.backends.cudnn as cudnn
from utils.model_selector import model_selector
from utils import config


def plot_and_save_training_results(data, label, num_epochs, save_path):
    plt.plot(range(1, num_epochs + 1), data['train'], label='train')
    plt.plot(range(1, num_epochs + 1), data['val'], label='validation')
    plt.title(f'Training and validation {label}')
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.legend()

    plt.savefig(os.path.join(save_path, f"{label}.png"))
    plt.close()

    print(f"Training graph saved to {save_path}")


def train_val_step(dataloader, model, loss_function, optimizer, device):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    running_loss = 0
    correct = 0
    total = 0

    for data in dataloader:
        image, labels = data
        image, labels = image.to(device), labels.to(device)
        outputs = model(image)
        loss = loss_function(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if optimizer is not None:
            optimizer.zero_grad()
            # perform backpropagation
            loss.backward()
            # update the model parameters
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader.dataset), correct / total


def train(model, train_loader, val_loader, device, num_epochs=5, additional_text='', augmentation=''):

    graphs_and_logs_save_directory = './training_graphs_and_logs'
    model_name = model.__class__.__name__
    graphs_and_logs_save_path = os.path.join(graphs_and_logs_save_directory, f"{model_name}_epochs_{num_epochs}")

    # define criterion and optimizer for training
    criterion = torch.nn.CrossEntropyLoss()

    model_selector(model, 2)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=0.9)

    accuracy_tracking = {'train': [], 'val': []}
    loss_tracking = {'train': [], 'val': []}
    best_loss = float('inf')

    # Early stopping
    patience = 5
    minDelta = 0.01
    currentPatience = 0

    os.makedirs(graphs_and_logs_save_path, exist_ok=True)

    log_file_path = os.path.join(graphs_and_logs_save_path, 'log.txt')
    log_file = open(log_file_path, 'a')

    # we iterate for the specified number of epochs
    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        training_loss, training_accuracy = train_val_step(train_loader, model, criterion, optimizer, device)
        loss_tracking['train'].append(training_loss)
        accuracy_tracking['train'].append(training_accuracy)

        with torch.inference_mode():
            val_loss, val_accuracy = train_val_step(val_loader, model, criterion, None, device)
            loss_tracking['val'].append(val_loss)
            accuracy_tracking['val'].append(val_accuracy)
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                current_patience = 0

                # Save the model when a new best loss is found
                print('Saving best model')
                save_model('./trained_models', model, num_epochs, additional_text=additional_text,
                           augmentation=augmentation)
            else:
                current_patience += 1

                # Early stopping
            if current_patience >= patience:
                print('Early stopping triggered.')
                break

        print(f'Training accuracy: {training_accuracy:.6}, Validation accuracy: {val_accuracy:.6}')
        print(f'Training loss: {training_loss:.6}, Validation loss: {val_loss:.6}')

        # Append the information to the log file
        log_file.write(f"Epoch {epoch + 1}: "
                       f'Training accuracy: {training_accuracy:.6}, Validation accuracy: {val_accuracy:.6}, '
                       f'Training loss: {training_loss:.6}, Validation loss: {val_loss:.6}\n')

    print('\nFinished Training\n')

    plot_and_save_training_results(loss_tracking, 'loss', num_epochs, graphs_and_logs_save_path)
    plot_and_save_training_results(accuracy_tracking, 'accuracy', num_epochs, graphs_and_logs_save_path)

    log_file.close()


if __name__ == "__main__":
    # model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # model = models.mobilenet_v3_large()

    model = models.resnet18(weights='IMAGENET1K_V1')
    # model = models.resnet34(weights='IMAGENET1K_V1')
    # model = models.mobilenet_v3_large(pretrained=True)
    # model = models.mobilenet_v2(pretrained=True)

    train_loader, val_loader, _ = get_dataloaders()
    train(model, train_loader, val_loader, config.DEVICE, num_epochs=config.NUMBER_OF_EPOCHS)