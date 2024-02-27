from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils import config
import os

train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    # transforms.RandomResizedCrop(config.IMAGE_SIZE),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.GaussianBlur(kernel_size=3),
    # transforms.RandomAffine(degrees=0, shear=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])

test_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])


def get_dataloaders():
    train_path = os.path.join(config.DATA_PATH, 'train')
    test_path = os.path.join(config.DATA_PATH, 'test')

    full_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)

    # Splitting the train data to train and validation data
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    print(f'Length of train_dataset: {len(train_dataset)}')
    print(f'Length of Validation dataset: {len(val_dataset)}')
    print(f'Length of test dataset: {len(test_dataset)}')

    return train_loader, val_loader, test_loader





