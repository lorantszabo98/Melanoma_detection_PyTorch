import torch

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 25
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DATA_PATH = './data'




