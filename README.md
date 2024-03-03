Melanoma detection with PyTorch using fine-tuned ResNet18
===============================

This repository contains code for training and evaluating for melanoma detection . The code is organized into three main scripts:

1.  `train.py`: This script is responsible for training models on a custom dataset which includes images of moles. It includes functions for loading the dataset, training the model, saving the trained weights, and plotting the training curves.

2.  `evaluate.py`: This script performs evaluating on the test dataset, it calculates different metrics like, test accuracy, precision, F1-score. It also creates classification reports and heatmap based on the confusion matrix. It also includes inference on 8 random images from the testing dataset.

3.  `dataset.py`: This script loads the training and testing dataset. The training dataset is then split into training and validation data. It also includes different data augmentations.

4.  `config.py`: This file contains all the hyperparameters.

Usage
-----

### 1\. Training the Model

To train the models (transfer learning), execute the following command:

```bash
python train.py
```
This will train ResNet18, for a specified number of epochs using the custom dataset.

### 2\. Evaluate with the trained models

For evaluation using the trained models, run the following command:
```bash
python evalaute.py
```
This script loads the trained model, calculates the metrics, plot the classification report and the confusion matrix heatmap.

Dataset
-------

The melanoma detection dataset used in this project can be downloaded from [here](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data). It includes images in 2 classes: Benign, Malignant. The total number of images is 13900.

Trained Models
--------------
The table below shows the models that have been trained. I only used ResNet18 model and fine-tuned its hyperparameters for the best results. During the process, I tried different data augmentation techniques, regularization techniques to prevent overfitting. I also tried different learning rates, image and batch sizes to achieve the best test accuracy.  

| Model                         | Epochs /25 | Learning rate | Image size | Batch size | Augmentation | Note                                | Test accuracy (%) |
| ----------------------------- | ---------- | ------------- | ---------- | ---------- | ------------ | ----------------------------------- | ----------------- |
| ResNet_models_epochs_25       | 1          | 0.001         | 224        | 64         | No           | Early stopping min delta = 0.001    | 90.55             |
| ResNet_models_epochs_25_aug   | 10         | 0.0001        | 112        | 64         | Yes (small)  |                                     | 88.30             |
| ResNet_models_epochs_25_aug_2 | 10         | 0.0001        | 112        | 64         | Yes (heavy)  |                                     | 89.50             |
| ResNet_models_epochs_25_aug_3 | 17         | 0.001         | 112        | 256        | Yes(small)   |                                     | 87.40             |
| ResNet_models_epochs_25_aug_4 | 16         | 0.0001        | 112        | 256        | Yes(small)   |                                     | 84.14             |
| ResNet_models_epochs_25_aug_6 | 1          | 0.001         | 224        | 64         | No           | Early stopping min delta = 0.000001 | 89.75             |
| ResNet_models_epochs_25_aug_7 | 25         | 0.0001        | 224        | 64         | Yes(small)   | Weight decay = 1e-5                 | 91.25             |
