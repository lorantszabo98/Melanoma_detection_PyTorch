import os
from utils import config
import matplotlib.pyplot as plt
import PIL


def get_data_distribution(data):
    data_path = data
    class_counts = {}

    for class_label in os.listdir(data_path):
        class_path = os.path.join(data_path, class_label)
        number_of_images = len(os.listdir(class_path))
        class_counts[class_label] = number_of_images

    for class_name, counts in class_counts.items():
        print(f"Class: {class_name}, Number of images: {counts}")

    plot_distribution(class_counts)


def plot_distribution(class_counts):
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of images')
    plt.title('Class distribution')
    plt.tight_layout()
    plt.show()


def visualize_classes(data):
    dict_for_five_item_in_each_class = {}
    for class_label in os.listdir(data):
        class_path = os.path.join(data, class_label)
        items_paths = []
        for idx, item in enumerate(os.listdir(class_path)):
            if idx < 5:
                items_paths.append(os.path.join(class_path, item))
        dict_for_five_item_in_each_class[class_label] = items_paths

    plot_5_images_from_each_class(dict_for_five_item_in_each_class)


def plot_5_images_from_each_class(dict_for_five_item_in_each_class):
    for class_labels, image_paths in dict_for_five_item_in_each_class.items():
        plt.figure(figsize=(15, 3))
        for i, image_path in enumerate(image_paths):
            img = PIL.Image.open(image_path)
            plt.subplot(1, len(image_paths), i + 1)
            plt.imshow(img)
            plt.title(class_labels)
            plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    get_data_distribution(os.path.join(config.DATA_PATH, 'train'))
    visualize_classes(os.path.join(config.DATA_PATH, 'train'))