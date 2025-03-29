import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load images and split them into training, validation, and test sets.
    
    :param dataset_path: Path to the dataset directory.
    :param test_size: Proportion of dataset to be used for testing.
    :param val_size: Proportion of dataset to be used for validation.
    :param random_state: Seed for reproducibility.
    :return: Splitted datasets (train, val, test) along with labels.
    """
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                image_paths.append(image_path)
                labels.append(class_to_index[class_name])
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Split into train, test, and validation sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size, random_state=random_state, stratify=train_labels
    )
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_to_index
