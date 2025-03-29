import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image.
    
    :param image_path: Path to the image file.
    :param target_size: Desired image size (height, width).
    :return: Preprocessed image tensor.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

def augment_image(image):
    """
    Apply data augmentation to an image.
    
    :param image: Input image tensor.
    :return: Augmented image tensor.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2
    )
    image = np.expand_dims(image, axis=0)
    augmented_image = next(datagen.flow(image, batch_size=1))[0]
    return augmented_image

def generate_random_image(target_size=(224, 224)):
    """
    Generate a random image with noise.
    
    :param target_size: Desired image size (height, width).
    :return: Randomly generated image tensor.
    """
    random_image = np.random.rand(target_size[0], target_size[1], 3).astype(np.float32)
    return random_image

def prepare_dataset(image_paths, batch_size=32):
    """
    Prepare a dataset for training.
    
    :param image_paths: List of image file paths.
    :param batch_size: Number of images per batch.
    :return: TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: tf.py_function(func=preprocess_image, inp=[x], Tout=tf.float32))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def visualize_image(image, title="Image"):
    """
    Display an image using matplotlib.
    
    :param image: Image tensor.
    :param title: Title of the plot.
    """
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Example usage
if __name__ == "__main__":
    random_img = generate_random_image()
    visualize_image(random_img, title="Randomly Generated Image")
