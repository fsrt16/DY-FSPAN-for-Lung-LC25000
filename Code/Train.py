import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from dataloader import DataLoader
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

class Trainer:
    """
    Class to handle model training, validation, and checkpointing.

    Attributes:
        model_path (str): Path to load the pre-trained model.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        callbacks (list): List of callback functions for training.
    """

    def __init__(self, model_path: str, batch_size: int = 32, epochs: int = 20, learning_rate: float = 0.001):
        """
        Initialize the Trainer class with hyperparameters and model loading.

        :param model_path: Path to the saved model (.h5 file).
        :param batch_size: Number of samples per training batch.
        :param epochs: Total training epochs.
        :param learning_rate: Learning rate for optimizer.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Load dataset using DataLoader
        self.data_loader = DataLoader()
        self.X, self.y = self.data_loader.load_data()

        # Encode labels
        self.y = self._encode_labels(self.y)

        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=42
        )

        print(f"Train data    : {self.X_train.shape}")
        print(f"Test data     : {self.X_test.shape}")
        print(f"Train Output  : {self.y_train.shape}")
        print(f"Test Output   : {self.y_test.shape}")

        # Load pre-trained model
        self.model = self._load_model()

        # Define callbacks
        self.callbacks = self._setup_callbacks()

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Convert categorical labels into one-hot encoded vectors.

        :param y: Array of categorical labels.
        :return: One-hot encoded labels.
        """
        y = y.reshape(-1, 1)
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(y)
        return encoder.transform(y).toarray()

    def _load_model(self):
        """
        Load the pre-trained model from the specified path.

        :return: Loaded Keras model.
        """
        try:
            model = load_model(self.model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def _setup_callbacks(self):
        """
        Setup model training callbacks for optimization.

        :return: List of Keras callbacks.
        """
        weight_path = "mammo_result.weights.h5"

        checkpoint = ModelCheckpoint(
            weight_path,
            monitor='val_categorical_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_weights_only=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_categorical_accuracy',
            factor=0.8,
            patience=10,
            verbose=1,
            mode='auto',
            min_lr=0.0001
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        return [checkpoint, reduce_lr, early_stopping]

    def train(self):
        """
        Train the model using the provided dataset.

        :return: Training history object.
        """
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks
        )
        print("Training completed.")
        return history


if __name__ == "__main__":
    model_path = "pretrained_model.h5"  # Update with actual path
    trainer = Trainer(model_path=model_path)
    trainer.train()
