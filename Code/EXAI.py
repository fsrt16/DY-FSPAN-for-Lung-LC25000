import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random

class GradCAM:
    def __init__(self, model: Model):
        """
        Initialize the GradCAM class.
        
        :param model: Pretrained Keras model
        """
        self.model = model
    
    def compute_heatmap(self, image: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        :param image: Input image
        :param layer_name: Target convolutional layer for Grad-CAM
        :return: Heatmap array
        """
        image_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
        grad_model = Model(inputs=self.model.input, outputs=[self.model.get_layer(layer_name).output, self.model.output])
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image_tensor)
            class_idx = int(np.argmax(predictions[0]))
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.
        
        :param image: Original image
        :param heatmap: Grad-CAM heatmap
        :param alpha: Intensity factor for blending
        :return: Superimposed image with heatmap
        """
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return superimposed

class AttentionVisualizer:
    def __init__(self, model: Model, attention_layers: list):
        """
        Initialize the Attention Visualizer class.
        
        :param model: Pretrained Keras model
        :param attention_layers: List of layer names to extract attention maps
        """
        self.model = model
        self.attention_layers = attention_layers
        self.attention_model = Model(inputs=model.input, outputs=[model.get_layer(layer).output for layer in attention_layers])
    
    def compute_attention_maps(self, image: np.ndarray) -> list:
        """
        Extract attention maps from specified layers.
        
        :param image: Input image
        :return: List of attention maps from selected layers
        """
        image = np.expand_dims(image, axis=0) / 255.0
        return self.attention_model.predict(image)
    
    def visualize_attention(self, image: np.ndarray, attention_maps: list):
        """
        Visualize attention maps overlaid on the original image.
        
        :param image: Original image
        :param attention_maps: List of extracted attention maps
        """
        fig, axs = plt.subplots(1, len(self.attention_layers) + 1, figsize=(18, 6))
        axs[0].imshow(image)
        axs[0].set_title("Original Image", fontsize=14, fontweight="bold")
        axs[0].axis("off")
        
        for i, (att_map, layer_name) in enumerate(zip(attention_maps, self.attention_layers)):
            att_map = np.mean(att_map[0], axis=-1)
            att_map = cv2.resize(att_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            axs[i + 1].imshow(image)
            axs[i + 1].imshow(att_map, cmap="jet", alpha=0.4)
            axs[i + 1].set_title(f"Attention: {layer_name}", fontsize=12, fontweight="bold")
            axs[i + 1].axis("off")
        
        plt.tight_layout()
        plt.show()

class PCAFeatureAnalyzer:
    def __init__(self, feature_matrix: np.ndarray):
        """
        Initialize the PCA feature analyzer.
        
        :param feature_matrix: Matrix of extracted features
        """
        self.feature_matrix = feature_matrix
        self.scaler = StandardScaler()
        self.pca = PCA()
    
    def compute_pca(self) -> np.ndarray:
        """
        Perform PCA on feature matrix and return principal components.
        
        :return: Transformed PCA feature matrix
        """
        scaled_features = self.scaler.fit_transform(self.feature_matrix)
        return self.pca.fit_transform(scaled_features)
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Compute statistical metrics (variance, explained variance, etc.).
        
        :return: DataFrame of PCA statistics
        """
        explained_variance = self.pca.explained_variance_ratio_
        components = np.arange(1, len(explained_variance) + 1)
        return pd.DataFrame({"Component": components, "Explained Variance": explained_variance})

# Example usage
if __name__ == "__main__":
    # Load model
    model = tf.keras.models.load_model("model.h5")
    
    # Load dataset
    image_array = np.load("Images.npy")
    labels = np.load("Labels.npy")
    
    # Select random image
    idx = random.randint(0, len(image_array) - 1)
    image = image_array[idx]
    
    # Grad-CAM
    grad_cam = GradCAM(model)
    heatmap = grad_cam.compute_heatmap(image, "conv2d")
    result = grad_cam.overlay_heatmap(image, heatmap)
    plt.imshow(result)
    plt.title("Grad-CAM Visualization")
    plt.axis("off")
    plt.show()
    
    # Attention Visualization
    attention_layers = ["conv2d", "Y_Block1_relu", "Y_Block2_relu", "Y_Block3_relu"]
    attention_visualizer = AttentionVisualizer(model, attention_layers)
    att_maps = attention_visualizer.compute_attention_maps(image)
    attention_visualizer.visualize_attention(image, att_maps)
    
    # PCA Feature Analysis
    features = np.random.rand(100, 50)  # Example feature matrix
    pca_analyzer = PCAFeatureAnalyzer(features)
    pca_transformed = pca_analyzer.compute_pca()
    stats_df = pca_analyzer.get_statistics()
    print(stats_df)