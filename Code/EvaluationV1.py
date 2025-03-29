import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
import cv2

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, class_names):
        """
        Initialize the ModelEvaluator with a trained model and test dataset.
        :param model: Trained deep learning model
        :param X_test: Test images
        :param y_test: True labels for test images
        :param class_names: List of class labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.y_pred = None
    
    def evaluate_model(self):
        """Evaluates the model on test data and prints classification metrics."""
        print("Evaluating Model...")
        predictions = self.model.predict(self.X_test)
        self.y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        print("Classification Report:")
        print(classification_report(y_true, self.y_pred, target_names=self.class_names))
        
        print("ROC-AUC Score:", roc_auc_score(self.y_test, predictions, multi_class='ovr'))
        
    def plot_confusion_matrix(self):
        """Generates and plots a confusion matrix."""
        y_true = np.argmax(self.y_test, axis=1)
        cm = confusion_matrix(y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

class FeatureAnalyzer:
    def __init__(self, X_features):
        """
        Initializes the PCA-based feature analyzer.
        :param X_features: Extracted feature representations
        """
        self.X_features = X_features
        self.pca = None
    
    def perform_pca(self, n_components=10):
        """Performs PCA on the feature set and prints variance ratios."""
        self.pca = PCA(n_components=n_components)
        transformed_features = self.pca.fit_transform(self.X_features)
        
        print("Explained Variance Ratios:")
        print(self.pca.explained_variance_ratio_)
        
        return transformed_features
    
    def plot_pca_variance(self):
        """Plots the cumulative variance explained by the PCA components."""
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid()
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Load test dataset
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    class_names = ["Class A", "Class B", "Class C"]
    
    # Load trained model
    model = tf.keras.models.load_model("trained_model.h5")
    
    evaluator = ModelEvaluator(model, X_test, y_test, class_names)
    evaluator.evaluate_model()
    evaluator.plot_confusion_matrix()
    
    # Extract features for PCA Analysis
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    X_features = feature_extractor.predict(X_test)
    
    analyzer = FeatureAnalyzer(X_features)
    transformed_features = analyzer.perform_pca(n_components=10)
    analyzer.plot_pca_variance()