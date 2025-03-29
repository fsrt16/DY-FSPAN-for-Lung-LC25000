import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
import itertools
import pandas as pd
import scikitplot as skplt

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, class_names):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.class_names = class_names
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Model Accuracy')
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')
        
        plt.show()
    
    def compute_metrics(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred, target_names=self.class_names)
        print(f'Accuracy: {acc:.4f}\n')
        print('Classification Report:\n', report)
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_roc_curve(self):
        y_prob = self.model.predict_proba(self.X_test)
        skplt.metrics.plot_roc_curve(self.y_test, y_prob)
        plt.title('ROC Curve')
        plt.show()
    
    def analyze_misclassifications(self):
        misclassified_indices = np.where(self.y_pred != self.y_test)[0]
        print(f'Misclassified Samples: {len(misclassified_indices)} out of {len(self.y_test)}')
        return misclassified_indices
    
    def feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), indices, rotation=45)
            plt.show()
        else:
            print('Feature importance not available for this model.')
    
    def visualize_sample_predictions(self, num_samples=5):
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(indices):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(self.X_test[idx].reshape(28, 28), cmap='gray')  # Modify shape if needed
            plt.title(f'True: {self.class_names[self.y_test[idx]]}\nPred: {self.class_names[self.y_pred[idx]]}')
            plt.axis('off')
        plt.show()

# Example usage:
# evaluator = ModelEvaluator(trained_model, X_test, y_test, class_names)
# evaluator.compute_metrics()
# evaluator.plot_confusion_matrix()
# evaluator.plot_roc_curve()
# evaluator.visualize_sample_predictions()