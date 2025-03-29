import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import jaccard_score, f1_score, accuracy_score

class RandomSamplingTest:
    def __init__(self, model, test_images, test_masks, attention_maps=None, num_samples=5):
        self.model = model
        self.test_images = test_images
        self.test_masks = test_masks
        self.attention_maps = attention_maps
        self.num_samples = num_samples
        
    def sample_random_images(self):
        indices = random.sample(range(len(self.test_images)), self.num_samples)
        return [(self.test_images[i], self.test_masks[i], i) for i in indices]
    
    def predict(self, image):
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.model.predict(image)[0]
        return (prediction > 0.5).astype(np.uint8)  # Binarize the output
    
    def compute_metrics(self, y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        dice = f1_score(y_true_flat, y_pred_flat)
        iou = jaccard_score(y_true_flat, y_pred_flat)
        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        return {'Dice': dice, 'IoU': iou, 'Accuracy': accuracy}
    
    def visualize_results(self):
        sampled_data = self.sample_random_images()
        fig, axes = plt.subplots(self.num_samples, 4, figsize=(12, 4 * self.num_samples))
        
        for i, (image, true_mask, idx) in enumerate(sampled_data):
            pred_mask = self.predict(image)
            metrics = self.compute_metrics(true_mask, pred_mask)
            
            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title(f"Sample {idx} - Original")
            
            axes[i, 1].imshow(true_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title(f"Prediction (IoU: {metrics['IoU']:.2f})")
            
            if self.attention_maps is not None:
                attention_map = self.attention_maps[idx]
                axes[i, 3].imshow(attention_map, cmap='jet', alpha=0.5)
                axes[i, 3].set_title("Attention Map")
            else:
                axes[i, 3].axis('off')
                
        plt.tight_layout()
        plt.show()
    
    def run_evaluation(self):
        sampled_data = self.sample_random_images()
        all_metrics = []
        
        for image, true_mask, _ in sampled_data:
            pred_mask = self.predict(image)
            metrics = self.compute_metrics(true_mask, pred_mask)
            all_metrics.append(metrics)
        
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
        print("Average Metrics:")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
        
        self.visualize_results()
        return avg_metrics

# Example usage:
# model = tf.keras.models.load_model('**********.h5')
# tester = RandomSamplingTest(model, test_images, test_masks, attention_maps, num_samples=5)
# tester.run_evaluation()