# Towards Automated and Reliable Lung Cancer Detection in Histopathological Images Using DY-FSPAN: A Feature-Summarized Pyramidal Attention Network for Explainable AI

## Overview
Lung cancer remains one of the most challenging diseases to detect at an early stage due to its complex histopathological patterns. This research introduces **DY-FSPAN (Dynamic Feature-Summarized Pyramidal Attention Network)**, an advanced deep learning framework designed for **automated and explainable lung cancer detection** from histopathological images. The model incorporates pyramidal feature extraction, spatial attention, and feature summarization techniques to enhance classification accuracy while providing interpretability through attention visualization.



Overview

Lung cancer remains one of the most challenging diseases to detect at an early stage due to its complex histopathological patterns. This research introduces DY-FSPAN (Dynamic Feature-Summarized Pyramidal Attention Network), an advanced deep learning framework designed for automated and explainable lung cancer detection from histopathological images. The model incorporates pyramidal feature extraction, spatial attention, and feature summarization techniques to enhance classification accuracy while providing interpretability through attention visualization.

Background on Lung Cancer

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Histopathological image analysis plays a crucial role in diagnosing lung cancer, but manual examination is time-consuming and prone to variability. Deep learning-based approaches, particularly convolutional neural networks (CNNs), have demonstrated promising results in automating lung cancer detection. This study introduces DY-FSPAN, an advanced framework integrating attention mechanisms, feature summarization, and contextual masking for improved performance and interpretability.

Performance Metrics and Their Significance in Lung Cancer Detection

Performance Metrics and Their Significance in Lung Cancer Detection

Metric

Significance

Accuracy

Measures the overall correctness of predictions. High accuracy ensures reliable lung cancer classification.

Precision

Represents the proportion of correctly identified positive cases out of total predicted positives. Important for reducing false positives, preventing misdiagnosis.

Recall (Sensitivity)

Measures the model's ability to correctly identify all positive cases. Critical for early cancer detection to avoid false negatives.

F1-Score

Harmonic mean of precision and recall. Useful for balancing false positives and false negatives.

AUC-ROC

Evaluates the model's ability to differentiate between cancerous and non-cancerous cases. A high AUC-ROC score indicates robust performance across different thresholds.

Dice Coefficient (F1-Score for Segmentation)

Measures overlap between predicted and ground truth segmentation. Ensures accurate tumor localization.

Intersection over Union (IoU)

Quantifies how well the predicted segmentation matches the ground truth. Higher IoU reflects better model performance.

Ablation Experiments

A series of ablation experiments were conducted to evaluate the impact of different components on performance.

Model Variant

Attention Mechanism

Y-Block

FSPAN

Feature Masking

IMA

Base Model (ConvNext-Tiny)

✗

✗

✗

✗

✗

Base Model + Attention

✓

✗

✗

✗

✗

Base Model + Y-Block

✓

✓

✗

✗

✗

Base Model + Y-Block + FSPAN

✓

✓

✓

✗

✗

Base Model + Y-Block + FSPAN + Masking

✓

✓

✓

✓

✗

Base Model + Y-Block + FSPAN + IMA

✓

✓

✓

✗

✓

Base Model + Y-Block + FSPAN + Masking + IMA

✓

✓

✓

✓

✓

Proposed DY-FSPAN (Final Model)

✓

✓

✓

✓

✓

Cross-Validation with Different Backbones

The model was trained using various CNN backbones to assess generalizability.

Model

Validation Performance

VGG16

X%

DenseNet121

X%

Inception V3

X%

VGG19

X%

ResNet50

X%

Xception

X%

Inception-ResNet V2

X%

ConvNext Tiny

X%

Optimization Techniques

To enhance the efficiency and robustness of DY-FSPAN, different optimization techniques were explored.

SL No.

Optimization Technique

1

Genetic Algorithm (GA)

2

Particle Swarm Optimization (PSO)

3

Grey Wolf Optimizer (GWO)

4

Dragonfly Algorithm (DA)

5

Ant Colony Optimization (ACO)

6

CHA Proposed Architecture

Experimental Setup for Contextual Masked-Dilation Attention

The table below outlines the different experimental setups used for the attention mechanisms.

Experiment

PCMS Layers Used

Contextual Masked-Dilation Attention

EXP 1

1×1 Conv Only

No

EXP 2

3×3 (D=2), 1×1

No

EXP 3

5×5 (D=3), 3×3 (D=2), 1×1

No

EXP 4

5×5 (D=3), 3×3 (D=2), 1×1

Yes

EXP 5

5×5 (D=3), 3×3 (D=2), 1×1

Yes

EXP 6

5×5 (D=3), 3×3 (D=2), 1×1

Yes




## Methodology
### 1. **Dataset**
We use publicly available **histopathological lung cancer datasets** to train and evaluate the model. The dataset comprises multiple classes, including:
- Normal (Benign)
- Adenocarcinoma (Malignant)
- Squamous Cell Carcinoma (Malignant)

### 2. **Preprocessing**
- **Image resizing** using `cv2.INTER_AREA` for optimal resampling.
- **Normalization** to standardize pixel values.
- **Augmentation** with transformations like rotation, flipping, and color jitter to improve generalization.

### 3. **Model Architecture: DY-FSPAN**
DY-FSPAN integrates:
- **Feature Pyramid Networks (FPN)**: Extracts multi-scale hierarchical features.
- **Spatial Attention Mechanism**: Enhances relevant feature regions for better tumor localization.
- **Dynamic Feature Summarization**: Reduces redundancy and strengthens critical feature representations.
- **Residual Convolutions**: Improves gradient flow for deeper networks.

### 4. **Training and Optimization**
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam with an initial learning rate of `1e-4`
- **Batch Size**: 32
- **Number of Epochs**: 100
- **Early Stopping** to prevent overfitting

### 5. **Evaluation Metrics**
We evaluate DY-FSPAN using:
- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC-AUC Curve**
- **Confusion Matrix**
- **Explainability Analysis**: Using Grad-CAM to highlight decision-making regions

## Explainability & Attention Visualization
- **Grad-CAM heatmaps** provide insights into the model’s decision regions.
- **Feature importance analysis** using attention scores to assess model reliability.

## Results & Comparative Analysis
DY-FSPAN outperforms state-of-the-art models such as ResNet, VGG, and Transformer-based architectures in terms of:
- **Higher classification accuracy (~98.5%)**
- **Better generalization across diverse histopathological images**
- **Improved model interpretability via attention maps**

## Installation & Usage
### Requirements
Ensure you have the following installed:
```bash
pip install -r requirements.txt
```
### Running the Model
1. **Preprocess the dataset**
```bash
python DataPreprocessing.py
```
2. **Train the model**
```bash
python TrainModel.py
```
3. **Evaluate the model**
```bash
python EvaluateModel.py
```
4. **Perform random sampling with attention visualization**
```bash
python RandomSamplingTest.py
```

## Error Analysis & Hyperparameter Tuning
- **Error Analysis**: We analyze incorrect predictions through misclassification mapping.
- **Hyperparameter Tuning**: Conducted using Grid Search for optimal learning rates, batch sizes, and dropout rates.

## Future Work
- **Integration with real-world clinical workflows**
- **Expansion to multi-class and multi-modal lung cancer detection**
- **Enhancement using self-supervised learning**

## Citation
If you use this research in your work, please cite:
```

```

## Contact
For inquiries, please contact: **banerjeetathagat@gmail.com**

