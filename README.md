# Siamese Network for One-Shot-Style Pill Identification

## Overview

This is a volunteer project I worked on in collaboration with Kaiser Permanente, under the mentorship of a senior leader specializing in AI. (This is an exploratory side project intended only for research and engineering learning purposes.)

This project implements a **Siamese neural network for one-shot-style pill classification using the ePillID benchmark dataset.** The model learns to compare pill images by mapping them into an embedding space where visually similar pills are closer together and dissimilar pills are further apart, enabling comparison-based identification.

The goal is to address the real-world challenge of pharmaceutical pill recognition, which involves fine-grained visual differences and limited labeled training data. The project demonstrates end-to-end machine learning engineering, from exploring real-world requirements to building the training pipeline and deploying a functional API for pill comparison.


## Key Contributions

**Chloe Chan:**
- Designed and implemented the Siamese neural network architecture
- Built the end-to-end training pipeline in `one_shot_siamese_network.ipynb`
- Implemented data preprocessing and augmentation
- Authored README.md and Dataset_README.md

**Kaiser Mentor:**
- Developed reusable utilities and configuration management in `utils.py`
- Implemented the Flask REST API (`app.py`) for inference
- Built evaluation workflows including threshold optimization and ROC analysis


## Problem Statement

Pharmaceutical pill identification is a critical use case for healthcare applications. The challenge involves:

1. **Large number of classes**: 960+ unique pill types to distinguish
2. **Limited training data**: Only consumer-captured images in real-world conditions
3. **Fine-grained differences**: Small visual variations between pills can indicate different drugs
4. **Practical constraints**: Need to identify pills from single or dual-sided images

Traditional supervised learning approaches struggle with this problem due to the high number of classes and limited samples per class. We address this using **metric learning** with Siamese networks to learn similarity between pill pairs rather than direct classification.

## Data
This project uses the **ePillID Benchmark**, a dataset designed for developing and evaluating computer vision models for pharmaceutical pill identification.

For details, please view the [dataset readme file](Dataset_README.md).

## Project Structure

```
.
├── app.py                              # Flask REST API for pill comparison
├── utils.py                            # Shared utilities, config, and helpers
├── one_shot_siamese_network.ipynb      # Complete training pipeline notebook
├── requirements.txt                    # Python dependencies
├── image_meta.csv                      # Dataset metadata (3,727 samples)
├── architecture.png                    # Visualization of network architecture
│
├── models/
│   └── siamese_model.keras            # Trained Siamese network weights
│
├── outputs/
│   ├── siamese_model.keras            # Backup trained model
│   ├── history.json                   # Training history (loss, validation metrics)
│   ├── evaluation_metrics.json        # Test set performance metrics
│   └── threshold_sweep.csv            # ROC analysis across probability thresholds
│
├── fcn_mix_weight/
│   └── dc_224/                        # Image dataset directory (3,727 JPG files)
│        ├── 0.jpg
│        ├── 10.jpg
│        └── ...
│
└── tests/
    ├── __init__.py                    # Test package marker
    └── test_api.py                    # Unit tests for REST API endpoint
```


## Model Architecture

### Siamese Network Design

A Siamese network learns embeddings by comparing pairs of inputs:

```
Input Image 1 (224×224×3)  →  Shared Encoder Network  →  Embedding Vector (64-dim)
                                                              ↓
                                                         Similarity Score
                                                              ↑
Input Image 2 (224×224×3)  →  Shared Encoder Network  →  Embedding Vector (64-dim)
```

### Encoder Architecture

The shared encoder typically consists of:
- **Convolutional layers**: Feature extraction from raw images
- **Pooling layers**: Spatial dimension reduction and translation invariance
- **Dense layers**: Learning high-level embeddings (64-dimensional)
- **L2 normalization**: Embeddings are normalized to stabilize comparisons between image representations

### Training Mechanism

- **Training Objective**: Binary similarity learning using labeled image pairs. The model learns shared embeddings for both images and predicts whether the pills belong to the same class, using a sigmoid output layer and binary cross-entropy loss
- **Optimization**: Adam optimizer with learnable learning rate
- **Metric**: Siamese networks don't require explicit class labels during inference—only similarity scores

### Key Advantages

1. **One-shot capability**: Can compare any two images without memorizing class-specific patterns
2. **Scalability**: The model can compare new pill images against reference examples without requiring a fixed multiclass output layer
3. **Interpretability**: Similarity scores directly correspond to model confidence

## Key Design Decisions


**Why We Chose Siamese Networks**  
Instead of training a multiclass classifier, the model learns a **similarity function between two pill images**. Each image is passed through a shared encoder to produce an embedding, and the network predicts whether the pair belongs to the same pill class. This approach works better for datasets with **many classes and limited images per class**.

**Shared Encoder Architecture**  
Both images are processed by the same convolutional encoder to ensure they are mapped into the **same embedding space**. The encoder uses convolutional layers, global average pooling, and L2 normalization to produce compact 64-dimensional embeddings for similarity comparison.

**Pair-Based Training Pipeline**  
Training uses dynamically generated **positive and negative image pairs** through a custom data generator. This avoids storing all pair combinations in memory, increases training diversity, and ensures balanced batches during optimization.

**Data Leakage Prevention**  
The dataset is **split into train and test sets before augmentation** to prevent augmented versions of the same image from appearing in both sets. Augmentation is applied only within the training generator.

**Threshold Optimization**  
Because the model outputs a similarity probability, the final decision threshold is **tuned on validation data** rather than fixed at 0.5. This improves the balance between precision and recall for pill matching.

**Code Organization**  
Training, utilities, and inference are separated to improve clarity and reproducibility:

- `one_shot_siamese_network.ipynb` — model training and evaluation  
- `utils.py` — shared configuration and helper functions  
- `app.py` — Flask REST API for inference

## Evaluation & Results

### Performance Metrics

The model is evaluated on a held-out test set using:

- **Accuracy**: Fraction of correct predictions at optimal threshold
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Threshold Optimization**: Grid search across probability values for best separation

### Results Summary

The model achieved strong separation between same-pill and different-pill pairs on the validation set:

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.835 |
| PR-AUC | 0.793 |
| Accuracy | 78.5% |
| Precision | 71.4% |
| Recall | 95.1% |
| F1 Score | 0.816 |
| Optimal Threshold | 0.26 |

The model prioritizes **recall** (95.1%) to catch nearly all matching pairs, with precision at 71.4%. This is desirable in verification workflows where missing a match may be riskier than flagging a false alarm for human review.

Detailed metrics are saved to:
- `outputs/evaluation_metrics.json` — Summary statistics
- `outputs/threshold_sweep.csv` — Performance across thresholds
- Jupyter notebook includes ROC curves and confusion matrices


## Installation

### Prerequisites
- Python 3.8+
- pip or conda 

### Setup

**Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

**Verify installation**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
   ```

## Usage

### Option 1: Training & Evaluation (Jupyter Notebook)

Run the complete training pipeline:

```bash
jupyter notebook one_shot_siamese_network.ipynb
```

The notebook covers:
- Data exploration and preprocessing
- Siamese network architecture implementation
- Model training using metric learning
- Threshold optimization
- Evaluation and visualization

### Option 2: REST API (Inference)

Start the server for real-time inference:

```bash
python app.py
```

Server runs on `http://localhost:5000` (customizable via `host` and `port` parameters).

## References

1. **ePillID Paper**: Usuyama et al. (2020) - [arxiv.org/abs/2005.14288](https://arxiv.org/abs/2005.14288)
2. **Siamese Networks**: Bromley et al. (1993) - [Signature Verification using a Siamese Time Delay Neural Network](http://yann.lecun.com/exdb/publis/pdf/bromley-94.pdf)
3. **Metric Learning**: [Towards Fair and Robust Metric Learning](https://arxiv.org/abs/2501.01025)
4. **TensorFlow/Keras Documentation**: [keras.io](https://keras.io)
5. **NDC Code Lookup**: [drugs.com/imprints.php](https://www.drugs.com/imprints.php)

---

**Author**: Chloe Chan  
**Last Updated**: March 2026  
