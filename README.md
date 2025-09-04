# Brain Tumor Detection

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25--98%25-brightgreen)](https://github.com/yourusername/your-repo)

## Overview
Brain Tumor Detection is an AI-powered project that detects and classifies brain tumors from MRI images using deep learning. Early detection is crucial for effective treatment, and this automated solution helps doctors make faster and more accurate decisions.

## Problem Statement
Manual diagnosis of brain tumors from MRI scans is time-consuming and prone to human error. This project leverages Convolutional Neural Networks (CNNs) to automate tumor detection and classification, improving both speed and accuracy.

## Dataset
- MRI images categorized into:
  - Pituitary Tumor
  - Meningioma Tumor
  - Glioma Tumor
  - No Tumor
- Data split into training and testing sets.
- Preprocessing steps:
  - Resizing images
  - Normalization (scaling pixel values between 0–1)
  - Data augmentation (rotation, flipping, scaling) for model robustness

## Methodology
1. **Exploratory Data Analysis (EDA)**
   - Understanding dataset distribution
   - Visualizing MRI samples
2. **Preprocessing**
   - Convert images into tensors
   - Scale pixel values
   - Augment training images
3. **Model Architecture**
   - Convolution + ReLU layers
   - MaxPooling layers
   - Dropout for regularization
   - Fully connected Dense layers with Softmax for classification
4. **Training**
   - Loss Function: Categorical Cross-Entropy
   - Optimizer: Adam

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion matrix to analyze class-wise performance

## Results
- CNN achieved **95–98% accuracy** on test data
- Effectively differentiates between different tumor types and normal cases


## Conclusion
- Demonstrates how deep learning can assist in medical imaging
- Future improvements:
  - Implement advanced architectures like ResNet or VGG
  - Apply transfer learning for better performance
- Can be integrated into clinical decision support systems to aid radiologists

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV / NumPy / Pandas / Matplotlib
- Jupyter Notebook

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Gnanendra494/Brain-Tumor-Detection.git
2.	Install dependencies:
    ```bash
    pip install -r requirements.txt
3.	Run the training script:
     ```bash
    python train_model.py
4.	Test the model:
     ```bash
    python test_model.py

Author

K.Gnanendra Reddy 
	 •	LinkedIn: https://www.linkedin.com/in/gnanendra494/
	 •	Email: gnanendragnana143@gmail.com

## License

This project is licensed under the MIT License.

