# Deep Learning for Image Classification in Python with CNN

A Convolutional Neural Network (CNN) implementation for classifying chest X-ray images to detect pneumonia, built using Python, Keras, and TensorFlow.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Introduction

Image classification is a fundamental task in computer vision, where the goal is to categorize images into predefined classes. This project implements a CNN model to detect pneumonia from chest X-ray images, aiding in automated medical diagnosis.

## Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset, which contains X-ray images in two categories:
- Normal
- Pneumonia

You can download the dataset from [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Installation

### Prerequisites
- Python 3.9
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

Install the required packages using pip:
```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

## Usage

1. Clone the repository:
   ```bash
   https://github.com/styvetchoumi1998/Machinelles_lernen.git
   cd image-classification-cnn
   ```

2. Download and prepare the dataset:
   - Download the dataset from Kaggle
   - Extract the contents into a `data` directory within the project folder

3. Run the training script:
   ```bash
   python train.py
   ```

4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Model Architecture

The CNN model consists of the following layers:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Fully connected (Dense) layers
- Dropout layers for regularization
- Output layer with softmax activation

## Training

The model training process includes:
- Optimizer: Adam
- Loss function: Categorical cross-entropy
- Dataset split: Training, validation, and test sets
- Built-in mechanisms to prevent overfitting

## Evaluation

Model performance evaluation includes:
- Accuracy and loss metrics on the test set
- Confusion matrix analysis
- Detailed classification reports

## Results

The model achieves the following performance metrics:
- Training Accuracy: 
- Validation Accuracy:
- Test Accuracy: 

These results demonstrate the model's effectiveness in classifying chest X-ray images for pneumonia detection.

## References

- [Deep Learning for Image Classification in Python with CNN](https://github.com/yourusername/image-classification-cnn)
- [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
