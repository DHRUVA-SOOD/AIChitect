# AIChitect
# GAN-Based Interior Design Generator

## Overview

This project implements a Generative Adversarial Network (GAN) for generating interior design images based on a dataset of room designs. The model consists of a generator and a discriminator, which are trained together to improve the quality of generated images.

## Features

- Uses GAN architecture with a generator and a discriminator.
- Applies data augmentation for better training.
- Saves the trained models to avoid retraining from scratch.
- Implements gradient descent optimization for improved training.
- Supports loading pre-trained models to generate images quickly.

## Requirements

- Python 3.10+
- TensorFlow
- NumPy
- Matplotlib
- PIL (Pillow)
- OpenCV
- Tkinter (for GUI)
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib pillow opencv-python scikit-learn
   ```

## Dataset

- The dataset should be placed in the following directories:
  - `dataset_train/` (for training images)
  - `dataset_test/` (for testing images)

## Training the Model

- If a pre-trained model exists, it will be loaded.
- Otherwise, the model is trained from scratch.
- To train the GAN, run:
  ```bash
  python gan_room_design.py
  ```
- The generator and discriminator models are saved as `generator.h5` and `discriminator.h5`.

## Generating Images

- After training, the model generates new room designs.
- The output images are displayed using Tkinter.
- Run the script to visualize generated images:
  ```bash
  python generate_images.py
  ```

## Model Architecture

### Generator

- Fully connected layers with LeakyReLU activation.
- Batch normalization for stability.
- Output reshaped to 64x64x3 RGB images.

### Discriminator

- Convolutional layers for feature extraction.
- LeakyReLU activation.
- Binary classification (real vs. fake images).

## Optimization

- Generator uses Adam optimizer with a learning rate of `0.0001`.
- Discriminator uses Stochastic Gradient Descent (SGD) with a learning rate of `0.0001` and momentum `0.9`.

## Future Improvements

- Improve generator architecture for higher-quality images.
- Experiment with different loss functions.
- Implement conditional GANs for user-specified styles.

## Author

- Developed by Dhruva Kashyap

