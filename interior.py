import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import glob
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths to datasets
TRAIN_DATASET_PATH = "C:/internshhip/interiordesign/dataset_train/"
TEST_DATASET_PATH = "C:/internshhip/interiordesign/dataset_test/"

# Paths to save models
GENERATOR_MODEL_PATH = "generator.h5"
DISCRIMINATOR_MODEL_PATH = "discriminator.h5"

# Load images
def load_images_from_folder(dataset_path):
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    print(f"Loaded {len(image_paths)} images from {dataset_path}")
    return image_paths

train_images = load_images_from_folder(TRAIN_DATASET_PATH)
test_images = load_images_from_folder(TEST_DATASET_PATH)

if len(train_images) > 0:
    train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)
else:
    print("Error: No images found in the training dataset folder.")
    train_images, val_images = [], []

print(f'Training images: {len(train_images)}')
print(f'Validation images: {len(val_images)}')
print(f'Testing images: {len(test_images)}')

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)

def load_images_for_model(folder, img_size=(64, 64)):
    images = []
    for img_path in load_images_from_folder(folder):
        try:
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img)
            img = (img.astype("float32") - 127.5) / 127.5
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images)

def load_data():
    x_train = load_images_for_model(TRAIN_DATASET_PATH)
    return x_train if len(x_train) > 0 else np.zeros((1, 64, 64, 3))

# Generator model
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(256, input_shape=(100,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64 * 64 * 3, activation="tanh"))
    model.add(layers.Reshape((64, 64, 3)))
    return model

# Discriminator model
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Compile GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential([generator, discriminator])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss="mse")
    return model

# Train GAN
def train_gan(generator, discriminator, gan, epochs=100000, batch_size=64):
    x_train = load_data()
    if x_train.shape[0] == 1:
        print("No valid training images found. Check dataset directory.")
        return
    
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        real_images = next(datagen.flow(real_images, batch_size=batch_size))

        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real_labels)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss[0]}, G Loss: {g_loss}")
            generator.save(GENERATOR_MODEL_PATH)
            discriminator.save(DISCRIMINATOR_MODEL_PATH)

# Load or train models
if os.path.exists(GENERATOR_MODEL_PATH) and os.path.exists(DISCRIMINATOR_MODEL_PATH):
    generator = load_model(GENERATOR_MODEL_PATH)
    discriminator = load_model(DISCRIMINATOR_MODEL_PATH)
else:
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    train_gan(generator, discriminator, gan, epochs=100000, batch_size=64)

if __name__ == "__main__":
    print("Training completed. Model saved.")
    print("Run the GUI application to generate images.")