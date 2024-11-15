# LiverCompactNet-
LiverCompactNet, to accurately classify liver cancer stages from medical images. Utilizing a ResNet-inspired CNN architecture, advanced data augmentation, and model optimization techniques, LiverCompactNet aims to assist in early and precise liver cancer diagnosis. 
# Advanced Project Structure
LiverCancerDetection
├── README.md
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── liver_compact_net.py
│   ├── lwcnn.py
│   └── squeezenet.py
├── main.py
├── train.py
├── evaluate.py
├── utils.py
├── config.py
└── requirements.txt
# Requirements (requirements.txt)
tensorflow==2.13.0
numpy
pandas
scikit-learn
matplotlib
opencv-python
albumentations
# Configuration File (config.py)# config.py
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "models/saved_model.h5"
CHECKPOINT_PATH = "models/checkpoint.ckpt"
NUM_CLASSES = 4
LABEL_DICT = {"Non-Demented": 0, "Very Mild Demented": 1, "Mild Demented": 2, "Moderate Demented": 3}
#  Data Preprocessing and Augmentation (utils.py)
# utils.py
import os
import numpy as np
import cv2
import albumentations as A
from tensorflow.keras.utils import to_categorical
from config import IMG_SIZE, LABEL_DICT

# Define an augmentation pipeline
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Blur(p=0.3),
        A.GaussNoise(p=0.3)
    ])

def load_data(data_dir, augment=False):
    images, labels = [], []
    augment_pipeline = get_augmentation_pipeline() if augment else None

    for label in LABEL_DICT.keys():
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)
            if augment_pipeline:
                img = augment_pipeline(image=img)["image"]
            images.append(img)
            labels.append(LABEL_DICT[label])

    images = np.array(images) / 255.0
    labels = to_categorical(np.array(labels), num_classes=len(LABEL_DICT))
    return images, labels
# Model Architecture (models/liver_compact_net.py)
# models/liver_compact_net.py
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Add, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def resnet_block(inputs, filters, kernel_size=3, strides=1, activation='relu'):
    x = Conv2D(filters, kernel_size, strides=strides, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding="same")(inputs)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def LiverCompactNet(input_shape=(224, 224, 3), num_classes=4):
    inputs = Input(shape=input_shape)

    # Initial Conv layer
    x = Conv2D(64, (7, 7), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # ResNet blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, strides=2)
    x = resnet_block(x, 256, strides=2)
    x = resnet_block(x, 512, strides=2)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model
 # Training Script (train.py)
 # train.py
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from utils import load_data
from config import *
from models.liver_compact_net import LiverCompactNet

# Load data
train_dir = "data/train"
X, y = load_data(train_dir, augment=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LiverCompactNet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Set up callbacks
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    callbacks=[checkpoint, early_stopping, lr_scheduler])
model.save(MODEL_SAVE_PATH)
# Evaluation Script (evaluate.py)
# evaluate.py
import tensorflow as tf
from sklearn.metrics import classification_report
from utils import load_data
from config import IMG_SIZE, MODEL_SAVE_PATH

# Load test data
test_dir = "data/test"
X_test, y_test = load_data(test_dir)

# Load the model
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)
y_test_classes = tf.argmax(y_test, axis=1)
print(classification_report(y_test_classes, y_pred_classes, target_names=["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]))
# Visualization of Training Results (main.py continued)
# main.py (continued)
import matplotlib.pyplot as plt

def plot_training(history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Accuracy Over Epochs")
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Over Epochs")
    
    plt.show()
 # README.md
 # Liver Cancer Detection using CNN - LiverCompactNet

This project implements a ResNet-based CNN architecture, LiverCompactNet, to classify liver cancer stages from medical images.

## Project Structure

- `data/`: Contains training and testing images, organized by labels.
- `models/`: Contains model definitions.
- `train.py`: Training script with checkpointing and early stopping.
- `evaluate.py`: Evaluation script to test the model.
- `utils.py`: Utility functions for data loading and augmentation.
- `config.py`: Configuration file for hyperparameters and file paths.
- `requirements.txt`: List of dependencies.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LiverCancerDetection.git
    cd LiverCancerDetection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. To train the model:
    ```bash
    python train.py
    ```

2. To evaluate the model:
    ```bash
    python evaluate.py
    ```

3. To visualize training results:
    ```bash
    python main.py
    ```

