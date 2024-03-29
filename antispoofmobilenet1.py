# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:45:38 2021

@author: Souradeep
"""


# Importing necessary libraries
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook as tqdm

# Importing deep learning libraries
import tensorflow as tf
from keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Flatten, MaxPool2D, Conv2D, Reshape, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.utils import Sequence
from keras.backend import epsilon
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Importing data and defining paths
real = "real_and_fake_face/training_real/"
fake = "real_and_fake_face/training_fake/"
real_path = os.listdir(real)
fake_path = os.listdir(fake)
dataset_path = "real_and_fake_face"

# Setting up data augmentation for training data
data_with_aug = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=1./255,
                                   validation_split=0.2)

# Creating data generators for training and validation
train = data_with_aug.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,
                                          subset="training")
val = data_with_aug.flow_from_directory(dataset_path,
                                        class_mode="binary",
                                        target_size=(96, 96),
                                        batch_size=32,
                                        subset="validation")

# Initializing MobileNetV2 base model
mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96,96,3))

# Clearing previous TensorFlow sessions
tf.keras.backend.clear_session()

# Building the model architecture
model = Sequential([mnet,
                    GlobalAveragePooling2D(),
                    Dense(512, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    Dense(2, activation="softmax")])

# Freezing MobileNetV2 layers
model.layers[0].trainable = False

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

# Displaying model summary
model.summary()

# Defining learning rate scheduler
def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001 
    else:
        return 0.00001

# Setting up learning rate scheduler callback
lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Training the model
hist = model.fit_generator(train,
                           epochs=5,
                           callbacks=[lr_callbacks],
                           validation_data=val)

# Making predictions on validation data
predictions = model.predict_generator(val)

# Defining a function to load and preprocess images
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[...,::-1]

# Displaying sample predictions
val_path = "real_and_fake_face/"
plt.figure(figsize=(15,15))
start_index = 250

for i in range(16):
    plt.subplot(4,4, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
  
    preds = np.argmax(predictions[[start_index+i]])
    
    gt = val.filenames[start_index+i][9:13]

    if gt == "fake":
        gt = 0
    else:
        gt = 1
    
    if preds != gt:
        col = "r"
    else:
        col = "g"

    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, preds, gt), color=col)
    plt.imshow(load_img(val_path + val.filenames[start_index+i]))
    plt.tight_layout()

plt.show()

# Saving the trained model
model.save('antispoofmobilenet2.h5')




