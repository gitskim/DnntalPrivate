import matplotlib.pyplot as plt
import numpy as np
import os

# The code in this notebook should work identically in TF v1 and v2
import tensorflow as tf
import zipfile

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

train_positive_dir = os.path.join(train_dir, 'cropped_positive_xrays')
train_negative_dir = os.path.join(train_dir, 'cropped_negative_xrays')
validation_positive_dir = os.path.join(validation_dir, 'cropped_positive_xrays')
validation_negative_dir = os.path.join(validation_dir, 'cropped_negative_xrays')

num_positive_tr = len(os.listdir(train_positive_dir))
num_negative_tr = len(os.listdir(train_negative_dir))

num_positive_val = len(os.listdir(validation_positive_dir))
num_negative_val = len(os.listdir(validation_negative_dir))

total_train = num_positive_tr + num_negative_tr
total_val = num_positive_val + num_negative_val

print('Training positive images:', num_positive_tr)
print('Training negative images:', num_negative_tr)

print('Validation positive images:', num_positive_val)
print('Validation negative images:', num_negative_val)

print("--")

print("Total training images:", total_train)
print("Total validation images:", total_val)

# Notice we do not include the `top`, or the Dense layers used to classify the 1,000 classes from ImageNet.
conv_base = VGG16(weights='imagenet',include_top=False, input_shape=(150, 150, 3))

TARGET_SHAPE = 150 
BATCH_SIZE = 16

# Cache activations for our training and validation data
datagen = ImageDataGenerator(rescale=1./255)

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(TARGET_SHAPE, TARGET_SHAPE),
        batch_size=BATCH_SIZE,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
      
        features_batch = conv_base.predict(inputs_batch)
        # print(features_batch.shape)
        # (32, 4, 4, 512)
        # Think: batch_size, rows, cols, channels
        
        features[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)

FLATTENED_SHAPE = 4 * 4 * 512

train_features = np.reshape(train_features, (total_train, FLATTENED_SHAPE))
validation_features = np.reshape(validation_features, (total_val, FLATTENED_SHAPE))

EPOCHS = 50

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=FLATTENED_SHAPE))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(validation_features, validation_labels))