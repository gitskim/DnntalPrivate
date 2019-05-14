from keras.models import load_model
import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import cv2
from keras import backend as K
import json

model_path_classifier = './models/model_classifier.h5'
model = load_model(model_path_classifier)

def predict(img):
    train_data = []
    valid_data = []

    valid_labels = []

    train_data.append((img, 0))

    # Get a pandas dataframe from the data we have in our list
    train_data = pd.DataFrame(train_data, columns=['image', 'label'], index=None)

    # Shuffle the data
    train_data = train_data.sample(frac=1.).reset_index(drop=True)

    # We will convert into a image with 3 channels.
    # We will normalize the pixel values and resizing all the images to 224x224
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)

    # Convert the list into numpy arrays
    data = np.array(valid_data)

    print(data.shape)

    result = model.predict(x=data)
    print(result)
    return result


