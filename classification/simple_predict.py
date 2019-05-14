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

model = load_model('5_13_23p.h5')

# Define path to the data directory
data_dir = '/home/ek2993/DnntalPrivate/cropped'

# Path to train directory
train_dir = os.path.join(data_dir, 'train')

# Path to validation directory
validation_dir = os.path.join(data_dir, 'val')

# Get the path to the positive and negative sub-directories
train_positive_dir = os.path.join(train_dir, 'cropped_positive_xrays')
train_negative_dir = os.path.join(train_dir, 'cropped_negative_xrays')
validation_positive_dir = os.path.join(validation_dir, 'cropped_positive_xrays')
validation_negative_dir = os.path.join(validation_dir, 'cropped_negative_xrays')

# Get the list of all the images
positives_cases_train = glob.glob(os.path.join(train_positive_dir, '*.jpg'))
negatives_cases_train = glob.glob(os.path.join(train_negative_dir, '*.jpg'))
positives_cases_val = glob.glob(os.path.join(validation_positive_dir, '*.jpg'))
negatives_cases_val = glob.glob(os.path.join(validation_negative_dir, '*.jpg'))

# List that are going to contain validation images data and the corresponding labels
positive_valid_data = []
negative_valid_data = []
positive_valid_labels = []
negative_valid_labels = []


def predict(img):
    train_data = []

    train_data.append((img, 0))

    # Get a pandas dataframe from the data we have in our list
    train_data = pd.DataFrame(train_data, columns=['image', 'label'], index=None)

    # Shuffle the data
    train_data = train_data.sample(frac=1.).reset_index(drop=True)

    # We will convert into a image with 3 channels.
    # We will normalize the pixel values and resizing all the images to 224x224

    # Negatives cases
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=2)
    negative_valid_data.append(img)
    negative_valid_labels.append(label)

    # Convert the list into numpy arrays
    negative_data = np.array(negative_valid_data)

    print(negative_data.shape)

    negative_result = model.predict(x=negative_data)
    print(negative_result)



# Go through all the negatives cases. The label for these cases will be 0
c = 0
for img in negatives_cases_train:
    c += 1
    predict(img)
    if c == 1:
        break

