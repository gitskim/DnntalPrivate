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
from PIL import Image

model_path_classifier = './models/model_classifier.h5'
model = load_model(model_path_classifier)

def predict(img):

    # We will convert into a image with 3 channels.
    # We will normalize the pixel values and resizing all the images to 224x224
    img = cv2.imread(str(img))
    return result_predict(img)

def result_predict(img):

    valid_data = []

    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    valid_data.append(img)

    # Convert the list into numpy arrays
    data = np.array(valid_data)

    result = model.predict(x=data)
    return result

# def post_processing(path_img):
#print(predict('../../Dataset/dentist_AI/cropped/val/cropped_positive_xrays/file2_20.jpg'))

def post_processing(img_path):

    img=cv2.imread(img_path)
    array2 = img.copy()
    length,wide, z = img.shape
    size = 224 
    length = int(length/size)*size
    wide = int(wide/size)*size

    for x in range(0,wide,size):
        for y in range(0,length,size):
            
            crop = array2[y:y+size,x:x+size]
            # Send to predcit 
            number = result_predict(crop)[0][1]
            if number > 0.5:
                array2[y:y+size,x:x+size]=0
    resultat=Image.fromarray(array2, 'RGB')
    return resultat