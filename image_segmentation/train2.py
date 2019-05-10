import os
import numpy as np
import zipfile
import subprocess
import glob
import cv2
from tqdm import tqdm
import sys
# For building the model_2
import tensorflow as tf
import keras as keras
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.layers import Activation, add, multiply, Lambda
from tensorflow.python.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.initializers import glorot_normal, random_normal, random_uniform
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # roc curve tools
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize

import loss
import model
import preprocessing as prep

# PATH_TRAIN = '/home/ek2993/DnntalPrivate/dentist_AI'
# Preprocessing
im_width = 128
im_height = 128
border = 5
im_chan = 1  # Number of channels: first is original and second cumsum(axis=0)

PATH_TRAIN = 'train/original/*.jpg'
filelist_original = glob.glob(PATH_TRAIN)
PATH_TRAIN2 = 'train/masks/*.jpg'
filelist_masks = glob.glob(PATH_TRAIN2)
filelist_masks = sorted(filelist_masks)

# filelist_original = glob.glob(
#     os.path.join(PATH_TRAIN + 'train/original', '*.jpg'))
# filelist_original = sorted(filelist_original)
# filelist_masks = glob.glob(
#     os.path.join(PATH_TRAIN + 'train/masks', '*.jpg'))


print("... starint clahe ...")
for path_original in filelist_original:
    print(path_original)

    prep.clahe(path_original)
    print(path_original)
    prep.center_crop(path_original)
for path_mask in filelist_original:
    # question: why are you clahe'ing path_mask
    prep.clahe(path_mask)
    prep.center_crop(path_original)

train_ids = filelist_original[2]



# Get and resize train images and masks
X = np.zeros((len(filelist_original), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(filelist_original), im_height, im_width, 1), dtype=np.float32)
sys.stdout.flush()
for i, filelist in enumerate(filelist_original):
    sys.stdout.flush()
    # Load X
    img = load_img(filelist_original[i], grayscale=True)
    x_img = img_to_array(img)

    # --> May not be good for our case, losses information
    x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

    # Load Y
    mask = img_to_array(load_img(filelist_masks[i], grayscale=True))
    # --> May not be good, same reason
    mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

    # Save images
    X[i] = x_img / 255
    y[i] = mask/255

# Build U-Net model
TARGET_SHAPE = 128
im_width = 128
im_height = 128
border = 5
BATCH_SIZE = 16


img_row = 128
img_col = 128
img_size = 128
img_chan = 1
epochnum = 100
batchnum = 16
input_size = (img_row, img_col, img_chan)
sgd = SGD(lr=0.01, momentum=0.9)
model = model.attn_unet(sgd, input_size, loss.tversky_loss)
hist = model.fit(X, y, validation_split=0.15,
                 shuffle=True, epochs=epochnum, batch_size=batchnum,
                 verbose=True)
model.save_weights('5-10-9p-attn_unet-tverskyloss.h5')


