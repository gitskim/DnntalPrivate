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

PATH_TRAIN = '/home/ek2993/DnntalPrivate/dentist_AI'
# Preprocessing
im_width = 128
im_height = 128
border = 5
im_chan = 1  # Number of channels: first is original and second cumsum(axis=0)

filelist_original = glob.glob(
    os.path.join(PATH_TRAIN + '/train/original', '*.jpg'))
filelist_original = sorted(filelist_original)
filelist_masks = glob.glob(
    os.path.join(PATH_TRAIN + '/train/masks', '*.jpg'))
filelist_masks = sorted(filelist_masks)

print("... starint clahe ...")
for i, path_original in enumerate(filelist_original):
    prep.clahe(path_original)
    print(path_original)
    #filelist_original[i] = prep.center_crop(path_original)
#for path_mask in enumerate(filelist_masks):
    #filelist_masks[i] = prep.center_crop(path_original)


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


'''
# explanation: https://software.intel.com/en-us/articles/hands-on-ai-part-14-image-data-preprocessing-and-augmentation
datagen = ImageDataGenerator(
    # to normalize the dataset such that the mean value of each data sample would be equal to 0
    featurewise_center=True,
    # sets the standard deviation value to 1.
    featurewise_std_normalization=True,
    shear_range=0.75,
    zoom_range=0.5,
    horizontal_flip=Ture,
    fill_mode="nearest"
)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X, y, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
'''

img_row = 128
img_col = 128
img_size = 128
img_chan = 1
epochnum = 100
batchnum = 16
input_size = (img_row, img_col, img_chan)
sgd = SGD(lr=0.01, momentum=0.9)
model = model.unet(sgd, input_size, loss.tversky_loss)
hist = model.fit(X, y, validation_split=0.15,
                 shuffle=True, epochs=epochnum, batch_size=batchnum,
                 verbose=True)

model.save_weights('5-9-9p-unet-tverskyloss.h5')

# Function to distort image
# https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
'''
# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]
'''
