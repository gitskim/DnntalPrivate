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

PATH_HOME = '/home/ek2993/dnntal/DnntalPrivate/dnntal/dentist_AI'


# --- get data ---
def run_command(command, logfile=None, print_output=True, return_output=True):
    # if logfile != None:
    #     command += ' |& tee ' + logfile
    output = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        executable='/bin/bash'
    ).stdout.read()
    if print_output:
        print(output)
    if return_output:
        return str(output)


# Preprocessing
im_width = 128
im_height = 128
border = 5
im_chan = 1  # Number of channels: first is original and second cumsum(axis=0)

filelist_original = glob.glob(
    os.path.join('/home/ek2993/dnntal/DnntalPrivate/dnntal/dentist_AI/train/original', '*.jpg'))
filelist_original = sorted(filelist_original)
filelist_masks = glob.glob(
    os.path.join('/home/ek2993/dnntal/DnntalPrivate/dnntal/dentist_AI/train/masks', '*.jpg'))
filelist_masks = sorted(filelist_masks)

resolution = 2.0


def clahe(path):
    clahe = cv2.createCLAHE(clipLimit=resolution, tileGridSize=(16, 16))
    img = cv2.imread(path, 0)
    cv2.imwrite(path, clahe.apply(img))


print("... starint clahe ...")
for path_original in filelist_original:
    clahe(path_original)
for path_mask in filelist_original:
    # question: why are you clahe'ing path_mask
    clahe(path_mask)

train_ids = filelist_original[2]

train_path = '/home/ek2993/dnntal/DnntalPrivate/dnntal/dentist_AI'
# Get and resize train images and masks
X = np.zeros((len(filelist_original), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(filelist_original), im_height, im_width, 1), dtype=np.float32)
print('suhyun before for loop')
sys.stdout.flush()
for i, filelist in enumerate(filelist_original):
    path = train_path
    print('suhyun id_')
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

# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


# Build U-Net model
TARGET_SHAPE = 128
im_width = 128
im_height = 128
border = 5
BATCH_SIZE = 16


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
    return tp


def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    return tn


def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

kinit = 'glorot_normal'
epsilon = 1e-5
smooth = 1

def UnetConv2D(input, outdim, is_batchnorm, name):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name + '_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name + '_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x


def unet(opt, input_size, lossfxn):
    inputs = Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 256, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv5)

    up6 = concatenate(
        [Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv7)

    up8 = concatenate(
        [Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up8)

    up9 = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=opt, loss=lossfxn, metrics=[dsc, tp, tn])
    return model

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

img_row = 128
img_col = 128
img_size = 128
img_chan = 1
epochnum = 100
batchnum = 16
input_size = (img_row, img_col, img_chan)
sgd = SGD(lr=0.01, momentum=0.9)
model = unet(sgd, input_size, tversky_loss)
hist = model.fit(X, y, validation_split=0.15,
                 shuffle=True, epochs=epochnum, batch_size=batchnum,
                 verbose=True)

# results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks)

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

# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]