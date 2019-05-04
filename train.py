import os
import numpy as np
import zipfile
import subprocess
import glob
import cv2

# For building the model_2
import tensorflow as tf
import keras as keras
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

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
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
 
filelist_original = glob.glob(os.path.join('/home/ek2993/dnntal/DnntalPrivate/dnntal/dentist_AI/train/original/', '*.jpg'))
filelist_original=sorted(filelist_original)
filelist_masks = glob.glob(os.path.join('/home/ek2993/dnntal/DnntalPrivate/dnntal/dentist_AI/train/masks/', '*.jpg'))
filelist_masks=sorted(filelist_masks)

resolution=2.0

def clahe(path):
  clahe = cv2.createCLAHE(clipLimit=resolution, tileGridSize=(16,16))
  img=cv2.imread(path, 0)
  cv2.imwrite(path, clahe.apply(img))

train_path = '/home/suhyunkim011/dentist_AI/dentist_AI'


for path_original in filelist_original:
  clahe(path_original)
for path_mask in filelist_original:
  clahe(path_mask)
