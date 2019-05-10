#import dependencies
%matplotlib inline
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils import data
import zipfile
import random as rd
import cv2

# --- get data ---

!wget https://storage.googleapis.com/dentist_ai/dentist_AI.zip\
    -O /tmp/dentist_AI.zip

local_zip = '/tmp/dentist_AI.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp/dentist_AI')
zip_ref.close()



train_path = '/tmp/dentist_AI/dentist_AI'

#For the preprocessing
import sys
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from PIL import Image

import glob
filelist_original = glob.glob(os.path.join('/tmp/dentist_AI/dentist_AI/train/original/', '*.jpg'))
filelist_original=sorted(filelist_original)
filelist_masks = glob.glob(os.path.join('/tmp/dentist_AI/dentist_AI/train/masks/', '*.jpg'))
filelist_masks=sorted(filelist_masks)

resolution=2.0

def clahe(path):
  clahe = cv2.createCLAHE(clipLimit=resolution, tileGridSize=(16,16))
  img=cv2.imread(path, 0)
  cv2.imwrite(path, clahe.apply(img))

for path_original in filelist_original:
  clahe(path_original)
for path_mask in filelist_original:
  clahe(path_mask)

train_ids = next(os.walk(train_path+"/train/original"))[2]


# Get train images and masks
X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for i, id_ in enumerate(train_ids), total=len(train_ids):
    path = train_path
    
    # Load X
    img = load_img(path + '/train/original/' + id_, grayscale=True)
    x_img = img_to_array(img)
    
    # --> May not be good for our case, losses information
    #x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True) 
    

    # Load Y
    mask=load_img(path + '/train/masks/' + id_, grayscale=True)
    mask = img_to_array(mask)
    # --> May not be good, same reason
    #mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True) 

    # Save images
    X[i] = x_img / 255
    y[i] = mask / 255

print('Done!')