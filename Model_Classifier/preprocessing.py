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


#main path

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


#CLAHE filter -- Apply contrast for all images

resolution=2.0

def clahe(path):
  clahe = cv2.createCLAHE(clipLimit=resolution, tileGridSize=(16,16))
  img=cv2.imread(path, 0)
  cv2.imwrite(path, clahe.apply(img))

for path_original in filelist_original:
  clahe(path_original)
for path_mask in filelist_original:
  clahe(path_mask)

#Cropping the images

im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)

#-->Function to crop and save
def crop(image_path, coords, saved_location, isMask=False):
  """
  @param image_path: The path to the image to edit
  @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
  @param saved_location: Path to save the cropped image
  """
  image_obj = Image.open(image_path)
  cropped_image = image_obj.crop(coords)
  if isMask or image_obj.mode == 'I':
    plt.imshow(cropped_image)
    cropped_image = cropped_image.convert("L")
  cropped_image.save(saved_location)

#Function to see is the image should be classify positive or negative
def ispositive(path):
  img=load_img(path, grayscale=True)
  arr=img_to_array(img)
  
  np_array = np.asarray(arr)
  copy = np.copy(np_array)
  # creating a all white array 
  all_black = copy.fill(0)
  
  return np.array_equal(np_array, all_black)

#cropping the original panoramics and the masks, classifying the masks to create positive and negative folders of cropped_panoramic
for i, path_original in enumerate(filelist_original):
  
  path_masks=filelist_masks[i]
  im = Image.open(path_original)
  width, height = im.size
  
  for k in range(50):
    
    
    x=rd.randint(0,width-im_width)
    y=rd.randint(0,height-im_height)
    crop(path_original, (x, y, x+im_width, y+im_height), '/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original/file'+str(i)+'_'+str(k)+'.jpg')
    crop(path_masks, (x, y, x+im_width, y+im_height), '/tmp/dentist_AI/dentist_AI/cropped_train/cropped_masks/file'+str(i)+'_'+str(k)+'.jpg')
    
    if ispositive('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_masks/file'+str(i)+'_'+str(k)+'.jpg'):
      Image.open('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original/file'+str(i)+'_'+str(k)+'.jpg').save('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original_positive/file'+str(i)+'_'+str(k)+'.jpg')
    else:
      Image.open('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original/file'+str(i)+'_'+str(k)+'.jpg').save('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original_negative/file'+str(i)+'_'+str(k)+'.jpg')
    break


  for k in range(50):
    
    
    x=rd.randint(0,width-im_width)
    y=rd.randint(0,height-im_height)
    crop(path_original, (x, y, x+im_width, y+im_height), '/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original/file'+str(i)+'_'+str(k)+'.jpg')
    crop(path_masks, (x, y, x+im_width, y+im_height), '/tmp/dentist_AI/dentist_AI/cropped_train/cropped_masks/file'+str(i)+'_'+str(k)+'.jpg')
    
    if ispositive('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_masks/file'+str(i)+'_'+str(k)+'.jpg'):
      Image.open('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original/file'+str(i)+'_'+str(k)+'.jpg').save('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original_positive/file'+str(i)+'_'+str(k)+'.jpg')
    else:
      Image.open('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original/file'+str(i)+'_'+str(k)+'.jpg').save('/tmp/dentist_AI/dentist_AI/cropped_train/cropped_original_negative/file'+str(i)+'_'+str(k)+'.jpg')
    break