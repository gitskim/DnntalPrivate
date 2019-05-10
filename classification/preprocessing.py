#import dependencies
#%matplotlib inline
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

# !wget https://storage.googleapis.com/dentist_ai/dentist_AI.zip\
#     -O /tmp/dentist_AI.zip

# local_zip = '/tmp/dentist_AI.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')

# zip_ref.extractall('/tmp/dentist_AI')

# zip_ref.close()


#main path

#train_path = '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/original/'


#For the preprocessing
import sys
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from PIL import Image

import glob
filelist_xrays = glob.glob(os.path.join('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/original/xrays/', '*.jpg'))
filelist_xrays=sorted(filelist_xrays)
filelist_masks = glob.glob(os.path.join('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/original/masks/', '*.jpg'))
filelist_masks=sorted(filelist_masks)


#CLAHE filter -- Apply contrast for all images

resolution=2.0

def clahe(path):
    clahe = cv2.createCLAHE(clipLimit=resolution, tileGridSize=(16,16))
    img=cv2.imread(path, 0)
    cv2.imwrite(path, clahe.apply(img))

for path_xrays in filelist_xrays:
    clahe(path_xrays)
# for path_mask in filelist_masks:
#     clahe(path_mask)

#Cropping the images

im_width = 256
im_height = 256

#-->Function to crop and save
def crop(image_path, coords, saved_location, isMask=False):
    '''
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    '''
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    if isMask or image_obj.mode != "L":
        cropped_image = cropped_image.convert("L")
    cropped_image.save(saved_location)

#Function to see is the image should be classify positive or negative
def ispositive(path):
    img=load_img(path, grayscale=True)
    return img.getbbox()!=None
    # arr=img_to_array(img)
  
    # np_array = np.asarray(arr)
    # copy = np.copy(np_array)
    # # creating a all white array 
    # all_black = copy.fill(0)
    # print(np.array_equal(np_array, all_black))
    # return np.array_equal(np_array, all_black)

#cropping the original panoramics and the masks, classifying the masks to create positive and negative folders of cropped_panoramic
for i, path_xray in enumerate(filelist_xrays):
    
    im = Image.open(path_xray)
    width, height = im.size
    
    #Center and finding a white pixel
    path_mask=filelist_masks[i]
    mask_array=np.array(Image.open(path_mask))/255
    x0=np.where(mask_array!=0)[1][0]
    y0=np.where(mask_array!=0)[0][0]

    if i<50:
        for k in range(50):
            x=x0+rd.randint(-256,0)#Center in a white pixel
            y=y0+rd.randint(-256,0)#Center in a white pixel
            crop(path_xray, (x, y, x+im_width, y+im_height), '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg')
            crop(path_mask, (x, y, x+im_width, y+im_height), '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/masks/file'+str(i)+'_'+str(k)+'.jpg')
            if ispositive('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/masks/file'+str(i)+'_'+str(k)+'.jpg'):
                Image.open('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg').save('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped/train/cropped_positive_xrays/file'+str(i)+'_'+str(k)+'.jpg')
            else:
                Image.open('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg').save('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped/train/cropped_negative_xrays/file'+str(i)+'_'+str(k)+'.jpg')
        for k in range(50):
            x=rd.randint(0,width-im_width)
            y=rd.randint(0,height-im_height)
            crop(path_xray, (x, y, x+im_width, y+im_height), '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg')
            crop(path_mask, (x, y, x+im_width, y+im_height), '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/masks/file'+str(i)+'_'+str(k)+'.jpg')
            if ispositive('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/masks/file'+str(i)+'_'+str(k)+'.jpg'):
                Image.open('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg').save('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped/train/cropped_positive_xrays/file'+str(i)+'_'+str(k)+'.jpg')
            else:
                Image.open('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg').save('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped/train/cropped_negative_xrays/file'+str(i)+'_'+str(k)+'.jpg')
    if i<50:
        for k in range(50):
            x=x0+rd.randint(-256,0)#Center in a white pixel
            y=y0+rd.randint(-256,0)#Center in a white pixel
            crop(path_xray, (x, y, x+im_width, y+im_height), '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg')
            crop(path_mask, (x, y, x+im_width, y+im_height), '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/masks/file'+str(i)+'_'+str(k)+'.jpg')
            if ispositive('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/masks/file'+str(i)+'_'+str(k)+'.jpg'):
                Image.open('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg').save('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped/train/cropped_positive_xrays/file'+str(i)+'_'+str(k)+'.jpg')
            else:
                Image.open('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg').save('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped/train/cropped_negative_xrays/file'+str(i)+'_'+str(k)+'.jpg')
        for k in range(50):
            x=rd.randint(0,width-im_width)
            y=rd.randint(0,height-im_height)
            crop(path_xray, (x, y, x+im_width, y+im_height), '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg')
            crop(path_mask, (x, y, x+im_width, y+im_height), '/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/masks/file'+str(i)+'_'+str(k)+'.jpg')
            if ispositive('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/masks/file'+str(i)+'_'+str(k)+'.jpg'):
                Image.open('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg').save('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped/val/cropped_positive_xrays/file'+str(i)+'_'+str(k)+'.jpg')
            else:
                Image.open('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped0/xrays/file'+str(i)+'_'+str(k)+'.jpg').save('/Users/arielcohencodar/Desktop/These_Phoebe/src/Dataset/dentist_AI/cropped/val/cropped_negative_xrays/file'+str(i)+'_'+str(k)+'.jpg')