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

#For the preprocessing
import sys
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from PIL import Image

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

run_command("wget https://storage.googleapis.com/dentist_ai/dentist_AI.zip\
    -O /home/suhyunkim011/dentist_AI.zip")

local_zip = '/home/suhyunkim011/dentist_AI.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall(local_zip)

zip_ref.close()

import glob
filelist_original = glob.glob(os.path.join('/home/suhyunkim011/dentist_AI/train/original/', '*.jpg'))
filelist_original=sorted(filelist_original)
filelist_masks = glob.glob(os.path.join('/home/suhyunkim011/dentist_AI/train/masks/', '*.jpg'))
filelist_masks=sorted(filelist_masks)

# This is not clahe, it's just showing an image
img = cv2.imread(filelist_original[0], 0)
# plt.imshow(img)
plt.imshow(img,cmap='gray', vmin=0, vmax=255)

resolution=2.0

def clahe(path):
  clahe = cv2.createCLAHE(clipLimit=resolution, tileGridSize=(16,16))
  img=cv2.imread(path, 0)
  cv2.imwrite(path, clahe.apply(img))


for path_original in filelist_original:
  clahe(path_original)
for path_mask in filelist_original:
  clahe(path_mask)


im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)

#Function to crop and save
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


# Function to see is the image should be classify positive or negative
def ispositive(path):
    img = load_img(path, grayscale=True)
    arr = img_to_array(img)

    np_array = np.asarray(arr)
    copy = np.copy(np_array)
    # creating a all white array
    all_black = copy.fill(0)

    return np.array_equal(np_array, all_black)


