import os
import numpy as np
import pandas as pd
import zipfile
import random as rd
import cv2
import subprocess
#For the preprocessing
import sys
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from skimage.transform import resize
#from PIL import Image

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
zip_ref = zipfile.ZipFile('/home/suhyunkim011/dentist_AI.zip', 'r')

zip_ref.extractall('/home/suhyunkim011/dentist_AI')

zip_ref.close()

train_path = '/home/suhyunkim011/dentist_AI/dentist_AI'

#For the preprocessing
import sys
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from PIL import Image


