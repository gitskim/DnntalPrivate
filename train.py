import os
import numpy as np
import zipfile
import subprocess

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

'''
for path_original in filelist_original:
  clahe(path_original)
for path_mask in filelist_original:
  clahe(path_mask)


im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
 
filelist_original = glob.glob(os.path.join('/home/ek2993/dnntal/train/original/', '*.jpg'))
filelist_original=sorted(filelist_original)
filelist_masks = glob.glob(os.path.join('/home/ek2993/dnntal/train/masks/', '*.jpg'))
filelist_masks=sorted(filelist_masks)
'''

resolution=2.0


train_path = '/home/suhyunkim011/dentist_AI/dentist_AI'


