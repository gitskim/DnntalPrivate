import subprocess
import numpy as np
import os.path
import json
import zipfile

# TODO: end it with the slash
ORIGINAL_PATH = '/Users/suhyunkim/git/DnntalPrivate/img/mask/'
# TODO: end it with the slash
RETRO_PATH = '/Users/suhyunkim/git/DnntalPrivate/img/radio/'

is_diving = True


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


# move the pictures to a certain directory and create labels
arr_picture = np.array([])
arr_total_score = np.array([])
arr_difficulty_score = np.array([])


def download_zip():
    # run_command("wget https://storage.googleapis.com/dentist_ai/dentist_AI.zip -O ./dnntal.zip")

    #TODO: fill the location of where zip is downloaded
    local_zip = ''
    zip_ref = zipfile.ZipFile(local_zip + '.zip', 'r')
    zip_ref.extractall(local_zip)
    zip_ref.close()

'''
def do():
    counter = 0
    with open('data.json') as json_file:
        data = json.load(json_file)
        for dictionary in data:
            original_link = dictionary["Labeled Data"]

            if "Masks" in dictionary:
                counter += 1
                mask = dictionary["Masks"]
                retro_link = mask["Retro"]
                filename = f"file{counter}.jpg"
                retro_filename = f"{RETRO_PATH}{filename}"
                original_filename = f"{ORIGINAL_PATH}{filename}"
                run_command(f"wget {retro_link} -O {retro_filename}")
                run_command(f"wget {original_link} -O {original_filename}")
'''

download_zip()
