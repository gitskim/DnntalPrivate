import subprocess
import numpy as np
import os.path
import pandas as pd
import pickle
import json

# TODO: end it with the slash
ORIGINAL_PATH = '/Users/suhyunkim/git/DnntalPrivate/img'
# TODO: end it with the slash
RETRO_PATH = '/Users/suhyunkim/git/DnntalPrivate/img'


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


def do():
    with open('data.json') as json_file:
        data = json.load(json_file)
        for dictionary in data:
            link = dictionary["Labeled Data"]
            mask = dictionary["Masks"]
            retro = mask["Retro"]

            run_command(f"wget {link} -P {ORIGINAL_PATH}")
            run_command(f"wget {retro} -P {RETRO_PATH}")


do()
