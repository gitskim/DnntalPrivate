import subprocess
import numpy as np
import os.path
import json

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


def do():
    result = run_command("cat /home/kim/git/hw-grading-scripts/hw5/user/test/azurill-p3-test/Makefile | grep TESTS")
    print(result)


do()
