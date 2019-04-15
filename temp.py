import subprocess
import os.path
import json

#TODO: don't end it with the slash
test_loc = "/home/kim/git/hw-grading-scripts/hw5/user/test"
test_loc2 = "/home/kim/git/hw-grading-scripts/hw5/user/test/azurill-p"
team_name = 'azurill'

def run_command(command, logfile=None, print_output=True, return_output=True):
    # if logfile != None:
    #     command += ' |& tee ' + logfile
    output = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        executable='/bin/bash'
    ).stdout.read().decode('utf-8').strip()
    if print_output:
        print(output)
    if return_output:
        return output


# move the pictures to a certain directory and create labels


def do():
    leaks = []
    for i in range(1, 5):
        run_command("git checkout -f hw5p%dhandin" %i)
        run_command("cd user/module/fridge")
        run_command("make")
        run_command("sudo /usr/local/bin/kedr start fridge.ko")
        run_command("sudo insmod fridge.ko")
        run_command("cd %s/%s-p%d-test" %(test_loc, team_name, i))
        run_command("make")


        result = run_command("cat %s%d-test/Makefile | grep \"TESTS =\"" %(test_loc2, i))
        result = result.split()
        result = result[2:]
        print(result)
        for test in result:
            run_command('./%s' %test)

        run_command("sudo rmmod fridge")
        run_command("make clean")
        run_command("sudo /usr/local/bin/kedr stop")
        result1 = run_command("sudo dmesg | grep \"Possible leaks\"")
        leaks.append(result1)

    print(leaks)

do()