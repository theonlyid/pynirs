"""
Python setup script that outputs Matlab.bat for running Matlab with anaconda dependencies.

Must be run with the Anaconda shell prompt. See README Appendix A1 for details.

@author: Ali Zaidi
@version: 0.1
"""

import os
import pytest
from shutil import which

a = which("conda.exe")
if a is None:
    print("Critical Error! Anaconda installation not found! Exiting...")
    exit()
if "Scripts" in a:
    path_val = os.path.split(a)
    conda_activate = os.path.join(path_val[0], "activate.bat")
    if not os.path.isfile(conda_activate):
        print('Critical Error! "activate.bat" not found. exiting...')
        exit()


b = which("matlab.exe")
if "R2024a" in b:
    matpath = b
    if not os.path.isfile(matpath):
        print("Critical Error! Matlab R2024a either not found, or is not the default Matlab. Exiting...")
        exit()

root_dir = os.getcwd()

pydir = which("python.exe")

# Write files to test.txt
with open("matlab.bat", "w+") as f:
    f.write('call "{}"\n'.format(conda_activate))
    f.write('start "" /D "{}" "{}"\n'.format(root_dir, matpath))

print("\nSuccess!\n".upper())
print("Please copy the statements below to the clipboard:\n")
print('pyversion("{}")'.format(pydir))
print('pyenv(ExecutionMode="OutOfProcess")\n')
print("\nRun Matlab by double clicking on 'Matlab.bat', paste the commands copied above and press enter.\n")
