## Quick suite2p
import os; import matplotlib.pyplot as plt; import tifffile as tf
path_added = os.path.split(os.path.split(os.getcwd())[0])[0]; os.chdir(path_added); print("Added path:",path_added)
import s2pfuns
from pathlib import Path
import numpy as np

import rootfun as rf # we can import this if our cwd is local
root = rf.dropbox_root(dropbox_folder='timspellman')

# interact with user for input
imgpath  = os.path.normpath(input("Enter path of your img.tif file without " " marks"))
savepath = os.path.split(imgpath)[0]
gcamp    = '6f'

# Don't change
print("Running suite2p and saving to:",savepath)

# run s2p
s2pfuns.fast_suite2p(imgpath=imgpath, savepath='', gcamp=gcamp)

