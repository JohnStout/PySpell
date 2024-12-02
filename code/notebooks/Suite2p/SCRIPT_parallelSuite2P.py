## Quick suite2p, run in parallel
# This code has not been tested yet
# If your session counts exceed the number of available cpu cores, this probably wont work
#
# John 9/10/24

import os; import matplotlib.pyplot as plt; import tifffile as tf
path_added = os.path.split(os.path.split(os.getcwd())[0])[0]; os.chdir(path_added); print("Added path:",path_added)
import s2pfuns
from pathlib import Path
import numpy as np
import multiprocessing 

import rootfun as rf # we can import this if our cwd is local
root = rf.dropbox_root(dropbox_folder='timspellman')

# important note
print("Please note that the code provides no feedback when busy")
print("It might seem as though the conversion is complete, when it indeed is not")

# need to sort all of these. 
imgpaths = [r"F:\John\L6 Experiments\recordings_L5CT\L6-05\FOV3\SEDS_day1_FOV3_optoRec\SEDS_day1_FOV3_optoRec_noProbe_img",
r"F:\John\L6 Experiments\recordings_panneuronal\T-30\FOV3\SEDS_day8_noOptoFOV3_probe\SEDS_day8_FOV3_noOpto_probe_img",
r"F:\John\L6 Experiments\recordings_IT\L607-T4\FOV2\SD1_whisker_day6_FOV2_optoRec\SD1_whisker_day6_FOV2_optoRec_img_000"]

# with help from generative AI
def worker(imgpath):
    """thread worker function"""
    print("Running suite2p on",imgpath)
    s2pfuns.fast_suite2p(imgpath=imgpath, savepath='', gcamp='6f')

if __name__ == '__main__':
    jobs = []
    for i in imgpaths:
        i = os.path.join(i,"img.tif") # append the img.tif
        p = multiprocessing.Process(target=worker, args=(i,)) # process
        jobs.append(p)
        p.start()

    # terminate process when complete
    for j in jobs:
        p.join()

    print("Jobs complete")
    #p.terminate()  # Send SIGTERM signal to the process
    #p.join()  # Wait for the process to terminate

