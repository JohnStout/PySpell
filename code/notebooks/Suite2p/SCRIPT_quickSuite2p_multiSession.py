## Quick Suite2p
# Written to allow users to run suite2p in batch
#
# The default method "spellman_ops='n'" uses primarily default suite2p parameters, with 
#  the addition that tau is 0.7
#
# spellman_ops = 'y' uses ops that Tim found useful and applies those to the data. At this point,
#  it is recommended to keep spellman_ops='y'
#
# John Stout
#
# Edits
#  - 10/5/2024: John added spellman_ops method

# load modules
import os; import matplotlib.pyplot as plt; import tifffile as tf
path_added = os.path.split(os.path.split(os.getcwd())[0])[0]; os.chdir(path_added); print("Added path:",path_added)
import s2pfuns
from pathlib import Path
import numpy as np

import rootfun as rf # we can import this if our cwd is local
root = rf.dropbox_root(dropbox_folder='timspellman')

# -- CHANGE ME IF NEEDED -- #
# set to y or n
spellman_ops = 'n'

# code to assign alt_ops as default or not default
if spellman_ops.lower() == 'y':
    print("Using Spellman params")

    # ops path
    ops_path = os.path.join(root,'timspellman','Python','suite2p_ops')

    # load ops data
    alt_ops = np.load(os.path.join(ops_path,'spellman_ops.npy'), allow_pickle=True).item()

else:
    print("Using default s2p params")
    alt_ops = None

# need to sort all of these. Accidentally redid the first one here
imgpaths = [
    r"E:\L6 Experiments\L609-pan\FOV4\SDodor_day8_optoRec_FOV4_LBC0\SDodor_day8_optoRec_FOV4_LBC0_img",
    r"E:\L6 Experiments\L605\FOV4\SEDS_day10_FOV4_LBC0_noOpto\SEDS_day10_FOV4_LBC0_noOpto_img",
    r"E:\L6 Experiments\L608\FOV1\SEDS_day8_FOV1_LBC0_noOpto\SEDS_day8_FOV1_LBC0_noOpto_img",
    r"E:\L6 Experiments\L608\FOV1\SD1_whisker_day5_FOV1_noOpto_LBC0\SD1_whisker_day5_FOV1_noOpto_LBC0_img",
    r"E:\L6 Experiments\L607T4\FOV3\SD2_odor_day7_FOV3_LBC0_noOpto\SD2_odor_day7_FOV3_LBC0_noOpto_img"
]

# loop over folders and run suite2p
for i in imgpaths:

    # Don't change
    print("Running suite2p and saving to:",i)

    # change i
    i = os.path.join(i,"img.tif")

    # zoom factor
    #if "L6-05" in i:
    #    zoom_factor = 1.0
    #else:
    #    zoom_factor = 2.0

    # run s2p
    s2pfuns.fast_suite2p(imgpath=i, savepath='', gcamp='6f', alt_ops=alt_ops)

