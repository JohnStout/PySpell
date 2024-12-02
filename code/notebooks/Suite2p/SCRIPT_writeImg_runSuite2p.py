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

# custom modules
import rootfun as rf # we can import this if our cwd is local
import thorfuns
root = rf.dropbox_root(dropbox_folder='timspellman')

# -- ENTER YOUR IMG PATHS HERE -- #
# imgpaths = [r"/path/to/your/folder", r"/path/to/your/next/folder"]
imgpaths = [
    r"E:\L6 Experiments\L607T4\FOV3\SEDS_day10_FOV3_optoRec_LBC0\SEDS_day10_FOV3_optoRec_LBC0_img",
    r"E:\L6 Experiments\L607T4\FOV3\SEDS_day11_FOV3_optoRec_noProbe_LBC0\SEDS_day11_FOV3_optoRec_noProbe_LBC0_img",

    r"E:\L6 Experiments\T30\FOV5\SEDS_day5_FOV5_optoRec\SEDS_day5_FOV5_optoRec_LBC0_img",
    
    r"E:\L6 Experiments\L6R11\SD1_whisker_day6_FOV2_optoRec_LBC0\SD1_whisker_day6_FOV2_optoRec_LBC0_img",
    r"E:\L6 Experiments\L608\FOV1\SEDS_day8_FOV1_LBC0_noOpto\SEDS_day8_FOV1_LBC0_noOpto_img",
    r"E:\L6 Experiments\L607T4\FOV3\SEDS_day8_FOV3_noProbe_noOpto_LBC0\SEDS_day8_FOV3_noProbe_noOpto_LBC0_img_001",
    r"E:\L6 Experiments\L607T4\FOV3\SDswitch_day6_FOV3_noOpto_LBC0\SDswitch_day6_FOV3_noOpto_LBC0_img",

    ]

# -- ENTER WHETHER TO USE DEFAULT SPELLMAN OPS OR SUITE2P OPS -- #
# when set to 'n', you use a modified version of the suite2p ops where tau = 0.7 (Carsen Stringer mentioned this in a video)
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

# loop over folders and run suite2p
for i in imgpaths:

    # Don't change
    print("Searching for img.tif and suite2p in:",i)

    # search for img.tif
    dir_contents = os.listdir(i)

    # search for key files
    imgFound = len([i for i in dir_contents if 'img.tif' in i])
    s2pFound = len([i for i in dir_contents if 'suite2p' in i])

    # check your session title for 'opto'
    if 'opto' in os.path.split(i)[-1].lower():
        led_artifact='y'
    else:
        led_artifact='n'

    # if the img.tif file is not found, run conversion
    if imgFound == 0:
        print("No img.tif file discovered. Writing file now...")
        thorfuns.RawToTif(filepath=i).convert(method='max_proj', 
                                              chunker=1000, 
                                              led_artifacts=led_artifact)

    # check if suite2p is found in the directory provided
    if s2pFound == 0:
        print("No suite2p folder discovered. Running suite2p now...")
        s2pfuns.fast_suite2p(imgpath=os.path.join(i,'img.tif'), 
                             savepath='', 
                             gcamp='6f', 
                             alt_ops=alt_ops)

