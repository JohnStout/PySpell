# SCRIPT to cleanup raw traces or run EMD on a few sessions
#
# This is particularly useful in cases where you have a ton of F traces and you 
# just want to clean them without running through recurseConvert, which is a much larger, batch-type script
#
# John Stout on 1/27/2025

# load modules
import os; import matplotlib.pyplot as plt; import tifffile as tf
path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)
import s2pfuns
from pathlib import Path
import numpy as np
import time
import suite2p
import csv
from datetime import datetime

# custom modules
import rootfun as rf # we can import this if our cwd is local
import thorfuns
import sessreg
root = rf.dropbox_root(dropbox_folder='timspellman')

# define datafolder to run over
sessions = [
    r"H:\Layer6\L615\SEDS_day5_FOV1_optoRec_LBC2_p70\SEDS_day5_FOV1_optoRec_LBC2_p70_img",
    r"H:\Layer6\L615\SEDS_day7_FOV1_optoRec_LBC2_p70\SEDS_day7_FOV1_optoRec_LBC2_p70_img_000"
]

# loop over sessions and clean
[s2pfuns.postProcess(s2ppath=sessi).save_modified_f() for sessi in sessions]
