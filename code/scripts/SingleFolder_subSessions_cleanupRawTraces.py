# SCRIPT to cleanup raw traces in a single folder with a bunch of subfolders
#
# This is particularly useful in cases where you have a ton of F traces and you 
# just want to clean them without running through recurseConvert, which is a much larger, batch-type script
#
# John Stout on 1/22/2025

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
Datafolder = r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p"

# get all subdirs
subdirs = rf.list_all_subdirs(phile_name = Datafolder)
sessions_to_clean = [i for i in subdirs if 'plane0' in i] # filter out for suite2p

# loop over sessions and clean
[s2pfuns.postProcess(s2ppath=sessi).cleanup_raw_traces() for sessi in sessions_to_clean]