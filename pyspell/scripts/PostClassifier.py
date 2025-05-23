# Classify/Clean sessions
import os; import matplotlib.pyplot as plt; import tifffile as tf
#path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)

import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import s2pfuns
from pathlib import Path
import numpy as np
import time
import suite2p
import csv
from datetime import datetime

# custom modules
import rootfun as rf # we can import this if our cwd is local
root = rf.dropbox_root(dropbox_folder='timspellman')

# folder to loop over
folder = os.path.join(rf.dropbox_root(),'OtherData','John\EXPERIMENTS\LAYER6\Subjects\Imaging')

# build classifier
obj=s2pfuns.classifyCells(training_sessions_directory=os.path.join(rf.dropbox_root(),'OtherData','ClassifierBuildSuite2p'))

# clean data
sessions = [i for i in rf.list_all_subdirs(folder) if 'suite2p' in i and 'plane0' in i] # get all the directories with suite2p in them
obj.classify(sessions[0])
#[obj.classify(i) for i in sessions]
