# Classify/Clean sessions
import os; import matplotlib.pyplot as plt; import tifffile as tf
#path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)

import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from s2pfuns import cellClassifier
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
build_classifier = True
if build_classifier == True:
    print('Building classifier...')
    obj = cellClassifier(training_sessions_directory=os.path.join(rf.dropbox_root(),'OtherData','ClassifierBuildSuite2p'))
    obj.check_classifier_loso(auto_feature_select=False, preset_features=False, feature_list=None)
    obj.build_classifier(preset_features=True, grid_search=True, drop_nan=True, skew_classifier=False)

    # get current datetime
    current_date = datetime.now().strftime("%m%d%Y")
    obj.save_model(filepath=os.path.join(rf.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier_{}.pkl'.format(current_date)))
    
else:
    print("Loading classifier from file:", os.path.join(rf.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier_06112025.pkl'))
    obj = cellClassifier(load_classifier=True, model_path=os.path.join(rf.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier_06112025.pkl'))
    
# to just process a single session
#obj.classify(sessions[0])
#[obj.classify(i) for i in sessions]
