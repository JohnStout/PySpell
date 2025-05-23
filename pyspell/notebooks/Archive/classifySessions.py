#!/usr/bin/env python

import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent folder to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from s2pfuns import classifyCells
from rootfun import dropbox_root, list_all_subdirs

# get relative root path
root_dir = dropbox_root()

# here's our training sessions path
training_sessions_directory = os.path.join(root_dir,'OtherData','ClassifierBuildSuite2p')

# now enter your testing path list, like recurse convert
test_paths = [
    # {'Folder': r"D:\L6 Experiments\L608"},                               'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"D:\L6 Experiments\L612"},                               'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"D:\L6 Experiments\L613"},  
    {'Folder': r"I:\Alex"}, 
]

# now build
obj=classifyCells(training_sessions_directory=training_sessions_directory)

# now test
for mousei in test_paths:

    # get all sessions with suite2p
    sessions = [i for i in list_all_subdirs(mousei['Folder']) if 'suite2p' in i and 'plane' in i]
    for sessi in sessions:

        # classify
        try:
            obj.classify(session_path=sessi)
        except:
            print(f"Failed to classify {sessi}")
            continue
        print(f"Classified {sessi}")



