# py

# load modules
import os; import matplotlib.pyplot as plt; import tifffile as tf

# TODO: This is not a safe fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import thorfuns

# enter paths with videos to convert
filepaths = [
    {'Folder':  r"H:\Layer6\E04_M_CC_FLEX-GCAMP_L6CTChrimson"},
]

# convert each video
for filepath in filepaths:

    # search for folders with _cam
    

    # convert to avi
    thorfuns.stitch_cam_to_avi(filepath['Folder'], fps=16.5, delete_tifs=True)