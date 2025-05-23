# read AVI

# load modules
import os; import matplotlib.pyplot as plt; import tifffile as tf
#path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)

import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import thorfuns

file_path = r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\videos and gifs\SDswitch_day10_optoRec_LBC2_cam_C2_F01_Z0000_2_230260_trim.avi"



