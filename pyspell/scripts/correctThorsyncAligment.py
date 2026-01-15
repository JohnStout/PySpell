# correctThorsyncAligment
"""
This script is used to correct the alignment of Thorsync data in a given folder structure.

After entering your folder into imgpaths, run the script and interact with the GUI to visually identify misalignments.

John Stout

MIGHT NOT BE NEEDED BC thorfuns.importThorsync actually flags misalignments already

"""

# load modules
import os; import matplotlib.pyplot as plt; import tifffile as tf

# TODO: This is not a safe fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pathlib import Path
import numpy as np
import time
import csv
from datetime import datetime

# custom modules
import rootfun as rf # we can import this if our cwd is local
root = rf.dropbox_root(dropbox_folder='timspellman')
import h5py

# recursive method
imgpaths = dict()
imgpaths = [

    # {'Folder': r"H:\Layer6\L618_M_CC-ConFoffGCaMP_L6CTChrimson"},
    # {'Folder': r"H:\Layer6\TA05-ConFoffGCaMP-Chrimson"},
    # {'Folder': r"H:\Layer6\L622_F_ConFoffGCaMP_L6CTChrimson"},
     #{'Folder': r"H:\Layer6\L623_M_ConFoffGCaMP_L6CTChrimson_L5CTrec"},
    # {'Folder': r"H:\Layer6\E04_M_CC_FLEX-GCAMP_L6CTChrimson"},
    
    #{'Folder': r"H:\Layer6\E04_M_CC_FLEX-GCAMP_L6CTChrimson"},
    {'Folder': r"H:\Layer6\TA05-ConFoffGCaMP-Chrimson"},
    # # # # Peyton folders
    #  {'Folder': r"Z:\Peyton\L602",     'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #  {'Folder': r"Z:\Peyton\L607",     'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #  {'Folder': r"Z:\Peyton\1",        'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #  {'Folder': r"Z:\Peyton\B02",      'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #  {'Folder': r"Z:\Peyton\T27",      'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #  {'Folder': r"Z:\Peyton\B03",      'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #  {'Folder': r"Z:\Peyton\48",       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #  {'Folder': r"Z:\Peyton\GFAP-hsyn",'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},

    # # John folders
    # {'Folder': r"E:\L6 Experiments\L608",                               'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # {'Folder': r"E:\L6 Experiments\L612",                               'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # {'Folder': r"E:\L6 Experiments\L613",                               'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # {'Folder': r"E:\L6 Experiments\L614",                               'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # {'Folder': r"E:\L6 Experiments\L616",                               'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # {'Folder': r"E:\L6 Experiments\L607T4",                             'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # {'Folder': r"E:\L6 Experiments\T30",                                'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # Finished: {'Folder': r"H:\Layer6\L615",                                       'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # # {'Folder': r"H:\Layer6\L609-pan",                                   'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},    
    # # # {'Folder': r"H:\Layer6\L1",                                         'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # # # {'Folder': r"H:\Layer6\L6R11",                                      'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
   # {'Folder': r"H:\Layer6\L605",                                       'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    # # # {'Folder': r"H:\Layer6\L645",                                       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    #{'Folder': r"H:\Layer6\L615",                                       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False,'rerunClassifier': True},    

    #{'Folder': r"F:\John\L6 Experiments\recordings_panneuronal\T-30",   'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    #{'Folder': r"F:\John\L6 Experiments\recordings_L5CT\L6-05",         'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    #{'Folder': r"H:\Layer6\L605",                                       'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    #{'Folder': r"F:\John\L6 Experiments\recordings_IT\L607-T4",         'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},
    
    # MDT folderP80
    #{'Folder': r"E:\ThalamicRec\MDT1", 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'rerunClassifier': True},      

    #Alex Folders
    #{'Folder': r"G:\2PData4",       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #{'Folder': r"Z:\Alex",       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},
    #{'Folder': r"Z:\Alex 2AB RIP",       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False, 'rerunOASIS': False, 'rerunClassifier': True},

     ]


for i in imgpaths:

    # print the folder we are working on
    print("------------------------------------------------------------")
    print("Searching for Episode.h5 files in ",i['Folder'])
    print("------------------------------------------------------------")  

    # get all subdirs
    subdirs = rf.list_all_subdirs(phile_name = i['Folder'])

    # filter subdirs to only those that contain Episode.h5
    subdirs = [subi for subi in subdirs if 'Episode.h5' in os.listdir(subi)]  # filter out directories that do not contain Episode.h5

    # loop over subdirectories
    failed_subi = []; success_subi = []
    for subi in subdirs:

        # get subcontents
        try:
            dir_contents = os.listdir(subi)
            logger = ''
        except:
            print("Skipping",subi)
            continue

        # load the Episode.h5 file
        print("Loading chunks of your Episode.h5 file in",subi)
        epi_path = os.path.join(subi, 'Episode.h5')
        if not os.path.exists(epi_path):
            print("No Episode.h5 found in",subi)
            continue

        # collect the first and last timestamps
        frame_rate     = 1000  # 1000 Hz
        first_time_min = 10  # User-specified first segment in minutes
        last_time_min  = 10   # User-specified last segment in minutes

        # converted to samples
        n_samples_first = frame_rate * 60 * first_time_min
        n_samples_last  = frame_rate * 60 * last_time_min

        with h5py.File(epi_path, 'r') as dataIn:

            # Lazily load only the required slices of data
            piezo_first = dataIn['AI']['PiezoMonitor'][:n_samples_first] if n_samples_first > 0 else []
            piezo_last = dataIn['AI']['PiezoMonitor'][-n_samples_last:] if n_samples_last > 0 else []
            piezo_concat = np.concatenate((piezo_first, piezo_last)) if len(piezo_first) + len(piezo_last) > 0 else []
            if len(piezo_concat) > 0:
                piezo_concat = piezo_concat / np.max(piezo_concat)  # Normalize

            # imaging frames data, sliced for our chunks of interest
            frameout_first = dataIn['DI']['FrameOut'][:n_samples_first] if n_samples_first > 0 else []
            frameout_last = dataIn['DI']['FrameOut'][-n_samples_last:] if n_samples_last > 0 else []
            frameout_concat = np.concatenate((frameout_first, frameout_last)) if len(frameout_first) + len(frameout_last) > 0 else []
            if len(frameout_concat) > 0:
                frameout_concat = frameout_concat / np.max(frameout_concat)  # Normalize

            # check for a misalignment between your frameout and piezo signals

            # generate the plot
            fig, ax = plt.subplots()
            ax.clear()
            if len(piezo_concat) > 0 and len(frameout_concat) > 0:
                ax.plot(piezo_concat, label=f'Piezo Norm (First {first_time_min} min & Last {last_time_min} min)')
                ax.plot(frameout_concat, label=f'FrameOut (First {first_time_min} min & Last {last_time_min} min)')
                ax.legend()
            max_length = max(len(piezo_concat), len(frameout_concat))
            ax.set_xlim(0, max_length - 1)
            ax.set_ylim(min(np.min(piezo_concat), np.min(frameout_concat)),
                        max(np.max(piezo_concat), np.max(frameout_concat)) if len(piezo_concat) > 0 and len(frameout_concat) > 0 else (0, 1))
            plt.title(f'Piezo and FrameOut Normalized Signals\n{Path(subi).name}')
            plt.xlabel('Samples')
            plt.ylabel('Normalized Value')
            plt.show()


            self.ax.clear()
            if len(piezo_concat) > 0 and len(frameout_concat) > 0:
                self.ax.plot(piezo_concat, label=f'Piezo Norm (First {first_time_min} min & Last {last_time_min} min)')
                self.ax.plot(frameout_concat, label=f'FrameOut (First {first_time_min} min & Last {last_time_min} min)')
                self.ax.legend()

            max_length = max(len(piezo_concat), len(frameout_concat))
            self.ax.set_xlim(0, max_length - 1)
            self.ax.set_ylim(min(np.min(piezo_concat), np.min(frameout_concat)),
                                max(np.max(piezo_concat), np.max(frameout_concat)) if len(piezo_concat) > 0 and len(frameout_concat) > 0 else (0, 1))

            self.canvas.draw()
