# SCRIPT_recurseConvert
"""
This code is designed to take folder paths, convert files, run suite2p, and run constrained foopsi

Important*: You should be able to copy files into folders while running this code. The worst issue will be memory/time to copy.
                This code has a built in variable "busyBee" that searches for a busy folder. If detected, it will skip said folder until later.

Here is the general order of operations:
    1) Search for Image----.raw files
    2) Search for converted img.tif files
        a) If Image----.raw was not converted to img.tif, this code will convert your data
    3) Search for suite2p folders
        a) If you have run suite2p, then the datafolder will contain a suite2p folder.
            If said folder is not discovered, this code will perform suite2p's algorithms
    4) Search for deconvolution and denoising steps using detrending and constrained foopsi
        a) This code will search for variables "C.npy" and "S.npy" 
        b) If not discovered, this code will store those variables in the suite2p folder

    This code will also save two .csv files in the 'Folder' path:
        'recurseConvertComplete.csv' and 'recurseConvertFailed.csv'

The user cares about this variable:
    imgpaths: a list of dictionaries, eaching containing a folder to loop on 'Folder'
                and whether to use spellman ops 'SpellOps'

`imgpaths` are defined as such:
    imgpaths = [
    {'Folder': r"/path/to/your/folder", 'SpellOps': True},
    {'Folder': r"/path/to/your/folder", 'SpellOps': False},
    ...
    ]

There are multiple optional arguments per each folder, the defaults are as such:
    'SpellOps':           True  - Uses spellman ops
    'imgReplace':         False - Does NOT replace existing img.tif
    's2pReplace':         False - Does NOT replace existing suite2p folder
    'cellRegReplace':     False - Does NOT replace existing cellreg files
    'cleanTracesTreplace: False - Does NOT replace existing C.npy and S.npy files
    'remTif':             False - Does NOT erase img.tif file

John Stout
Alex Mitchel wrote list_all_subdirs in MATLAB, which was converted to python
Tim wrote cleanup_raw_traces in MATLAB, which was converted to Python
CoPilot helped with parallel processing steps.
Detrending and denoising written by Andres Grosmark and converted to Python

Last edit: 
    - 10/21/2024: Fixed parallel processing issue where jobs would be returned and placed out of order, impairing indexing
    - 10/22/2024: Added additional fail-safe for misaligned frames due to parallel processing. Such fail-safe cancels out in thorfuns if misalignments are detected. This ensures that the frame you are reading is in the correct order.
                    - Added a save-out for successful and failed attempts at converting data as .csv files in the 'Folder' root given to imgpaths.
    - 10/23/2024: Added search for deconvolution steps and running tests
    - 10/31/2024: Added option to remove img.tif files when you're finished with them to conserve memory
    - 11/01/2024: Added option to replace S.npy and C.npy files
    - 11/29/2024: Added behavioral conversion and options on whether to run a forever loop or iterate

REFERENCES:
    Constrained Foopsi
        * Pnevmatikakis et al. 2016. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037
        * Machado et al. 2015. Cell 162(2):338-350
        * Code was taken from the CaImAn package: https://github.com/flatironinstitute/CaImAn
    Suite2p
        * Pachitariu, M., et al. (2016). bioRxiv, https://www.biorxiv.org/content/10.1101/061507v2.abstract
    Denoising steps and constrained foopsi on our data:
        * Grosmark et al. (2021). Nature Neuroscience, 24(11), 1574-1585.
        * Spellman et al., (2021). Cell, 184(10), 2750-2766.

TODO: At somepoint, merge this code with synchronizeToDropbox
"""

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

# whether or not to run iteratively or recursively
run_opts = ['forever', 'iterate']
run_method = 1 # change as needed. Set to 0 if forever loop

if run_opts[run_method]=='iterate':
    print("Iterating through available folders. Code will not iterate forever!")
else:
    print("Forever loop starting in 3...2.....1.......3")

# recursive method
imgpaths = dict()

# TODO: ADD 'DateReplace': 'Month, Day, Year'
# TODO: ADD mechanism to update the suite2p folder if something about the naming convention changed. So ops would need updating.
imgpaths = [

    {'Folder': r"E:\L6 Experiments\L612",                               'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},


    # John Folders
    {'Folder': r"E:\L6 Experiments\L613",                               'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},

    {'Folder': r"E:\L6 Experiments\L615",                               'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"E:\L6 Experiments\L614",                               'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"E:\L6 Experiments\L616",                               'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},

    {'Folder': r"E:\L6 Experiments\L609-pan",                           'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},    
    {'Folder': r"E:\L6 Experiments\L608",                               'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"E:\L6 Experiments\L607T4",                             'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"E:\L6 Experiments\L1",                                 'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"E:\L6 Experiments\L6R11",                              'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"E:\L6 Experiments\L605",                               'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"E:\L6 Experiments\T30",                                'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    
    {'Folder': r"F:\John\L6 Experiments\recordings_panneuronal\T-30",   'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"F:\John\L6 Experiments\recordings_L5CT\L6-05",         'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    {'Folder': r"F:\John\L6 Experiments\recordings_IT\L607-T4",         'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'cleanTracesReplace': True, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},

    # MDT folder
    {'Folder': r"E:\ThalamicRec\MDT1", 'SpellOps': True, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': False, 'cleanTracesReplace': False, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},   


    # peyton/alex
    #{'Folder': r"H:\Peyton\L602",     'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': False, 'cleanTracesReplace': False, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},
    #{'Folder': r"H:\Peyton\L607",     'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': False, 'cleanTracesReplace': False, 'remTif': False, 'behReplace': False, 'rerunSuite2p_keepReg': True},




    ]

# set defaults
for i in imgpaths:
    if 'SpellOps' not in i:
        i['SpellOps'] = True
        print("Default SpellOps==True for", i)
    if 'imgReplace' not in i:
        i['imgReplace'] = False
        print("Default imgReplace==False for", i)
    if 's2pReplace' not in i:
        i['s2pReplace'] = False
        print("Default s2pReplace==False for", i)
    if 'cellRegReplace' not in i:
        i['cellRegReplace'] = False
        print("Default cellRegReplace==False for", i)
    if 'remTif' not in i:
        i['remTif'] = False
        print("Default remTif==False for", i)
    if 'cleanTracesReplace' not in i:
        i['cleanTracesReplace'] = False
        print("Default cleanTracesReplace==False for", i)
    if 'behReplace' not in i:
        i['behReplace'] = False
        print("Default behReplace==False for", i)
    if 'rerunSuite2p_keepReg' not in i:
        i['rerunSuite2p_keepReg'] = False

# don't run in parallel bc thorfuns.RawToTif.convert('max_proj') uses parallel computing
next = 0
while next == 0:
    for i in imgpaths:
        print("Searching for .Raw files in ",i['Folder'])
        print("Please note that this loop will only end if you cancel the code")

        # identify if the user wants to use the preset spell ops file
        if i['SpellOps'] == True:
            print("Using Spellman params")

            # ops path
            ops_path = os.path.join(root,'timspellman','Python','suite2p_ops')

            # load ops data
            alt_ops = np.load(os.path.join(ops_path,'spellman_ops.npy'), allow_pickle=True).item()
            alt_ops['tau'] = 0.7 # for gcamp 6f

        else:
            print("Using default s2p params")
            alt_ops = None     

        # how to handle existing imgs and s2p folders
        img_replace        = i['imgReplace']
        s2p_replace        = i['s2pReplace']    
        cr_replace         = i['cellRegReplace']
        rem_tif            = i['remTif']
        cleanTracesReplace = i['cleanTracesReplace']
        behReplace         = i['behReplace']

        if img_replace == True:
            print("img_replace==True, wiping and replacing img.tif")
        else:
            print("img_replace==False, will NOT wipe and replace existing img.tif")
        if s2p_replace == True:
            print("s2p_replace==True, wiping and replacing suite2p folder")
        else:
            print("s2p_replace==False, will NOT wipe and replace suite2p folder")
        
        # get all subdirs
        subdirs = rf.list_all_subdirs(phile_name = i['Folder'])

        # for troubleshooting
        #if next == 1:
            #break

        # loop over subdirectories
        failed_subi = []; success_subi = []
        for subi in subdirs:

            # get subcontents
            try:
                dir_contents = os.listdir(subi)
            except:
                print("Skipping",subi)
                continue

            # search for .raw file
            rawSearch = [k for k in dir_contents if '.raw' in k and 'Image' in k]

            # search for behavior
            behSearch = [k for k in dir_contents if '.h5' in k and 'Episode' in k]

            # for troubleshooting
            #if len(rawSearch) > 0:ChC
                #next = 1
                #break

            # check if folder is busy
            busyBee = rf.is_folder_busy(subi)

            # if the .Raw file was discovered, then executive img.tif and suite2p conversion
            if len(rawSearch) > 0 and busyBee is False:

                # search for key files
                imgFound = []; s2pFound = []
                imgFound = len([j for j in dir_contents if 'img.tif' in j])
                s2pFound = len([j for j in dir_contents if 'suite2p' in j])

                # check your session title for 'opto'
                if 'opto' in os.path.split(subi)[-1].lower():
                    led_artifact='y'
                else:
                    led_artifact='n'

                # search for suite2p and deconvolved data
                dcSearch = []; dcFound = []; dcSearched = []; crSearched = []
                if s2pFound > 0:
                    creation_time = os.path.getctime(os.path.join(subi,'suite2p'))
                    creation_date = datetime.fromtimestamp(creation_time).strftime('%B %d, %Y')
                    
                    # the new code is to be applied to old sessions
                    #if creation_date < 'October 22, 2024':
                        #s2pFound = 0

                    # find whether deconvolution steps were already performed
                    dcSearch = [i for i in os.listdir(os.path.join(subi,'suite2p','plane0'))]
                    dcSearched  = [i for i in dcSearch if 'C.npy' in i or 'S.npy' in i]

                    # search for cellreg
                    crSearched  = [i for i in dcSearch if 's2pCellReg.mat' in i]

                if imgFound > 0:
                    creation_time = os.path.getctime(os.path.join(subi,'img.tif'))
                    creation_date = datetime.fromtimestamp(creation_time).strftime('%B %d, %Y')

                    # I might need to go back and fix this
                    #if creation_date < 'October 10, 2024':
                    #    imgFound = 0

                # if the img.tif file was NOT found or if you want to REPLACE
                try:
                    if imgFound == 0 or img_replace == True:
                        # for troubleshooting
                        #if len(rawSearch) > 0:
                            #next = 1
                            #break                    
                        print("No img.tif file discovered. Writing file to:", subi)
                        
                        # track timing
                        code_start = time.process_time()

                        # run conversion
                        thorfuns.RawToTif(filepath=subi).convert(method='max_proj', # don't change this for now
                                                            chunker=1000, # impacts the number of samples used for parallel computing
                                                            led_artifacts=led_artifact,
                                                            wipe_and_replace=img_replace)
                        
                        # update the subdirs folder
                        subdirs = rf.list_all_subdirs(phile_name = i['Folder'])

                        # report timing
                        process_end = time.process_time()
                        print(f"Total time in RawToTif: {(process_end - code_start)/60:.2f} minutes")

                    # if suite2p folder was NOT found, or if you want to REPLACE
                    if s2pFound == 0 or s2p_replace == True or i['rerunSuite2p_keepReg'] == True:
                        print("Running suite2p to", subi)
                        # track timing
                        code_start = time.process_time()   

                        # run conversion   
                        s2pfuns.fast_suite2p(imgpath=os.path.join(subi,'img.tif'), 
                                            savepath='', 
                                            gcamp='6f', 
                                            alt_ops=alt_ops,
                                            wipe_and_replace=s2p_replace)
                        
                        # update the subdirs folder
                        subdirs = rf.list_all_subdirs(phile_name = i['Folder'])

                        # report timing
                        process_end = time.process_time()
                        print(f"Total time in suite2p: {(process_end - code_start)/60:.2f} minutes")

                    # run denoising and constrained foopsi if:
                    # 1) dcSearched < 2: suite2p folder was found but the C and S variables missing
                    # 2) s2pFound==0: the suite2p folder was not found, was just created, and now you can add those variables
                    if len(dcSearched) < 2 or s2pFound==0 or cleanTracesReplace==True:
                        print("Postprocessing session:", subi)                        
                        code_start = time.process_time()  
                        s2pfuns.postProcess(s2ppath=os.path.join(subi,'suite2p','plane0')).cleanup_raw_traces(run_parallel=False)  
                        procFOVess_end = time.process_time()
                        print(f"Total time to postprocess: {(process_end - code_start)/60:.2f} minutes")
                    
                    # if the cellreg data is not present (as would happen if you wipe and replace)
                    if len(crSearched)==0 or s2pFound==0 or cr_replace==True:
                        print("Preparing cellReg data for session:", subi)                        
                        code_start = time.process_time() 
                        sessreg.suite2pToCellReg(fnames = subi, mask_overlap = True) 
                        process_end = time.process_time()   

                    # option to remove the .tif file bc there exists redundancy in data.bin
                    if rem_tif == True:
                        print("Erasing img.tif file:", subi)                        
                        thorfuns.remTif(os.path.join(subi,'img.tif'))

                    success_subi.append(subi) # save variable for reporting
                except:
                    failed_subi.append(subi) # save variable for reporting
                    print("Could not process",subi)

            # convert behavior
            if len(behSearch) > 0 and busyBee is False:

                # convert - haven't tested the behConvSearch
                behConvSearch = [k for k in dir_contents if 'beh.mat' in k]
                try:
                    if behReplace == True or len(behConvSearch) == 0:
                        thorfuns.importThorsync(bpath = subi)
                except:
                    print("Failed to convert behavioral data in:",subi)

        # save results of conversions
        if len(success_subi) > 0 or len(failed_subi) > 0:
            # Save list to .csv file
            with open(os.path.join(subi,'recurseConvertComplete.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(success_subi)
            with open(os.path.join(subi,'recurseConvertFailed.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(failed_subi)                        
    
    # to exit or not to exit?
    if run_opts[run_method]=='iterate':
        next = 1