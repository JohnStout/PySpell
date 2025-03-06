# SCRIPT_recurseConvert
"""
This code is designed to take folder paths, convert files, run suite2p, and run constrained foopsi

TODO: ISSUES:
    1) You cannot merge cells without introducing the C and S generation into the suite2p GUI
        This means that all merging steps should be performed before calculating C and S
        The same is true for cellReg, which shouldn't be run before merging is solved

***************** CONSTRAINED FOOPSI IS RUN USING THE OASIS BACKEND **************
            * Friedrich, J., Zhou, P., & Paninski, L. (2017). Fast online deconvolution of calcium imaging data. PLoS computational biology, 13(3), e1005423.
        
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

UPDATES: 
    - 10/21/2024: Fixed parallel processing issue where jobs would be returned and placed out of order, impairing indexing
    - 10/22/2024: Added additional fail-safe for misaligned frames due to parallel processing. Such fail-safe cancels out in thorfuns if misalignments are detected. This ensures that the frame you are reading is in the correct order.
                    - Added a save-out for successful and failed attempts at converting data as .csv files in the 'Folder' root given to imgpaths.
    - 10/23/2024: Added search for deconvolution steps and running tests
    - 10/31/2024: Added option to remove img.tif files when you're finished with them to conserve memory
    - 11/01/2024: Added option to replace S.npy and C.npy files
    - 11/29/2024: Added behavioral conversion and options on whether to run a forever loop or iterate
    - 12/??/2024:                     
                                        IMPORTANT ADDITION:
                        suite2p alt_ops were redefined in a critical way does increase false positives but captures everything. Additional classifier needed.
                        These ops take much of spellOps but do not trash overlapping components. This was actually updated in 
    - 1/10/2025: Added 'saveCleanedF' that saves out EMD denoised and sgolay/mad detrended F signals.
                                            CRITICAL: 
                        cleanupRawTraces is NOT run on EMD denoised data because the AR modeling in OASIS is designed to handle noisy data and this leads to overfitting and potentially increased false positives
                        cleanupRawTraces IS run on sgolay/mad detrending
                        However, EMD denoising +sgolay/mad detrending provide a really nice F signal and so 'saveCleanedF' was added so the user can save out
                            this cleaner signal in case they wanted to use the F trace for analysis

    - 1/14/2025: Included a datetime variable for rerunning code 's2p_update' and 'clean_traces_update'.
                    I selectively added these variables because suite2p will often be a source of changing parameters and
                    so will cleaning raw traces.
                    SO, if you specify the date of your change, this code will update the folders accordingly.
                                        MAJOR UPDATE:
                    The code now detects updates to img.tif files and if such update happened after a suite2p file was generated,
                    the suite2p file will be wiped and replaced. Moreover, all following steps (cellreg, cleanupRawTraces, cleanF)
                    are rerun if an update to suite2p is detected. This rerun ONLY happens if such file was generated in the first place.

    - 1/15/2025: Fixed a bug on the .csv writeout and now only save failed conversions
                 Fixed an issue where rerun items (like rerun suite2p) were a forced update and now have it such that
                    the forced update requires timedate data. For example, a detection of stat.npy creation happening after
                    the creation of C.npy would indicate that an update to suite2p was made but such an update hasn't been applied to C.npy
                 Updated datetime variables
                                        MAJOR UPDATE
                Removed the modifier items like 'rerunSuite2p_keepReg' that selectively rerun suite2p.
                If a major update was performed on suite2p, it would be applied on all sessions in the imgpaths folder
                and as such, will automatically be applied based on datetime updates.

    - 3/5/2025: Minor updates to code. 
                    - Fixed a bug where sessions were being skipped
                    - Fixed CellReg file naming convention and provided a tool to replace old naming
                    - Added a warning for imf improper fit

REFERENCES:
    Constrained Foopsi
        * Pnevmatikakis et al. 2016. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037
        * Machado et al. 2015. Cell 162(2):338-350
        * Code was taken from the CaImAn package: https://github.com/flatironinstitute/CaImAn
    Suite2p
        * Pachitariu, M., et al. (2016). bioRxiv, https://www.biorxiv.org/content/10.1101/061507v2.abstract
    Detrending steps and constrained foopsi on our data:
        * Friedrich, J., Zhou, P., & Paninski, L. (2017). Fast online deconvolution of calcium imaging data. PLoS computational biology, 13(3), e1005423.
        * Grosmark et al. (2021). Nature Neuroscience, 24(11), 1574-1585.
        * Spellman et al., (2021). Cell, 184(10), 2750-2766.
    EMD analysis for denoising
        * Tim Spellman

TODO: ISSUE WITH CODE JUST RERUNNING REGISTRATION FOR SUITE2P
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

# ------------------------------------------ #
# -------PARAMETER SPACE-------------------- #

# whether or not to run iteratively or recursively
run_opts = ['forever', 'iterate']
run_method = 1 # change as needed. Set to 0 if forever loop

# suite2p rerun date
s2p_update = 'January 10, 2025, 20:00:00' # If you set this to None, then it will ignore a date
s2p_update_datetime = datetime.strptime(s2p_update, '%B %d, %Y, %H:%M:%S')

# cleanupRawTraces update
clean_traces_update = 'January 10, 2025, 20:00:00'
clean_traces_datetime = datetime.strptime(clean_traces_update, '%B %d, %Y, %H:%M:%S')

# ------------------------------------------ #
# ------------------------------------------ #

# run parallel processing
run_parallel = False

if run_opts[run_method]=='iterate':
    print("Iterating through available folders. Code will not iterate forever!")
else:
    print("Forever loop starting in 3...2.....1.......3")

# recursive method
imgpaths = dict()

# ---------------------------------------------------------------------------- #
# ---------------DEFINE FOLDERS BELOW----------------------------------------- #

# TODO: ADD 'DateReplace': 'Month, Day, Year'
imgpaths = [

    # John folders

    # {'Folder': r"D:\L6 Experiments\L608",                               'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"D:\L6 Experiments\L612",                               'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"D:\L6 Experiments\L613",                               'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"D:\L6 Experiments\L614",                               'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"D:\L6 Experiments\L616",                               'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
     #{'Folder': r"D:\L6 Experiments\L607T4",                             'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
     #{'Folder': r"D:\L6 Experiments\T30",                                'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},

    # {'Folder': r"H:\Layer6\L609-pan",                                   'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},    
    # {'Folder': r"H:\Layer6\L1",                                         'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"H:\Layer6\L6R11",                                      'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"H:\Layer6\L605",                                       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"H:\Layer6\L645",                                       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    #{'Folder': r"E:\Layer6\L615",                                       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},    

    # {'Folder': r"F:\John\L6 Experiments\recordings_panneuronal\T-30",   'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"F:\John\L6 Experiments\recordings_L5CT\L6-05",         'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    # {'Folder': r"F:\John\L6 Experiments\recordings_IT\L607-T4",         'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},

    # MDT folder
    #{'Folder': r"E:\ThalamicRec\MDT1", 'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'cellRegReplace': True, 'remTif': False, 'behReplace': False, 'saveCleanedF': False},      
    {'Folder': r"I:\Alex",         'SpellOps': False, 'imgReplace': False, 's2pReplace': True, 'remTif': False, 'behReplace': False, 'saveCleanedF': True},
    #rename the above to Alex's folder!



    # peyton/alex
    {'Folder': r"Z:\Peyton\L602",     'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False},
    {'Folder': r"Z:\Peyton\L607",     'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False},
    {'Folder': r"Z:\Peyton\1",        'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False},
    {'Folder': r"Z:\Peyton\B02",      'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False},
    {'Folder': r"Z:\Peyton\T27",      'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False},
    {'Folder': r"Z:\Peyton\B03",      'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False},
    {'Folder': r"Z:\Peyton\48",       'SpellOps': False, 'imgReplace': False, 's2pReplace': False, 'remTif': False, 'behReplace': False, 'saveCleanedF': False},

    ]

# ---------------DEFINE FOLDERS ABOVE----------------------------------------- #
# ---------------------------------------------------------------------------- #

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
    if 'remTif' not in i:
        i['remTif'] = False
        print("Default remTif==False for", i)
    if 'behReplace' not in i:
        i['behReplace'] = False
        print("Default behReplace==False for", i)
    if 'saveCleanedF' not in i:
        i['saveCleanedF'] = False

# don't run in parallel bc thorfuns.RawToTif.convert('max_proj') uses parallel computing
next = 0
while next == 0:
    for i in imgpaths:
        print("Searching for .Raw files in ",i['Folder'])
        #print("Please note that this loop will only end if you cancel the code")

        # identify if the user wants to use the preset spell ops file
        if i['SpellOps'] == True:
            print("Using Spellman params")

            # ops path
            ops_path = os.path.join(root,'timspellman','Python','suite2p_ops')

            # load ops data
            alt_ops = np.load(os.path.join(ops_path,'spellman_ops.npy'), allow_pickle=True).item()
            alt_ops['tau'] = 0.7 # for gcamp 6f
            alt_ops['fs'] = 7.5

        else:
            print("Using default s2p params")
            alt_ops = None     

        # how to handle existing imgs and s2p folders
        img_replace        = i['imgReplace']
        s2p_replace        = i['s2pReplace']    
        rem_tif            = i['remTif']
        behReplace         = i['behReplace']
        saveCleanedF       = i['saveCleanedF']

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
            print('Searching for busy folders...')
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

                    # Use creation time for suite2p because variables get removed
                    stat_search = [i for i in os.listdir(os.path.join(subi, 'suite2p', 'plane0')) if 'stat' in i]
                    if len(stat_search) > 0:
                        creation_time_suite2p = os.path.getctime(os.path.join(subi,'suite2p','plane0','stat.npy'))
                        creation_date_suite2p = datetime.fromtimestamp(creation_time_suite2p).strftime('%B %d, %Y, %H:%M:%S')
                        datetime_suite2p      = datetime.strptime(creation_date_suite2p, '%B %d, %Y, %H:%M:%S') 
                    else:
                        print("No stat file found, rerunning suite2p...")
                        datetime_suite2p = False
                        s2pFound = 0

                    # check the registered data.bin file
                    bin_search = [i for i in os.listdir(os.path.join(subi, 'suite2p', 'plane0')) if 'data.bin' in i]
                    if len(bin_search) > 0:
                        creation_time_binary = os.path.getctime(os.path.join(subi,'suite2p','plane0','data.bin'))
                        creation_date_binary = datetime.fromtimestamp(creation_time_binary).strftime('%B %d, %Y, %H:%M:%S')
                        datetime_binary      = datetime.strptime(creation_date_binary, '%B %d, %Y, %H:%M:%S') 
                    else:
                        print("No binary file detected. Running suite2p...")
                        datetime_binary = False
                        s2pFound = 0

                    # find whether deconvolution steps were already performed
                    dcSearch = [i for i in os.listdir(os.path.join(subi,'suite2p','plane0'))]
                    dcSearched  = [i for i in dcSearch if 'C.npy' in i or 'S.npy' in i]

                    # search for cellreg
                    crSearched  = [i for i in dcSearch if 's2pCellReg.mat' in i]

                if imgFound > 0:

                    # log datetime information
                    creation_time_img = os.path.getmtime(os.path.join(subi,'img.tif'))
                    creation_date_img = datetime.fromtimestamp(creation_time_img).strftime('%B %d, %Y, %H:%M:%S')
                    datetime_img      = datetime.strptime(creation_date_img,     '%B %d, %Y, %H:%M:%S')    

                # if the img.tif file was NOT found or if you want to REPLACE
                try:

                    # --------------------------------------------------------- #
                    # -------------------- CONVERT IMG DATA ------------------- #

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
                                                            wipe_and_replace=img_replace,
                                                            run_parallel=run_parallel)
                        
                        # update the subdirs folder
                        subdirs = rf.list_all_subdirs(phile_name = i['Folder'])

                        # report timing
                        process_end = time.process_time()
                        print(f"Total time in RawToTif: {(process_end - code_start)/60:.2f} minutes")

                    # ------------------------------------------------------------- #
                    # -------------------- RUNNING SUITE2P ------------------------ #

                    # if suite2p folder was NOT found, or if you want to REPLACE, or if img.tif file was modified after the binary registered suite2p data.bin file...
                    if s2pFound == 0 or s2p_replace == True:
                        
                        # track timing
                        code_start = time.process_time()   

                        # if no file is found, run suite2p
                        print("Running suite2p and saving to:", subi)

                        # run conversion   
                        s2pfuns.fast_suite2p(imgpath=os.path.join(subi,'img.tif'), 
                                            savepath='', 
                                            gcamp='6f', 
                                            alt_ops=alt_ops,
                                            wipe_and_replace=s2p_replace)
                        
                        process_end = time.process_time() # report
                        print(f"Total time in suite2p: {(process_end - code_start)/60:.2f} minutes")

                    # search for an update to the img.tif file or a forced update to suite2p
                    elif s2pFound==1:

                        # start tracker
                        code_start = time.process_time() 

                        # if datetimes are out of order, run suite2p  
                        if datetime_suite2p < s2p_update_datetime:
                            print("Suite2p Update detected. Rerunning and saving to:", subi)
                        
                        if datetime_suite2p < datetime_img or datetime_binary < datetime_img:
                            print("Update to img.tif file detected. Deleting old suite2p file and rerunning.")
                            s2p_replace = True

                        if datetime_suite2p < s2p_update_datetime or datetime_suite2p < datetime_img or len(stat_search) == 0 or datetime_binary < datetime_img:

                            # run conversion   
                            s2pfuns.fast_suite2p(imgpath=os.path.join(subi,'img.tif'), 
                                                savepath='', 
                                                gcamp='6f', 
                                                alt_ops=alt_ops,
                                                wipe_and_replace=s2p_replace)
                            
                            # reset for sanity
                            s2p_replace = i['s2pReplace']

                            # recapture timing data
                            creation_time_suite2p = os.path.getctime(os.path.join(subi,'suite2p','plane0','stat.npy'))
                            creation_date_suite2p = datetime.fromtimestamp(creation_time_suite2p).strftime('%B %d, %Y, %H:%M:%S')
                                    
                            # update the subdirs folder
                            subdirs = rf.list_all_subdirs(phile_name = i['Folder'])
                            process_end = time.process_time() # report
                            print(f"Total time in suite2p: {(process_end - code_start)/60:.2f} minutes")
                        
                            # find whether deconvolution steps were already performed
                            dcSearch = [i for i in os.listdir(os.path.join(subi,'suite2p','plane0'))]
                            dcSearched  = [i for i in dcSearch if 'C.npy' in i or 'S.npy' in i]

                            # search for cellreg
                            crSearched  = [i for i in dcSearch if 'CellReg.mat' in i]

                    # --------------------------------------------------------- #
                    # -------------------- POST PROCESSING -------------------- #

                    # run denoising and constrained foopsi if:
                    # 1) dcSearched < 2: suite2p folder was found but the C and S variables missing
                    # 2) s2pFound==0: the suite2p folder was not found, was just created, and now you can add those variables
                    if len(dcSearched) < 2 or s2pFound==0:

                        print("Postprocessing session:", subi)                        
                        code_start = time.process_time()  
                        s2pfuns.postProcess(s2ppath=os.path.join(subi,'suite2p','plane0')).cleanup_raw_traces()  
                        procFOVess_end = time.process_time()
                        print(f"Total time to postprocess: {(procFOVess_end - code_start)/60:.2f} minutes")
                    
                    # replacement traces are requested, s2p folder present and the C/S files also present
                    elif len(dcSearched) >= 2:

                        # grab modification data
                        creation_time_traces = os.path.getmtime(os.path.join(subi,'suite2p','plane0','C.npy'))
                        creation_date_traces = datetime.fromtimestamp(creation_time_traces).strftime('%B %d, %Y, %H:%M:%S')
                        datetime_traces      = datetime.strptime(creation_date_traces, '%B %d, %Y, %H:%M:%S')

                        # rerun if suite2p has been updated OR if cleanupRawTraces has been updated.
                        # in otherwords, if you've changed the suite2p parameter space or method for cleaning raw traces, then
                        # you should rerun this.
                        if datetime_traces < s2p_update_datetime or datetime_traces < clean_traces_datetime or datetime_traces < datetime_suite2p:
                            print("Update to suite2p or cleanupRawTraces detected. Forced update to C and S.npy variables...")
                            print("Postprocessing session:", subi)                        
                            code_start = time.process_time()  
                            s2pfuns.postProcess(s2ppath=os.path.join(subi,'suite2p','plane0')).cleanup_raw_traces()  
                            procFOVess_end = time.process_time()
                            print(f"Total time to postprocess: {(procFOVess_end - code_start)/60:.2f} minutes")
                    
                    # --------------------------------------------------------- #
                    # -------------------- CELL REG --------------------------- #

                    # if the cellreg data is not present (as would happen if you wipe and replace)
                    if len(crSearched)==0 or s2pFound==0:

                        print("Preparing cellReg data for session:", subi)                        
                        code_start = time.process_time() 

                        # create cell reg file name
                        reg_file_name = os.path.split(subi)[-1][0:20]
                        if reg_file_name[-1]=='_':
                            reg_file_name = reg_file_name+'CellReg.mat'
                        else:
                            reg_file_name = reg_file_name+'_CellReg.mat'

                        # make cellReg file
                        sessreg.suite2pToCellReg(fnames = subi, mask_overlap = True, save_name=reg_file_name) 
                        process_end = time.process_time()  

                    # if file detected or cellRegReplace == True
                    elif len(crSearched)>0:

                        # identify when the cellReg code was run
                        creation_time_reg = os.path.getmtime(os.path.join(subi,'suite2p','plane0','s2pCellReg.mat'))
                        creation_date_reg = datetime.fromtimestamp(creation_time_reg).strftime('%B %d, %Y, %H:%M:%S')
                        datetime_reg      = datetime.strptime(creation_date_reg, '%B %d, %Y, %H:%M:%S')

                        # if you have recently updated suite2p, then you MUST update cellREg
                        if datetime_reg < s2p_update_datetime or datetime_reg < datetime_suite2p:

                            print("Suite2p update detected. Forced rerun of cellReg mask saveout...")
                            code_start = time.process_time() 

                            # create cell reg file name
                            reg_file_name = os.path.split(subi)[-1][0:20]
                            if reg_file_name[-1]=='_':
                                reg_file_name = reg_file_name+'CellReg.mat'
                            else:
                                reg_file_name = reg_file_name+'_CellReg.mat'

                            # make cellReg file                        
                            sessreg.suite2pToCellReg(fnames = subi, mask_overlap = True, save_name=reg_file_name) 
                            process_end = time.process_time()   

                        else:

                            # rename from old file convention
                            crSearched  = [i for i in dcSearch if 's2pCellReg.mat' in i]

                            if len(crSearched) > 0:
                                print("Detected old s2pCellReg.mat file convention. Renaming to match your session.")
                                
                                # create cell reg file name
                                reg_file_name = os.path.split(subi)[-1][0:20]
                                if reg_file_name[-1]=='_':
                                    reg_file_name = reg_file_name+'CellReg.mat'
                                else:
                                    reg_file_name = reg_file_name+'_CellReg.mat'
                                
                                os.rename(os.path.join(subi,'suite2p','plane0','s2pCellReg.mat'), os.path.join(subi,'suite2p','plane0',reg_file_name))                            

                    # --------------------------------------------------------- #
                    # -------------------- OPTION FOR A CLEANED F ------------- #

                    # search for cleaned F
                    cFSearched  = [i for i in dcSearch if 'F_clean.mat' in i]
                    if saveCleanedF==True and len(cFSearched)==0:

                        print("Saving out EMD denoised and sgolay/mad detrended signals for:", subi)                        
                        code_start = time.process_time()  
                        s2pfuns.postProcess(s2ppath=os.path.join(subi,'suite2p','plane0')).save_modified_f() 
                        process_end = time.process_time()
                        print(f"Total time to postprocess: {(process_end - code_start)/60:.2f} minutes")
                    
                    elif len(cFSearched)>0:

                        # check datetime
                        # if you want to save out the cleaned F
                        creation_time_cleanF = os.path.getmtime(os.path.join(subi,'suite2p','plane0','F_clean.mat'))
                        creation_date_cleanF = datetime.fromtimestamp(creation_time_cleanF).strftime('%B %d, %Y, %H:%M:%S')
                        datetime_cleanF      = datetime.strptime(creation_date_cleanF, '%B %d, %Y, %H:%M:%S')

                        # rerun if a suite2p update was detected
                        if datetime_cleanF < s2p_update_datetime or datetime_cleanF < datetime_suite2p:
                            print("Suite2p Update detected. Forced rerun of cleaned F saveout")
                            print("Saving out EMD denoised and sgolay/mad detrended signals for:", subi)                        
                            code_start = time.process_time()  
                            s2pfuns.postProcess(s2ppath=os.path.join(subi,'suite2p','plane0')).save_modified_f() 
                            process_end = time.process_time()
                            print(f"Total time to postprocess: {(process_end - code_start)/60:.2f} minutes")

                    # option to remove the .tif file bc there exists redundancy in data.bin
                    if rem_tif == True:
                        print("Erasing img.tif file:", subi)                        
                        thorfuns.remTif(os.path.join(subi,'img.tif'))

                    success_subi.append(subi) # save variable for reporting

                except:

                    failed_subi.append(subi) # save variable for reporting
                    print("Could not process",subi)

            # --------------------------------------------------------- #
            # -------------------- CONVERT BEHAVIOR ------------------- #

            # convert behavior
            #TODO: Once you update behavioral code, include a datetime clause
            if len(behSearch) > 0 and busyBee is False:

                # convert - haven't tested the behConvSearch
                behConvSearch = [k for k in dir_contents if 'beh.mat' in k]
                try:
                    if behReplace == True or len(behConvSearch) == 0:
                        thorfuns.importThorsync(bpath = subi)
                except:
                    print("Failed to convert behavioral data in:",subi)

            # --------------------------------------------------------- #
            # --------------------- HOUSE KEEPING --------------------- #

            # delete datetime variable so we are only working with each sessions unique timedate data
            try: # sometimes, some loops don't have these data, like say loop #1
                del datetime_binary, datetime_cleanF, datetime_reg, datetime_suite2p, datetime_img, datetime_traces
                del creation_date_binary, creation_date_cleanF, creation_date_img
                del creation_date_reg, creation_date_suite2p, creation_date_traces, creation_time_binary, creation_time_cleanF
                del creation_time_img, creation_time_reg, creation_time_suite2p, creation_time_traces
            except:
                pass

        # --------------------------------------------------------- #
        # -------------------- SAVING RESULTS --------------------- #

        # only save the failed attempts
        if len(failed_subi) > 0:
            # Save list to .csv file
            #with open(os.path.join(i['Folder'],'recurseConvertComplete.csv'), 'w', newline='') as f:
                #writer = csv.writer(f)
                #writer.writerow(success_subi)
            with open(os.path.join(i['Folder'],'recurseConvertFailed.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(failed_subi)                        
    
    # to exit or not to exit?
    if run_opts[run_method]=='iterate':
        next = 1