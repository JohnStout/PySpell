# SCRIPT_saveToDropbox
"""
This code will be designed to:
    1) Transfer suite2p data to dropbox (excluding data.bin)
    2) Synchronize updates between dropbox and the original file storage

    Currently, this code copies all suite2p variables, but if we hack the GUI, we could reduce this to just the .mat file.
    Hacking the gui to include .mat variables is exceedingly difficult and became a sunk cost.

# TODO: packageToNWB comes after we have a full python workflow    

John Stout 
    CoPilot helped write some of this code

Updates:
    11/21/2024: fixed the issue where files were constantly re-synchronizing even if their files were identical.
                Added suite2p conversion
                Should add a way to copy behavioral files

    11/28/2024: Update includes dropbox synchronizing with behavioral folders. HOWEVER, you should have one _beh folder per session!

"""

# TODO: NOW that I have the behavior converter working in recurseConvert, UPDATE THIS CODE
# TODO: Add busybee
# TODO: Make sure the time offset method is detecting actual time since files were updated, verses time passed

import os; import matplotlib.pyplot as plt; import tifffile as tf
path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)
from pathlib import Path
import numpy as np
import time
import csv
from datetime import datetime, timedelta
import shutil
from suite2p.io.save import save_mat

# custom modules
import rootfun as rf # we can import this if our cwd is local
root = rf.dropbox_root(dropbox_folder='timspellman')
from s2pfuns import read_s2p, fast_suite2p
from s2pfuns import postProcess
from sessreg import suite2pToCellReg

# recursive method
syncpaths = dict()

syncpaths = [

    # John Folders     
    {'Folder':  r"E:\L6 Experiments\L612",  
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L612_F_RightPFC_L6Chr_PFCgcamp8f_L6PAN")   
     },
    {'Folder':  r"E:\L6 Experiments\L613",  
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L613_F_LeftPFC_L6Chr_PFCgcamp8f_L6PAN")   
     },
    {'Folder':  r"E:\L6 Experiments\L614",  
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L614_F_LeftPFC_L6Chr_PFCgcamp6f_L6PAN")   
     }, 
    {'Folder':  r"E:\L6 Experiments\L615",  
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L615_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN")   
     },      
    {'Folder':  r"E:\L6 Experiments\L616",  
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L616_M_RightPFC_L6Chr_PFCgcamp6f_L6PAN")   
     }, 
    {'Folder':  r"E:\L6 Experiments\L609-pan", 
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L609_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN")   
     },     

    {'Folder':  r"E:\ThalamicRec\MDT1",  
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\Thalamus\Subjects\MDT1_M_LeftMD_MDgcamp_pilot")   
     }, 

    {'Folder':  r"E:\L6 Experiments\L607T4",  
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L607T4_M_RightPFC_L6Chr_PFCgcamp6f_L6CC")   
     },
    {'Folder':  r"E:\L6 Experiments\L608",
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L608_F_LeftPFC_L6Chr_PFCgcamp6f_L6PAN")  
     },

    {'Folder':  r"E:\L6 Experiments\L1",   
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L1_F_LeftPFC_L6Chr_PFCgcamp6f_L6PAN")   
     },
    {'Folder':  r"E:\L6 Experiments\L6R11",
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L6R11_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN")   
     },
    {'Folder':  r"E:\L6 Experiments\L605",   
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L605_M_RightPFC_L6Chr_PFCgcamp6f_L6L5")     
     },
    {'Folder':  r"E:\L6 Experiments\T30", 
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\T30_M_LeftPFC_L6Chr_PFCgcamp6f_L6PAN")        
     },
    {'Folder':  r"F:\John\L6 Experiments\recordings_panneuronal\T-30",
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\T30_M_LeftPFC_L6Chr_PFCgcamp6f_L6PAN")        
     },
    {'Folder':  r"F:\John\L6 Experiments\recordings_L5CT\L6-05",
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L605_M_RightPFC_L6Chr_PFCgcamp6f_L6L5")     
     },
    {'Folder':  r"F:\John\L6 Experiments\recordings_IT\L607-T4",
     'Dropbox': os.path.join(root,"OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L607T4_M_RightPFC_L6Chr_PFCgcamp6f_L6CC")   
     },

         
        ]

# copy files from origin
files_to_copy = ['stat.npy', 'spks.npy', 'F.npy', 
                 'Fneu.npy', 'ops.npy', 'iscell.npy', 
                 'S.npy',    'C.npy',   'Fall.mat', 
                 's2pCellReg.mat']

# function to synchronize specific files in two folders
def synchronize_folders(origin: str, destination: str, files_to_copy: list):
    '''
    synchronize_folders copies data between the origin and destination, according to the file_list in files_to_copy

    Args:
        >>> origin: path to copy from
        >>> destination: path to receive copies
        >>> files_to_copy: list of files from origin to copy to destination

    CoPilot wrote loop, John made into function
    '''

    # Copy the selected files from source to destination
    for filename in files_to_copy:
        full_file_name = os.path.join(origin, filename)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination)
            print(f"Copied: {full_file_name} to {destination}")
        else:
            print(f"File not found: {full_file_name}")      

# function to return date and time
def get_date_time(file_path):
    # Get the creation time
    creation_time = os.path.getctime(file_path)
    creation_date = datetime.fromtimestamp(creation_time)

    # Get the last modified time
    modified_time = os.path.getmtime(file_path)
    modified_date = datetime.fromtimestamp(modified_time)

    # Format the creation and modified dates and times in the desired format
    creation_date_str = creation_date.strftime("%d-%m-%Y")
    creation_time_str = creation_date.strftime("%H:%M")
    modified_date_str = modified_date.strftime("%d-%m-%Y")
    modified_time_str = modified_date.strftime("%H:%M")

    print(f"File creation date: {creation_date_str}")
    print(f"File creation time: {creation_time_str}")
    print(f"File last modified date: {modified_date_str}")
    print(f"File last modified time: {modified_time_str}")

    return modified_date_str, modified_time_str

# a specific function to detect a missing FOV field and rename a subfolder
def match_FOV_toSubfolder(fpath: str):
    '''
    This function identifies whether the folder containing your suite2p data does not have "FOV",
    but the root folder outside of it does. If so, it appends the FOV name.

    Args:
        >>> fpath: path/to/your/suite2p/plane0

    Returns:
        >>> new_subfolder_name
    '''

    # back out of the suite2p/plane0 folder
    fpath = os.path.split(os.path.split(fpath)[0])[0]
    folder_name = os.path.split(os.path.split(fpath)[0])[-1]
    subfolder_name = os.path.split(fpath)[-1]

    # search for "FOV"
    if 'FOV' in folder_name and 'FOV' not in subfolder_name:
        idx = folder_name.find('FOV')
        full_index = range(idx, idx + len('FOV')+1) # copilot
        substring = ''.join([folder_name[i] for i in full_index]) # copilot

        # append to the subfolder
        new_subfolder_name = subfolder_name + '_' + substring
    else:
        new_subfolder_name = False

    return new_subfolder_name

# loop over paths nd synchronize suite2p folder
for i in syncpaths:
    print("Origin Folder:",i['Folder'])

    # get all subdirs
    subdirs = rf.list_all_subdirs(phile_name = i['Folder'])

    # filter out folders without suite2p
    datafolders = [i for i in subdirs if 'suite2p\\plane0' in i]
    behfolders = [i for i in subdirs if '_beh' in i]

    # this should be in recurseConvert because suite2p might struggle to play
    # TODO: add suite2p code that updates the ops or stat variable to detect a changed path
    FOV_naming_convention = True
    if FOV_naming_convention == True:
        # do an initial loop over datafolders and change the folder names as needed
        for fpath in datafolders:

            # identify if "FOV" is in the subfolder name and if it isn't, but it is in the root, rename the folder
            new_subfolder_name = match_FOV_toSubfolder(fpath = fpath)

            if new_subfolder_name != False:
                # create a new folder name
                fold_path = os.path.split(os.path.split(fpath)[0])[0]
                fnew_path = os.path.join(os.path.split(fold_path)[0], new_subfolder_name)
                print("Renamed", fold_path)
                print("To", fnew_path)
                os.rename(fold_path, fnew_path) 

        # filter out folders without suite2p
        print("Updating datafolders based on new folder names")
        datafolders = [k for k in subdirs if 'suite2p\\plane0' in k]

    # loop over datafolders
    for fpath in datafolders:

        # generate a file name
        fname = fpath.split('\\suite2p')[0] # remove suite2p extensions
        rname = os.path.split(fname)[0]
        fname = os.path.split(fname)[-1]    # use the last file path folder

        # search for behavioral folder
        behfolder = [k for k in os.listdir(rname) if '_beh' in k]

        # create a folder in the dropbox folder
        dropbox_folder = os.path.join(i['Dropbox'],fname) # generate a folder name
        folder_exist = os.path.exists(dropbox_folder)     # check if it exists
        if folder_exist == False:
            print("Generating folder:",dropbox_folder)
            os.mkdir(path = dropbox_folder)

        # copy behavior files
        for bpi in behfolder:

            # within the behavioral folder, identify files
            bpath = os.path.join(rname,bpi)
            bfiles = os.listdir(bpath)
            print("Behavior files detected in",bpath)

            # check for the file on dropbox
            behfound = [k for k in os.listdir(dropbox_folder) if k=='beh.mat']
            if len(behfound) > 0:

                # search for update
                mod_date_dropbox, mod_time_dropbox = get_date_time(file_path = os.path.join(dropbox_folder, 'beh.mat'))

                # get origin date
                mod_date_origin, mod_time_origin = get_date_time(file_path = os.path.join(bpath, 'beh.mat'))
            
                # if the modified times are 2 min offset, then copy the most recent
                if mod_date_dropbox != mod_date_origin or mod_time_dropbox != mod_time_origin:
                    # Define the format of the time strings 
                    time_format = "%d-%m-%Y %H:%M" 

                    # Convert time strings to datetime objects 
                    mod_time_dropboxdt = datetime.strptime(mod_date_dropbox + ' ' + mod_time_dropbox, time_format) 
                    mod_time_origindt  = datetime.strptime(mod_date_origin  + ' ' + mod_time_origin, time_format)
                    offset = mod_time_dropboxdt - mod_time_origindt
                    #days = offset.days 
                    ##hours, remainder = divmod(offset.seconds, 3600) 
                    #minutes, seconds = divmod(remainder, 60)

                    # search for updates and make sure that those updates didn't occur outside of a 2 minute bounbdary to avoid a never ending cycle
                    # TODO: Check the timedelta req
                    if mod_time_dropboxdt > mod_time_origindt and offset > timedelta(minutes=2):
                        
                        # if the dropbox file was updated more recently, copy the files to origin
                        print("Detected update to",dropbox_folder)
                        print("Copying to", bpath)
                        synchronize_folders(origin = dropbox_folder, destination = bpath, files_to_copy = ['beh.mat'])

                    elif mod_time_dropboxdt < mod_time_origindt and offset > timedelta(minutes=2):

                        # if the dropbox file was updated more recently, copy the files to origin
                        print("Detected update to",bpath)
                        print("Copying to", dropbox_folder)                    
                        synchronize_folders(origin = bpath, destination = dropbox_folder, files_to_copy = ['beh.mat'])
            
            else:

                # copy the file based on recent updates
                synchronize_folders(origin = bpath, destination = dropbox_folder, files_to_copy = ['beh.mat'])

            '''
            # reassign the name of the file based on the _ delimeter
            #if len(fname.split('_')) > len(fname.split('-')):
            #    delimeter = '_'
            #else:
            #    delimeter = '-'
            #fsplit = fname.split(delimeter)
            #bname_new = '_'.join(fsplit[0:4]) + '_beh.mat'

            # rename the beh.mat file in the dropbox folder - this is named according to origin to confirm that the beh file is accurately placed
            #os.rename(os.path.join(dropbox_folder,'beh.mat'), os.path.join(dropbox_folder,bname_new))

            # identify whether this file name already exists and if so, create a new name
            #next = 0; counter = 0
            #while next == 0:
            #    counter+=1
            #    bname_found = [k for k in bfiles if bname_new in k]
            #    if len(bname_found) > 0:
            #        bname_new = bname_new.split('.mat')[0]+str(counter)+'.mat'
            #    else:
            #        next = 1
            '''

        # check if there exists a suite2p folder
        dropbox_s2p_folder = os.path.join(dropbox_folder,'suite2p','plane0')
        folder_exist = os.path.exists(dropbox_s2p_folder)     # check if it exists
        if folder_exist == False:
            print("Generating folder:",dropbox_s2p_folder)
            os.makedirs(name = dropbox_s2p_folder, exist_ok = False)

        # make sure you search for all of the files in the files_to_copy, they should exist at origin
        search_list = files_to_copy
        target_list = os.listdir(fpath) # target list are the files within the origin folder (your drive, not dropbox)

        # detect if primary suite2p variables are missing and if so, run suite2p
        if 'F.npy' not in target_list or 'stat.npy' not in target_list or 'ops.npy' not in target_list or 'Fneu.npy' not in target_list or 'spks.npy' not in target_list or 'iscell.npy' not in target_list:
            # run suite2p  
            print("Suite2p files not detected. Running suite2p...")    
            root_folder = os.path.split(os.path.split(fpath)[0])[0]         
            out = fast_suite2p(imgpath             = os.path.join(root_folder,'img.tif'), # path to the img.tif file
                                savepath         = '',                                  # default saveout to the appropriate folder
                                gcamp            = '6f',                                # use of gcamp6f reduces tau to 0.7
                                alt_ops          = None,                                # This does NOT use spellman ops
                                wipe_and_replace = True)                                # This DOES wipe and erase the suite2p folder
            if out == False:
                print("Skipping session:",fpath)
                continue

        # detect missing .mat file in suite2p folder and generate
        if 'Fall.mat' not in target_list:
            print("Fall.mat file missing, saving...")
            F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath = fpath)
            ops['save_path'] = fpath
            save_mat(ops, stat, F, Fneu, spks, iscell, redcell = False)

            # I should probably rerun recurse convert to regen these vars because I ran this code without the line below, meaning that there is a very small possibility there exist sessions with wrong data
            del F, Fneu, spks, stat, ops, iscell, blF

        # detect missing cleaned traces to repair
        if 'C.npy' and 'S.npy' not in target_list:
            print("Postprocessing session:", fpath)                        
            code_start = time.process_time()  
            postProcess(s2ppath=fpath).cleanup_raw_traces(run_parallel=False)  
            process_end = time.process_time()
            print(f"Total time to postprocess: {(process_end - code_start)/60:.2f} minutes")
        
        # if the cellreg data is not present, run
        if 's2pCellReg.mat' not in target_list:
            print("Preparing cellReg data for session:", fpath)                        
            code_start = time.process_time() 
            suite2pToCellReg(fnames = fpath, mask_overlap = True) 
            process_end = time.process_time() 
            print(f"Total time to generate cellReg formatted data: {(process_end - code_start)/60:.2f} minutes")

        # check if there exist files within the folder
        filesn = os.listdir(dropbox_s2p_folder)
        file_check = all(ei in filesn for ei in files_to_copy)
        if file_check == False:

            # synch
            synchronize_folders(origin = fpath, destination = dropbox_s2p_folder, files_to_copy = files_to_copy)

        # update the target list and identify missing elements to continue sync      
        target_list = os.listdir(fpath)

        # Convert lists to set
        search_set = set(search_list)
        target_set = set(target_list)

        # Find the missing elements
        missing_elements = search_set - target_set

        # Remove missing elements from search_list
        filtered_search_list = [item for item in search_list if item not in missing_elements]

        # get updates
        for fi in filtered_search_list:

            # get dropbox date
            mod_date_dropbox, mod_time_dropbox = get_date_time(file_path = os.path.join(dropbox_s2p_folder, fi))

            # get origin date
            mod_date_origin, mod_time_origin = get_date_time(file_path = os.path.join(fpath, fi))

            # if the modified times are 2 min offset, then copy the most recent
            if mod_date_dropbox != mod_date_origin or mod_time_dropbox != mod_time_origin:
                # Define the format of the time strings 
                time_format = "%d-%m-%Y %H:%M" 

                # Convert time strings to datetime objects 
                mod_time_dropboxdt = datetime.strptime(mod_date_dropbox + ' ' + mod_time_dropbox, time_format) 
                mod_time_origindt  = datetime.strptime(mod_date_origin  + ' ' + mod_time_origin, time_format)
                offset = mod_time_dropboxdt - mod_time_origindt
                #days = offset.days 
                ##hours, remainder = divmod(offset.seconds, 3600) 
                #minutes, seconds = divmod(remainder, 60)

                # search for updates and make sure that those updates didn't occur outside of a 2 minute bounbdary to avoid a never ending cycle
                # TODO: Check the timedelta req
                if mod_time_dropboxdt > mod_time_origindt and offset > timedelta(minutes=2):
                    
                    # if the dropbox file was updated more recently, copy the files to origin
                    print("Detected update to",dropbox_s2p_folder)
                    print("Copying to", fpath)
                    synchronize_folders(origin = dropbox_s2p_folder, destination = fpath, files_to_copy = files_to_copy)

                elif mod_time_dropboxdt < mod_time_origindt and offset > timedelta(minutes=2):

                    # if the dropbox file was updated more recently, copy the files to origin
                    print("Detected update to",fpath)
                    print("Copying to", dropbox_s2p_folder)                    
                    synchronize_folders(origin = fpath, destination = dropbox_s2p_folder, files_to_copy = files_to_copy)

        # TODO: ADD a method that searches for dropbox folders that are not present on the origin folder.
        # IF such folders are detected, tag them for deletion but NEVER actually delete. CHange the name of the dropbox folder to DELETEME___

# TODO: synchronize behavior
# essentially, search for _beh.mat files within a syncpath, i, and copy/paste them into the dropbox folder



