# Script that converts NWB for all files in the reanalysis folder
import os
import sys 
import numpy as np
import suite2p
import scipy.io as sio
import matplotlib.pyplot as plt

# this is custom and hacked code below
code_path = os.path.join(os.getcwd()[0:os.getcwd().find('code')-1],'code')
os.chdir(code_path)
print("Path added:",code_path)
import rootfun as rf # we can import this if our cwd is local
rf.addcode_paths(os.getcwd())
import stringfun as hf
import nwbfun as nwbfun
import matpy

# define the imaging and behavior folders
root = rf.dropbox_root(dropbox_folder='timspellman')
Datafolders = os.path.join(root,r"timspellman\Imaging\PFCMDTReanalysis\Imaging")
Datafolders_beh = os.path.join(root,r"timspellman\Behavior2P_Archive")
savepath = os.path.join(root,r"OtherData\John\ProjectData\project_CT_layer6")

# animals with layer-specific PFC->MD projection neuron recordings
layer_5_animals=['t284','t286','t291','t296','t314','t381','t383','t400','t403']
layer_6_animals=['t450','t451','t454','t455']
all_animals = layer_5_animals+layer_6_animals

# loop over each animals data, extract suite2p sources/images and store in a dict
for animali in all_animals: # loop over each animal

    # join the rootfolder with each animal
    datafolder = os.path.join(Datafolders,animali) 

    # filter out SEDS sessions that are joined
    dir_content   = os.listdir(datafolder) # gets subfolders in directory
    seds_sessions = [i for i in dir_content if 'SEDS' in i] # find any sessions that are SEDS
    #idx_rem       = [i for i in range(len(seds_sessions)) if 'SED' in ''.join(hf.find_duplicate_characters(seds_sessions[i]))]
    #seds_merged   = [seds_sessions.pop(i) for i in idx_rem] # this both deletes SEDS+SEDS sessions + stores which session was deleted as 'seds_merged' which will NOT be used here
    #seds_merged   = [i for i in seds_sessions if 'x' not in i] # removes those xx sessions
    suite2p_path  = [os.path.join(datafolder,i,'suite2p') for i in seds_sessions if 'x' not in i] # define suite2p paths for each recorded session

    # behavior
    datafolder_beh = os.path.join(Datafolders_beh,animali)

    for pathi in suite2p_path:

        # get session name
        session_name = os.path.split(os.path.split(pathi)[0])[-1]
        behfolder    = os.path.join(datafolder_beh,session_name+'Beh.mat')

        # skip duplicated sessions
        if 'SED' in ''.join(hf.find_duplicate_characters(session_name)):
            print("Skipping",session_name)
            continue

        # only write files if they both exist
        beh_report = [i for i in os.listdir(datafolder_beh) if session_name+'Beh.mat' in i]
        for i in range(len(beh_report)):
            if 'SED' in ''.join(hf.find_duplicate_characters(beh_report[i])):
                beh_report[i]='NA'
        beh_report = [True for i in beh_report if session_name+'Beh.mat' in i]

        if beh_report is False:
            print("Behavioral file not detected. Skipping",session_name+'Beh.mat')
            continue

        # layer indicator
        if animali in layer_5_animals:
            layer_ind = 'L5'

        elif animali in layer_6_animals:
            layer_ind = 'L6'

        # gen a savename
        savename = animali+'_'+session_name+'_'+layer_ind+'.nwb'
        if savename in os.listdir(savepath):
            print(savename,"found in path. Renaming.")
            savename = savename.split('.nwb')[0]+'_new'+'.nwb'

        if nwbfun.checkConsistency(suite2p_path=pathi, matpath=behfolder) is True:

            # write file to nwb - first works. Just need to align to behavior timeseries
            nwbpath = nwbfun.suite2p_nwb().save_nwb(datafolder = pathi, nwbsavename = savename, savefolder = datafolder)

            print("This may take a few minutes as the code works in a ton of data to NWB")
            nwbfun.behavior_nwb(matpath = behfolder).readwrite(nwbpath)
            print("File written to:", nwbpath)

            # checkfile
            nwbfun.validate_nwb(nwbpath=nwbpath)

        else:
            print("Fluorescence and behavioral data not aligned for",session_name,"... Skipping.")
