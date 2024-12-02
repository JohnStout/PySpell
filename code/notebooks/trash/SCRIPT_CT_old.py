import os
import sys 
import numpy as np
import suite2p
import scipy.io as sio
import matplotlib.pyplot as plt

local_root = os.path.split(os.path.split(os.getcwd())[0])[0] # local folder (John)
local_packages = os.listdir(local_root) # packages in John folder

# get lab root
lab_root = os.path.split(os.path.split(local_root)[0])[0]
lab_folders = os.listdir(lab_root) # all folders in timspellman dropbox

# get matlab and python folder
matlab_folder = os.path.join(lab_root,'MATLAB')
python_folder = os.path.join(lab_root,'Python')

# add the local packages to path
[sys.path.append(os.path.join(local_root,i,'code')) for i in local_packages]
print("Added the following packages to path:",local_packages)

# import raw_to_tif
from utils import helper_funs
import dataWrangler

# root folder
Datafolders = r"C:\Users\uggriffin\SpellmanLab Dropbox\timspellman\Imaging\PFCMDTReanalysis\Imaging"
Datafolders_beh = r"C:\Users\johnj\SpellmanLab Dropbox\timspellman\Behavior2P_Archive"

# animals with layer-specific PFC->MD projection neuron recordings
layer_5_animals=['t284','t286','t291','t296','t314','t381','t383','t400','t403']
layer_6_animals=['t450','t451','t454','t455']
all_animals = layer_5_animals+layer_6_animals

# loop over each animals data, extract suite2p sources/images and store in a dict
neural_data = dict()
for animali in all_animals: # loop over each animal

    # join the rootfolder with each animal
    datafolder = os.path.join(Datafolders,animali) 

    # filter out SEDS sessions that are joined
    dir_content   = os.listdir(datafolder) # gets subfolders in directory
    seds_sessions = [i for i in dir_content if 'SEDS' in i] # find any sessions that are SEDS
    idx_rem       = [i for i in range(len(seds_sessions)) if 'SED' in helper_funs.find_duplicate_characters(seds_sessions[i])]
    seds_merged   = [seds_sessions.pop(i) for i in idx_rem] # this both deletes SEDS+SEDS sessions + stores which session was deleted as 'seds_merged' which will NOT be used here
    suite2p_path  = [os.path.join(datafolder,i,'suite2p') for i in seds_sessions if 'x' not in i] # define suite2p paths for each recorded session

    # loop over suite2p_paths and extract information, including summary images
    session = dict()
    for s2pi in suite2p_path:

        # get session name
        session_name = os.path.split(os.path.split(s2pi)[0])[-1]

        if 'x' in session_name:
            print("Session:", session_name," skipped")
            continue

        # make sure only 1 plane is listed (max projection was performed on multiplane data)
        plane_check = [i for i in os.listdir(s2pi) if 'plane' in i]
        assert len(plane_check) == 1, "multiple dimension data discovered" # sanity check for multiplane data
        plane_name = plane_check[0] # name of plane

        # load suite2p variables
        ops         = np.load(os.path.join(s2pi,plane_name,'ops.npy'), allow_pickle=True).item(); ops.keys() == ops.keys() # options
        stats_file  = np.load(os.path.join(s2pi,plane_name,'stat.npy'),allow_pickle=True) # statistics
        iscell      = np.load(os.path.join(s2pi,plane_name,'iscell.npy'),allow_pickle=True)[:, 0].astype(bool) # index for cell or not
        f_cells     = np.load(os.path.join(s2pi,plane_name,'F.npy'),allow_pickle=True) # f of cells
        f_neuropils = np.load(os.path.join(s2pi,plane_name,'Fneu.npy'),allow_pickle=True) # f of neuropil
        spks        = np.load(os.path.join(s2pi,plane_name,'spks.npy'),allow_pickle=True) # spk via deconvolution
        img_sum     = suite2p.ROI.stats_dicts_to_3d_array(stats_file, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True) # summary image

        # layer indicator
        if animali in layer_5_animals:
            layer_ind = 'L5'
        elif animali in layer_6_animals:
            layer_ind = 'L6'

        # save data - redundant, exhausts more memory, but cleaner this way
        session[str(session_name)] = {'ops':         ops,
                                      'stats':       stats_file,
                                      'iscell':      iscell,
                                      'f_cells':     f_cells,
                                      'f_neuropils': f_neuropils,
                                      'spks':        spks,
                                      'img_summary': img_sum,
                                      'Layer':       layer_ind}

    neural_data[str(animali)] = session