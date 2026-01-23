# code meant to load suite2p data, then save as a Fall.mat file
import os
import numpy as np
from scipy.io import savemat

import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from s2pfuns import read_s2p
from rootfun import list_all_subdirs

# change me to your mouse_path
mouse_path = r'C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L632_F_LeftPFC_L6Chr_L5CT-FLEXgcamp6f'  # Replace with your Suite2p directory path

def save_fall_mat(suite2p_dir):
    """
    Save Suite2p data in a format compatible with Fall.mat.

    Parameters:
    i (str): Directory path where the data is stored.
    F (np.ndarray): Fluorescence data.
    Fneu (np.ndarray): Neuropil fluorescence data.
    iscell (np.ndarray): Cell identification array.
    stat (list): List of statistics for each cell.
    C (np.ndarray): Component matrix.
    S (np.ndarray): Spatial components.
    ops (dict): Options dictionary containing parameters like 'fs', 'nplanes', etc.
    """
    # Load the Suite2p data
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(suite2p_dir)
    C = np.load(os.path.join(suite2p_dir, 'C.npy'), allow_pickle=True)
    S = np.load(os.path.join(suite2p_dir, 'S.npy'), allow_pickle=True)
    
    # Convert ops to a format suitable for MATLAB
    ops_matlab = {k: v for k, v in ops.items() if not isinstance(v, np.ndarray)}
    
    # Save the data to a .mat file
    spks = np.zeros((F.shape[0], 0), dtype=np.float32)  # Placeholder for spikes
    savemat(os.path.join(suite2p_dir, 'Fall.mat'), mdict={'F': F, 'Fneu': Fneu, 'iscell': iscell,
                                                 'stat': stat, 'C': C, 'S': S,
                                                 'ops': ops_matlab, 's2pSpk': spks})
    
if __name__ == "__main__":
    # Example usage
    s2p_folders = [i for i in list_all_subdirs(mouse_path) if 'suite2p' in i and 'plane0' in i]

    for s2pi in s2p_folders:
        print(f"Processing {s2pi}...")
        if 'Fall.mat' not in os.listdir(s2pi):
            print(f"Saving Fall.mat for {s2pi}...")
            save_fall_mat(s2pi)

