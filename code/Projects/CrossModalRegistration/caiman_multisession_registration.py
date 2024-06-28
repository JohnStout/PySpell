import numpy as np
import os
import suite2p
import sys

# import caiman code
import os; path_added = os.getcwd(); os.chdir(path_added); print("Added path:",path_added)
import caimanfuns

print("Please cite 'CaImAn' for multisession registration")

fnames = [r"C:\Users\johnj\SpellmanLab Dropbox\timspellman\Imaging\PFCMDTReanalysis\Imaging\t284\SEDS2",
          r"C:\Users\johnj\SpellmanLab Dropbox\timspellman\Imaging\PFCMDTReanalysis\Imaging\t284\SEDS3"]

def multisess_register_suite2p(fnames: list, template = 'max_proj', single_plane: bool = True):
    """
    Args:
        >>> fnames: list of sessions with suite2p data. YOu can have as many sessions as you'd like :)
        >>> template: the template to use to compare ROI across sessions or planes. 
                    Options:
                        'max_proj': the maximum projection; Default and recommended
                        'meanImg': the mean image across time
                        'Vcorr': spatial correlation map
    
    """
    # TODO: build NWB mechanism

    assert single_plane == True, "This code supports comparison between 1 planed data between datasets. You cannot use two 3D datasets"

    # search for 'suite2p' folder in datafiles
    temp_ops = []; s2p_path = []
    for fname in fnames:
        # search for suite2p path
        checkpoint = [i for i in os.listdir(fname) if 'suite2p' in i][0]
        assert 'suite2p' in checkpoint, "suite2p data not detected. Please run suite2p or name the file 'suite2p'" # sanity check

        # path
        s2p_path.append(os.path.join(fname,'suite2p','plane0'))

    s2pdata = dict(); backgrounds = dict(); counter = 0
    ops = []; stats_file = []; iscell=[]; f_cells = []; f_neuropils = []; spks = []
    for i in s2p_path:
        # session
        sess = os.path.split(fnames[counter])[-1]

        # add a dict to the s2p variable
        s2pdata[sess] = dict()

        # load in data
        temp_ops = np.load(os.path.join(i,'ops.npy'), allow_pickle=True).item()
        temp_ops.keys() == temp_ops.keys()  

        # save data
        s2pdata[sess]['ops']    = temp_ops
        s2pdata[sess]['stat']   = np.load(os.path.join(i,'stat.npy'),allow_pickle=True)
        s2pdata[sess]['iscell'] = np.load(os.path.join(i,'iscell.npy'),allow_pickle=True)[:, 0].astype(bool)
        s2pdata[sess]['F']      = np.load(os.path.join(i,'F.npy'),allow_pickle=True)
        s2pdata[sess]['Fneu']   = np.load(os.path.join(i,'Fneu.npy'),allow_pickle=True)
        s2pdata[sess]['spks']   = np.load(os.path.join(i,'spks.npy'),allow_pickle=True)

        # get image masks
        temp = suite2p.ROI.stats_dicts_to_3d_array(s2pdata[sess]['stat'], Ly=s2pdata[sess]['ops']['Ly'], Lx=s2pdata[sess]['ops']['Lx'], label_id=True)
        #temp[temp > 0] = 1.0
        s2pdata[sess]['im'] = temp

        # get background data
        #for opi in s2pdata[sess]['ops']:
        # BACKGROUNDS
        # (meanImg, Vcorr and max_proj are REQUIRED)
        bg_strs = ["meanImg", "Vcorr", "max_proj", "meanImg_chan2"]
        #sess = os.path.split(fnames[counter])[-1]
        backgrounds[sess] = dict()
        for bstr in bg_strs:
            if bstr in s2pdata[sess]['ops']:
                if bstr == "Vcorr" or bstr == "max_proj":
                    img = np.zeros((s2pdata[sess]['ops']["Ly"], s2pdata[sess]['ops']["Lx"]), np.float32)
                    img[
                        s2pdata[sess]['ops']["yrange"][0]:s2pdata[sess]['ops']["yrange"][-1],
                        s2pdata[sess]['ops']["xrange"][0]:s2pdata[sess]['ops']["xrange"][-1],
                    ] = s2pdata[sess]['ops'][bstr]
                else:
                    img = s2pdata[sess]['ops'][bstr]
            backgrounds[sess][bstr]=img

        # add 1 per loop to iterate over session names
        counter += 1

    # spatial components
    multisess = dict(); multisess['templates'] = []
    for keys in s2pdata.keys():

        # initialize some variables
        multisess[keys] = dict(); spatial = []

        # loop over every roi
        for roi in range(len(s2pdata[keys]['im'])):

            # here, we want a samples x components matrix and so we just flatten.
            # you recreate this array by running: np.reshape(data,(dims),'C')
            spatial.append(s2pdata[keys]['im'][roi].flatten())

        # save data
        multisess[keys]['spatial']  = spatial
        multisess[keys]['max_proj'] = backgrounds[keys]['max_proj']
        multisess[keys]['dims']     = multisess[keys]['max_proj'].shape

        # templates
        multisess['templates'].append(multisess[keys]['max_proj'])
        
    # get list of session names
    sessions = list(s2pdata.keys())

    # multisession registration
    A         = [np.array(multisess[sessions[0]]['spatial']).T, np.array(multisess[sessions[1]]['spatial']).T] # list of csc_matrices of shape (pixels x components)
    dims      = multisess[keys]['dims'] # these have to be the same across sessions
    templates = multisess['templates']
    spatial_union, assignments, matchings = caimanfuns.register_multisession(A=A, dims=dims, templates=templates)

    # save results
    multisess['spatial_union'] = spatial_union; multisess['assignments'] = assignments; multisess['matchings'] = matchings

    # Filter components by number of sessions the component could be found
    n_reg = 2  # minimal number of sessions that each component has to be registered in; default should be all sessions

    # Use number of non-NaNs in each row to filter out components that were not registered in enough sessions
    assignments_filtered = np.array(np.nan_to_num(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]), dtype=int);

    # Use filtered indices to select the corresponding spatial components - THIS NEEDS ADJUSTING
    # The code below represents components observed on session 1 also observed on other sessions (planes)
    spatial_filtered = spatial[0][:, assignments_filtered[:, 0]]

    # 159 repeatedly observed cells
    unique_cells = np.abs(assignments_filtered.shape[0]-spatial[0].shape[1])
    print("Unique cells:",unique_cells, ",",str((unique_cells/spatial[0].shape[1])*100),"% unique to plane0")

    visualization.plot_contours(spatial[0],templates[0].T,swap_dim=True);