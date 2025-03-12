import numpy as np
import os
import suite2p
import sys

# import caiman code
import os; path_added = os.getcwd(); os.chdir(path_added); print("Added path:",path_added)
import code.caiman_funs.roi_funs as roi_funs

print("Please cite 'CaImAn' for multisession registration")

def multisess_register_suite2p(fnames: list, template = 'max_proj', single_plane: bool = True, max_distance: int = 50, n_reg = None):
    """
    Args:
        >>> fnames: list of sessions with suite2p data. YOu can have as many sessions as you'd like :)
        >>> template: the template to use to compare ROI across sessions or planes. 
                    Options:
                        'max_proj': the maximum projection; Default and recommended
                        'meanImg': the mean image across time
                        'Vcorr': spatial correlation map

        >>> max_distance: maximum distance between ROI. This can vary highly depending on your pixel res.
        >>> n_reg: the number of required sessions for an ROI to be present. The default is the number of input sessions
 
    Returns:
        >>> multisess: dictionary containing the following information from CaImAn

            '''
                spatial_union: csc_matrix # pixels x # of total distinct components
                    union of all kept ROIs 

                assignments: ndarray int of size # of total distinct components x # sessions
                    element [i,j] = k if component k from session j is mapped to component
                    i in the A_union matrix. If there is no much the value is NaN

                matchings: list of lists
                    matchings[i][j] = k means that component j from session i is represented
                    by component k in A_union
            '''                       

    """
    # TODO: build NWB mechanism
    assert single_plane == True, "This code supports comparison between 1 planed data between datasets. You cannot use two 3D datasets"

    # automatically assume that the user wants the components to persist across multiple sessions
    if n_reg is None:
        n_reg = len(fnames)
        print("Assuming that the user wants the ROI to be present across the number of input paths = ",n_reg)

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
    multisess = dict(); multisess['templates'] = []; spatial = []; coms = []
    for keys in s2pdata.keys():

        # initialize some variables
        multisess[keys] = dict()

        # loop over every roi
        roi_data = []
        for roi in range(len(s2pdata[keys]['im'])):

            # here, we want a samples x components matrix and so we just flatten.
            # you recreate this array by running: np.reshape(data,(dims),'C')
            roi_data.append(s2pdata[keys]['im'][roi].flatten())

        spatial.append(np.array(roi_data).T)

        # save data
        multisess[keys]['spatial']  = spatial
        multisess[keys]['max_proj'] = backgrounds[keys]['max_proj']
        multisess[keys]['dims']     = multisess[keys]['max_proj'].shape

        # templates
        multisess['templates'].append(multisess[keys]['max_proj'])

        # get center of mass for each
        med = np.array([i['med'] for i in s2pdata[keys]['stat']])
        multisess[keys]['coms'] = med
        
        # also store here
        coms.append(med)
        
    # get list of session names
    sessions = list(s2pdata.keys())

    # multisession registration
    A         = spatial
    dims      = multisess[keys]['dims'] # dimensions for reshape
    templates = multisess['templates']  # templates used for registration
    spatial_union, assignments, matchings = roi_funs.register_multisession(A=A, dims=dims, coms=coms, templates=templates, max_dist=50)

    # save results
    multisess['spatial_union'] = spatial_union; multisess['assignments'] = assignments; multisess['matchings'] = matchings

    # now it works
    for i in range(len(spatial)):
        # get session name
        sess_name = os.path.split(fnames[i])[-1]
        _, fig = roi_funs.plot_contours(spatial[i],templates[i].T,com=coms[i],swap_dim=True,display_numbers=False);
        ax=fig.gca()
        ax.set_title("Session: "+sess_name)

    #coors, fig  = caimanfuns.plot_contours(spatial[0],templates[0].T,com=coms[0],swap_dim=True,display_numbers=False);
    #coors, fig2 = caimanfuns.plot_contours(spatial[1],templates[1].T,com=coms[1],swap_dim=True,display_numbers=False);

    # Filter components by number of sessions the component could be found
    #n_reg = 2  # minimal number of sessions that each component has to be registered in; default should be all sessions

    # Use number of non-NaNs in each row to filter out components that were not registered in enough sessions
    assignments_filtered = np.array(np.nan_to_num(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]), dtype=int);
    multisess['common_components'] = assignments_filtered

    for i in range(len(spatial)):
        # get session name
        sess_name = os.path.split(fnames[i])[-1]

        # now find only common center of masses
        coms_common = coms[i][assignments_filtered[:,i]]
        _, fig  = roi_funs.plot_contours(spatial[i][:,assignments_filtered[:,i]],templates[i].T,com=coms_common,swap_dim=True,display_numbers=False);
        ax=fig.gca()
        ax.set_title("Session: "+sess_name+" Common ROI")

    # Use filtered indices to select the corresponding spatial components - THIS NEEDS ADJUSTING
    # The code below represents components observed on session 1 also observed on other sessions (planes)
    #spatial_filtered = spatial[0][:, assignments_filtered[:, 0]]
    #multisess['spatial'] = assignments_filtered

    # 159 repeatedly observed cells
    #unique_cells = np.abs(assignments_filtered.shape[0]-spatial[0].shape[1])
    #print("Unique cells:",unique_cells, ",",str((unique_cells/spatial[0].shape[1])*100),"% unique to plane0")
