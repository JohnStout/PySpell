# This code performs multi session registration
import numpy as np
import os
import suite2p
import sys
import itertools
from copy import deepcopy

# import caiman code that I hacked
import os
import code.caiman_funs.roi_funs as roi_funs

# local stuff
import rootfun as rf # we can import this if our cwd is local
import s2pfuns

from scipy import io as sio

# this code will save .mat files of shape NxMxK, where N is the # of neurons, M is the # of pixels on y-axis, and k is the number of pixels on x-axis
def suite2pToCellReg(fnames, mask_overlap: bool = True, save_name: str = 's2pCellReg.mat'):
    
    '''
    Code that converts suite2p results to a format compatible with cellReg

    Args:
        >>> fnames: a list of directories containing the root of your suite2p folder.
                fnames[0] = r"path/to/your/directory" in this directory will be "suite2p/plane0"
                Also accepts a string type, but the result is converted to list.

        >>> mask_overlap: True, whether to eliminate cells with overlapping ROI


    s2pToCellReg.mat was converted to python using coPilot, then John Stout edited the code to make it functional
    and fitting with the existing modules/functions

    Created on 10/30/2024
    '''

    # if the input type is a string
    if type(fnames) == str:
        print("String type detected, converting to list")
        from copy import deepcopy
        fnames_old = deepcopy(fnames); del fnames; fnames = []
        fnames.append(fnames_old)

    # loop over data
    for fi in fnames:

        # check that the file name is oriented appropriately for later steps
        fiabs = os.path.abspath(fi)
        if 'suite2p\\plane0' in fiabs:
            fi = os.path.split(os.path.split(fiabs)[0])[0]

        # load suite2p data
        F, __, __, stat, ops, __, __ = s2pfuns.read_s2p(fpath = fi)

        # report to user
        if mask_overlap is True:
            print("Removing cells with overlap")

        # Process data
        n_cells = len(stat)
        footprint = np.zeros((n_cells, ops['Lx'], ops['Ly']))

        for it_cell in range(n_cells):

            # Find subindices of the current neuron
            footprint_cell = np.zeros((footprint.shape[1], footprint.shape[2]))
            idx = np.ravel_multi_index((stat[it_cell]['ypix'], stat[it_cell]['xpix']), 
                                    dims=(footprint.shape[1], footprint.shape[2]))

            # Remove pixels that overlap with surrounding cells
            if mask_overlap:
                idx = idx[~stat[it_cell]['overlap'][0]]
                lam = stat[it_cell]['lam'][~stat[it_cell]['overlap']]
            else:
                lam = stat[it_cell]['lam']

            # Update the footprint matrix
            footprint_cell.flat[idx] = lam
            footprint[it_cell, :, :] = footprint_cell

            #
            #i = 100
            #fig, ax = plt.subplots(nrows=1, ncols=2)
            #ax[0].imshow(im[i])
            #ax[1].imshow(footprint[i])

        # save
        save_dir = os.path.join(fi,'suite2p','plane0')
        print("Saving cellReg compatible footprint array to",os.path.join(save_dir,save_name))
        sio.savemat(os.path.join(save_dir, save_name), {'footprints': footprint})

    return footprint

# multiple session registration
def multisess_register_suite2p(fnames: list, template = 'max_proj', single_plane: bool = True, max_distance: int = 50, n_reg = None, onlyAccepted=False):

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
    # TODO: build NWB mechanism - super simple. Just add the reader functions

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

        if onlyAccepted == True:
            print("ONLY INCLUDING ACCEPTED VARIABLES - this is not a flexible approach")
            s2pdata[sess]['stat']   = s2pdata[sess]['stat'][s2pdata[sess]['iscell']]
            s2pdata[sess]['F']      = s2pdata[sess]['F'][s2pdata[sess]['iscell']]
            s2pdata[sess]['spks']   = s2pdata[sess]['spks'][s2pdata[sess]['iscell']]
            s2pdata[sess]['Fneu']   = s2pdata[sess]['Fneu'][s2pdata[sess]['iscell']]
            s2pdata[sess]['im']     = s2pdata[sess]['im'][s2pdata[sess]['iscell']]

            # rewrite this
            print("Rewriting the `iscell` variable to only include accepted components")
            s2pdata[sess]['iscell'] = s2pdata[sess]['iscell'][s2pdata[sess]['iscell']]

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
    spatial_union, assignments, matchings = roi_funs.register_multisession(A=A, dims=dims, coms=coms, templates=templates, max_dist=max_distance)

    # save results
    multisess['inputs'] = dict(); multisess['inputs']['coms']=coms
    multisess['inputs']['spatial'] = spatial; multisess['inputs']['dims'] = dims; multisess['inputs']['templates']=templates
    multisess['spatial_union'] = spatial_union; multisess['assignments'] = assignments; multisess['matchings'] = matchings

    # now it works
    for i in range(len(spatial)):
        # get session name
        sess_name = os.path.split(fnames[i])[-1]
        _, fig = roi_funs.plot_contours(spatial[i],templates[i].T,com=coms[i],swap_dim=True,display_numbers=False);
        ax=fig.gca()
        ax.set_title("Session: "+sess_name)

    # Use number of non-NaNs in each row to filter out components that were not registered in enough sessions
    assignments_filtered = np.array(np.nan_to_num(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]), dtype=int);
    assignments_filtered_nan = assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]
    
    # cheap but works. We have a problem when requiring the cell to exist in less num of sessions than total provided
    if n_reg != len(fnames):
        multisess['common_components'] = assignments_filtered_nan
    else:
        multisess['common_components'] = assignments_filtered
    #multisess['component_index']   = assignments_filtered

    #TODO: Make a better plotting function that accounts for shared ROI
    for i in range(len(spatial)):
        # get session name
        sess_name = os.path.split(fnames[i])[-1]

        # now find only common center of masses
        coms_common = coms[i][assignments_filtered[:,i]]
        _, fig  = roi_funs.plot_contours(spatial[i][:,assignments_filtered[:,i]],templates[i].T,com=coms_common,swap_dim=True,display_numbers=False);
        ax=fig.gca()
        ax.set_title("Session: "+sess_name+" Common ROI")

    # identify which cells need to be tosssed. These might be 'nans' or just not a cell
    counter = 0; cell_rem = []
    for sess in s2pdata.keys():

        # extract `iscell`
        iscell = s2pdata[sess]['iscell'] 

        # identify nans and exclude
        keepdata = np.isfinite(assignments_filtered_nan[:,counter]) # check for nan
        #assignments_temp = assignments_filtered[keepdata,counter]

        # cellpass
        cellpass = iscell[assignments_filtered[:,counter]] # if the cell is a cell
        cell_rem.append(np.where((cellpass == False) | (keepdata == False))[0])
        #cell_rem.append(np.where((cellpass == False) or (keepdata == False))[0])

        counter += 1

    # index of cells to remove
    cell_rem = np.unique(np.array(list(itertools.chain.from_iterable(cell_rem))))

    counter = 0
    for sess in s2pdata.keys(): 

        # extract `iscell`
        iscell = s2pdata[sess]['iscell']               

        # find the index of cells that are accepted
        cell_idx = np.where(iscell==1)[0]; com_cell_idx = []

        # loop over the 'common_components' output using `counter` to tell you which session you are on
        #keep_cell = np.full((iscell.shape),False,dtype=bool) # make a boolean array like `iscell`
        for comi in multisess['common_components'][:,counter]:

            # now loop over all of the cell sin the accepted index
            for celli in cell_idx:

                # if the common cell is also an accepted cell, save the common cell ID
                if ~np.isnan(comi):
                    if int(comi)==celli: # if the common cell is an accepted component
                        com_cell_idx.append(int(comi))

        # finish the job
        com_cell_idx      = np.array(com_cell_idx) # convert to numpy
        iscell_consistent = np.full((iscell.shape),False,dtype=bool) # make a boolean array like `iscell`
        iscell_consistent[com_cell_idx]=True # set all common cells in the `iscell` equivalent array to TRUE
        
        # save your data and progress
        s2pdata[sess]['iscell_reg'] = iscell_consistent

        counter+=1 # move forward your counter

    # index of cell registration information
    # multisess['all_cellmatch'] = deepcopy(multisess['common_components'])
    #if cell_rem.size > 0:
        #multisess['all_cellmatch'] = np.delete(multisess['idx_cellmatch'],cell_rem,axis=0)

    return multisess, backgrounds, s2pdata

# multiplane registration
fpath = r"E:\L6 Experiments\L612\FOV1\SEDS_day11_LBC2_p70_FOV1\SEDS_day11_LBC2_p70_FOV1_img"
def multiplane_registeration(fpath: str, template = 'max_proj', single_plane: bool = True, max_distance: int = 50, n_reg = None, onlyAccepted=False):

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
    # TODO: build NWB mechanism - super simple. Just add the reader functions

    # get suite2p plane
    fpath = s2pfuns.parse_fpath(fpath = fpath)
    fpath = os.path.split(fpath)[0]

    # ignore combined
    planes = [i for i in os.listdir(fpath) if 'combine' not in i]

    # automatically assume that the user wants the components to persist across multiple sessions
    if n_reg is None:
        n_reg = len(planes)
        print("Assuming that the user wants the ROI to be present across the number of input paths = ",n_reg)

    # search for 'suite2p' folder in datafiles
    temp_ops = []; s2p_path = []
    s2pdata = dict(); backgrounds = dict(); counter = 0
    ops = []; stats_file = []; iscell=[]; f_cells = []; f_neuropils = []; spks = []
    for i in planes:

        # path to sess
        temp_path = os.path.join(fpath, i)

        # add a dict to the s2p variable
        s2pdata[i] = dict()

        # load in data
        temp_ops = np.load(os.path.join(temp_path,'ops.npy'), allow_pickle=True).item()
        temp_ops.keys() == temp_ops.keys()  

        # save data
        s2pdata[i]['ops']    = temp_ops
        s2pdata[i]['stat']   = np.load(os.path.join(temp_path,'stat.npy'),allow_pickle=True)
        s2pdata[i]['iscell'] = np.load(os.path.join(temp_path,'iscell.npy'),allow_pickle=True)[:, 0].astype(bool)
        s2pdata[i]['F']      = np.load(os.path.join(temp_path,'F.npy'),allow_pickle=True)
        s2pdata[i]['Fneu']   = np.load(os.path.join(temp_path,'Fneu.npy'),allow_pickle=True)
        s2pdata[i]['spks']   = np.load(os.path.join(temp_path,'spks.npy'),allow_pickle=True)

        # get image masks
        temp = suite2p.ROI.stats_dicts_to_3d_array(s2pdata[i]['stat'], Ly=s2pdata[i]['ops']['Ly'], Lx=s2pdata[i]['ops']['Lx'], label_id=True)
        #temp[temp > 0] = 1.0
        s2pdata[i]['im'] = temp

        if onlyAccepted == True:
            print("ONLY INCLUDING ACCEPTED VARIABLES - this is not a flexible approach")
            s2pdata[i]['stat']   = s2pdata[i]['stat'][s2pdata[i]['iscell']]
            s2pdata[i]['F']      = s2pdata[i]['F'][s2pdata[i]['iscell']]
            s2pdata[i]['spks']   = s2pdata[i]['spks'][s2pdata[i]['iscell']]
            s2pdata[i]['Fneu']   = s2pdata[i]['Fneu'][s2pdata[i]['iscell']]
            s2pdata[i]['im']     = s2pdata[i]['im'][s2pdata[i]['iscell']]

            # rewrite this
            print("Rewriting the `iscell` variable to only include accepted components")
            s2pdata[i]['iscell'] = s2pdata[i]['iscell'][s2pdata[i]['iscell']]

        # get background data
        #for opi in s2pdata[i]['ops']:
        # BACKGROUNDS
        # (meanImg, Vcorr and max_proj are REQUIRED)
        bg_strs = ["meanImg", "Vcorr", "max_proj", "meanImg_chan2", "meanImgE"]
        #i = os.path.split(fnames[counter])[-1]
        backgrounds[i] = dict()
        for bstr in bg_strs:
            if bstr in s2pdata[i]['ops']:
                if bstr == "Vcorr" or bstr == "max_proj":
                    img = np.zeros((s2pdata[i]['ops']["Ly"], s2pdata[i]['ops']["Lx"]), np.float32)
                    img[
                        s2pdata[i]['ops']["yrange"][0]:s2pdata[i]['ops']["yrange"][-1],
                        s2pdata[i]['ops']["xrange"][0]:s2pdata[i]['ops']["xrange"][-1],
                    ] = s2pdata[i]['ops'][bstr]
                else:
                    img = s2pdata[i]['ops'][bstr]
            backgrounds[i][bstr]=img

        # add 1 per loop to iterate over session names
        counter += 1

    # spatial components
    multiplane = dict(); multiplane['templates'] = []; spatial = []; coms = []
    for keys in s2pdata.keys():

        # initialize some variables
        multiplane[keys] = dict()

        # loop over every roi
        roi_data = []
        for roi in range(len(s2pdata[keys]['im'])):

            # here, we want a samples x components matrix and so we just flatten.
            # you recreate this array by running: np.reshape(data,(dims),'C')
            roi_data.append(s2pdata[keys]['im'][roi].flatten())

        spatial.append(np.array(roi_data).T)

        # save data
        multiplane[keys]['spatial']  = spatial
        multiplane[keys]['meanImgE'] = backgrounds[keys]['meanImgE']
        multiplane[keys]['dims']     = multiplane[keys]['meanImgE'].shape

        # templates
        multiplane['templates'].append(multiplane[keys]['meanImgE'])

        # get center of mass for each
        med = np.array([i['med'] for i in s2pdata[keys]['stat']])
        multiplane[keys]['coms'] = med
        
        # also store here
        coms.append(med)
        
    # multiplane registration across all planes first
    A         = spatial
    dims      = multiplane[keys]['dims'] # dimensions for reshape
    templates = multiplane['templates']  # templates used for registration
    max_distance = 15 # multiplane shifts should even be less
    spatial_union, assignments, matchings = roi_funs.register_multisession(A=A, dims=dims, coms=coms, templates=templates, max_dist=max_distance)

    # save results
    multiplane['inputs'] = dict(); multiplane['inputs']['coms']=coms
    multiplane['inputs']['spatial'] = spatial; multiplane['inputs']['dims'] = dims; multiplane['inputs']['templates']=templates
    multiplane['spatial_union'] = spatial_union; multiplane['assignments'] = assignments; multiplane['matchings'] = matchings

    # now we can use multiplane['assignments'] to identify components that are the same


    # now it works
    #for i in range(len(spatial)):
    #    # get session name
    #    sess_name = planes[i]
    #    _, fig = caimanfuns.plot_contours(spatial[i],templates[i].T,com=coms[i],swap_dim=True,display_numbers=False);
    #    ax=fig.gca()
    #    ax.set_title("Session: "+sess_name)

    # Use number of non-NaNs in each row to filter out components that were not registered in enough sessions
    assignments_filtered = np.array(np.nan_to_num(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]), dtype=int);
    assignments_filtered_nan = assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]
    
    # cheap but works. We have a problem when requiring the cell to exist in less num of sessions than total provided
    if n_reg != len(planes):
        multiplane['common_components'] = assignments_filtered_nan
    else:
        multiplane['common_components'] = assignments_filtered
    #multisess['component_index']   = assignments_filtered

    #TODO: Make a better plotting function that accounts for shared ROI
    for i in range(len(spatial)):
        # get session name
        sess_name = planes[i]

        # now find only common center of masses
        coms_common = coms[i][assignments_filtered[:,i]]
        _, fig  = roi_funs.plot_contours(spatial[i][:,assignments_filtered[:,i]],templates[i].T,com=coms_common,swap_dim=True,display_numbers=True);
        ax=fig.gca()
        ax.set_title("Session: "+sess_name+" Common ROI")

    # identify which cells need to be tosssed. These might be 'nans' or just not a cell
    counter = 0; cell_rem = []
    for sess in s2pdata.keys():

        # extract `iscell`
        iscell = s2pdata[sess]['iscell'] 

        # identify nans and exclude
        keepdata = np.isfinite(assignments_filtered_nan[:,counter]) # check for nan
        #assignments_temp = assignments_filtered[keepdata,counter]

        # cellpass
        cellpass = iscell[assignments_filtered[:,counter]] # if the cell is a cell
        cell_rem.append(np.where((cellpass == False) | (keepdata == False))[0])
        #cell_rem.append(np.where((cellpass == False) or (keepdata == False))[0])

        counter += 1

    # index of cells to remove
    cell_rem = np.unique(np.array(list(itertools.chain.from_iterable(cell_rem))))

    counter = 0
    for sess in s2pdata.keys(): 

        # extract `iscell`
        iscell = s2pdata[sess]['iscell']               

        # find the index of cells that are accepted
        cell_idx = np.where(iscell==1)[0]; com_cell_idx = []

        # loop over the 'common_components' output using `counter` to tell you which session you are on
        #keep_cell = np.full((iscell.shape),False,dtype=bool) # make a boolean array like `iscell`
        for comi in multiplane['common_components'][:,counter]:

            # now loop over all of the cell sin the accepted index
            for celli in cell_idx:

                # if the common cell is also an accepted cell, save the common cell ID
                if ~np.isnan(comi):
                    if int(comi)==celli: # if the common cell is an accepted component
                        com_cell_idx.append(int(comi))

        # finish the job
        com_cell_idx      = np.array(com_cell_idx) # convert to numpy
        iscell_consistent = np.full((iscell.shape),False,dtype=bool) # make a boolean array like `iscell`
        iscell_consistent[com_cell_idx]=True # set all common cells in the `iscell` equivalent array to TRUE
        
        # save your data and progress
        s2pdata[sess]['iscell_reg'] = iscell_consistent

        counter+=1 # move forward your counter

    # index of cell registration information
    # multisess['all_cellmatch'] = deepcopy(multisess['common_components'])
    #if cell_rem.size > 0:
        #multisess['all_cellmatch'] = np.delete(multisess['idx_cellmatch'],cell_rem,axis=0)

    return multiplane, backgrounds, s2pdata