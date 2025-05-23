# This code performs multi session registration
import numpy as np
import os
import suite2p
import sys
import itertools
from copy import deepcopy

#from datetime import datetime
#from pathlib import Path
#sys.path.append(str(Path(__file__).resolve().parent.parent))

# import caiman code that I hacked
import os
#from caiman_funs.roi_funs import roi_funs

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