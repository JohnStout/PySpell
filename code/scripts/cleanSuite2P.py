
# Load required modules
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy import stats
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter
from suite2p.extraction import dcnv
import scipy.io as sio
from datetime import datetime

path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)
from s2pfuns import read_s2p
import rootfun as rf

# path to your sessions
# predict_sessions = [r"/path/to/your/session", r"", r""....]
predict_sessions = []

# here's a way to put in one folder and search for suite2p folders
'''
Datafolder = r"/path/to/a/rootfolder"

# get all subdirs
subdirs = rf.list_all_subdirs(phile_name = Datafolder)
predict_sessions = [i for i in subdirs if 'plane0' in i] # filter out for suite2p
predict_sessions = [os.path.split(os.path.split(i)[0])[0] for i in predict_sessions] # get root
'''

# function to find calcium event peaks
def calcium_events(c, zscore_threshold = 1):
    '''
    calcium event detection using the C trace output from constrained foopsi OASIS
    
    Args:
        >>> c: a single c traces
        >>> zscore_threshold: default = 1 std which appears to be pretty good

    Returns:
        >>> idx_peak: peak indices
        >>> c_raw_peaks: raw C valued peaks that also exceed 1std
        >>> cZ_peaks: zscored trace valued peaks
    '''
    from scipy.stats import zscore
    from scipy.signal import find_peaks

    # use the c trace to identify peak times
    cZ              = zscore(c)
    idx_peaks, prop = find_peaks(cZ)          # find C trace events peaks
    cZ_peaks        = cZ[idx_peaks]           # zscored C trace events that are also peaks
    c_raw_peaks     = c[idx_peaks]            # raw C trace events that are also peaks
    c_filt_idx      = np.where(cZ_peaks > 1)  # find C trace events that are also peaks but also greater than 1std
    idx_peaks       = idx_peaks[c_filt_idx]   # indices of C peaks
    cZ_peaks        = cZ_peaks[c_filt_idx]    # zscored C peaks
    c_raw_peaks     = c_raw_peaks[c_filt_idx] # C peaks

    return idx_peaks, c_raw_peaks, cZ_peaks

# function to rescue cells
def rescue_and_reject(iscell, stat, skF):
    '''
    This code rescues candidate false rejection cells based on high skew and low compactness,
    then rejects cells with low asymmetry

    John Stout
    '''

    # if there are <100 pixels in a cell, toss it
    npix = np.array([i['npix'] for i in stat])
    iscell[npix<100]=False

    # if there are <100 pixels in a cell, toss it
    skew = np.array([i['skew'] for i in stat])
    iscell[skew<1.2]=False    

    return iscell

# function to reject overlapping roi and accept the best signal
def reject_overlapping_roi(stat, F, Fneu, C, iscell, fs = 7.5):

    '''
    Detect cells within 20 pixels, identify if a cell shares more than 50% of its activities with another cell,
    place the cell with the lower 'skew' estimate to 'not cells'.

    John Stout 1/23/2025

    '''

    # get ROI
    from scipy.stats import pearsonr
    from scipy.stats import zscore
    from scipy.signal import find_peaks

    print("Detecting overlapping ROI to toss")

    # --  detrend the signal -- #
    Fcor = F-Fneu; 
    
    # calculate skew using the non-detrended signal, like suite2p
    skew = stats.skew(F-Fneu, axis=1)

    # -- Identify cells with overlap and determine if one should be rejected -- #
    stat = stat
    cells = len(stat)
    for i in range(cells):

        # loop over cells, but ignore the current cell
        for ii in range(i + 1, cells):

            # get distance between centriods
            med_dist = (((stat[ii]['med'][0]-stat[i]['med'][0]) ** 2) + ((stat[ii]['med'][1]-stat[i]['med'][1]) ** 2)) ** 0.5
                
            # if the ROI are less than 15 pixels apart between two classified cells, check for event overlap
            if med_dist < 15 and iscell[i]==True and iscell[ii]==True: #and iscell[i] == True and iscell[ii] == True:

                # detect event peaks
                idx_peaks_i, c_raw_peaks_i, cZ_peaks_i    = calcium_events(c = C[i])
                idx_peaks_ii, c_raw_peaks_ii, cZ_peaks_ii = calcium_events(c = C[ii])
                
                # search over all events in cell #ii
                peakii_overlap = []
                for peakii in idx_peaks_ii:

                    # per each event in cell #ii, search for events in cell #i
                    for peaki in idx_peaks_i:

                        # estimate the time difference in peak events
                        event_offset = (peakii - peaki)/fs
                        
                        # if signal co-occured within a 2s window, then these are consdiered the same
                        if np.abs(event_offset) < 2:

                            # tag event
                            peakii_overlap.append(1)

                            # now move on to the next peak to avoid two-peak detections
                            break

                # identify the percentage of events in celii that co-occured with celli
                percent_overlap_ii_to_i = (len(peakii_overlap) / len(idx_peaks_ii)) * 100

                # do a search through events in peaki (peaks from cell i), and find overlapping peaks in cell #ii
                peaki_overlap = []
                for peaki in idx_peaks_i:

                    # per each event in cell #ii, search for events in cell #i
                    for peakii in idx_peaks_ii:

                        # estimate the time difference in peak events
                        event_offset = (peaki - peakii)/fs
                        
                        # if signal co-occured within a 2s window, then these are consdiered the same
                        if np.abs(event_offset) < 2:

                            # tag event
                            peaki_overlap.append(1)

                            # now move on to the next peak to avoid two-peak detections
                            break

                # identify the percentage of events in celii that co-occured with celli
                percent_overlap_i_to_ii = (len(peaki_overlap) / len(idx_peaks_i)) * 100
                
                if percent_overlap_i_to_ii > 25 or percent_overlap_ii_to_i > 25:
                    print(f'Detected overlapping cell candidate between cell {ii} and cell {i}')
                    
                    # determine which cell to send backwards using skew
                    if skew[ii] < skew[i]:
                        print(f'Sending cell{ii} to the "Not Cell" category')
                        iscell[ii] = False
                    elif skew[i] < skew[ii]:
                        print(f'Sending cell{i} to the "Not Cell" category')
                        iscell[i] = False
    return iscell

# function that calls in the iscell variable and rewrites it according to predict_cell predictions
def rewrite_data(predict_sessions):
    import scipy.io as sio
    from datetime import datetime

    for i in predict_sessions:

        # read og iscell
        F, Fneu, spks, stat, ops, iscell0, __ = read_s2p(fpath=i)
        C = np.load(os.path.join(i, 'suite2p', 'plane0', 'C.npy'), allow_pickle=True)
        S = np.load(os.path.join(i, 'suite2p', 'plane0', 'S.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(i,'suite2p','plane0','iscell.npy'), allow_pickle=True)
        iscell_og = iscell.copy()

        # rewrite
        #iscell_og = iscell.copy()
        #del iscell

        # rewrite iscell
        #iscell = np.zeros(iscell_og.shape)
        #iscell[predictions, 0] = 1.0
        #iscell[:, 1] = probabilities[:, 1]

        # --- Some important filtering and rescuing steps -- #
        iscell_in = iscell[:,0].astype(bool)

        # compactness
        compact = np.array([i['compact'] for i in stat])

        # skew
        skF = stats.skew(F-Fneu, axis=1)

        # identify "not cells" that have low compactness and high skew, then send them to "is cell" category
        iscell_out = rescue_and_reject(iscell=iscell_in, stat=stat, skF=skF)

        # find cells near one another amongst the true positives, then if strongly correlated, remove the weaker signal
        iscell_out = reject_overlapping_roi(stat=stat, F=F, Fneu=Fneu, C=C, iscell=iscell_out, fs=7.5)

        # regenerate iscell. Note that we are replacing the iscell with iscell_out because iscell_out is processed probabilities from the classifier
        iscell[:, 0] = 0.0
        iscell[iscell_out, 0] = 1.0

        # save
        np.save(os.path.join(i, 'suite2p', 'plane0', 'iscell.npy'), iscell, allow_pickle=True)
        np.save(os.path.join(i, 'suite2p', 'plane0', 'iscell_og.npy'), iscell_og, allow_pickle=True)

        # save to .mat
        ops_matlab = ops.copy()
        if ops_matlab.get("date_proc"):
            try:
                ops_matlab["date_proc"] = str(
                    datetime.strftime(ops_matlab["date_proc"], "%Y-%m-%d %H:%M:%S.%f"))
            except:
                pass        
        sio.savemat(os.path.join(i, 'suite2p', 'plane0','Fall_classified.mat'), mdict = {'F': F, 'Fneu': Fneu, 'iscell': iscell, 'stat': stat, 'C': C, 'S': S, 'ops': ops_matlab, 's2pSpk': spks})

# -- clean up data -- #
rewrite_data(predict_sessions=predict_sessions)
