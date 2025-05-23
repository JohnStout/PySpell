#!/usr/bin/env python

'''
A collection of functions that handle suite2p data and an object "postProcess" designed
to postprocess suite2p results.

To troubleshoot, click the 'run' button in the VScode IDE and make sure your settings are 
designed to run in the interactive window

If you care about postprocessing your data, try:
    self = postProcess(s2ppath=r"path/to/your/suite2p/rootfolder")

    And from here, you can run line by line what each method in the postProcess function does.

    If you wanted to call a select method, you could do self.cleanup_raw_traces() or something.


Updates:
    1/10/2025: JS added methods to postProcess that handle denoising and detrending based off Tims EMD denoising
                and Andres's savgolay/mad detrending. Please note that denoising SHOULD NOT be used for constrained foopsi as
                the model is designed to handle noisy data and will lead to overfitting.

    5/7/2025: Updates yesterday and today include addition of new functions for examining cells
              Addition of 'rescue_cells' which performs really well on detecting some false negatives
              Addition of calcium_events_2 which is a far superior version of calcium_events

Planned addon:
    1/10/25: Using the output C trace, identifying peaks, characterizing halfwidth and making sure the right side is longer than left of signal                

'''

#TODO: Go through code and snake_case the functions, camelCase objects

# load modules
from suite2p.extraction import dcnv
import numpy as np
from scipy import stats
import os
import time
import xmltodict
import suite2p
import tifffile as tf
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
from suite2p.io import BinaryFile
import shutil
import scipy.io as sio

# check path
if 'PySpell'.lower() in os.path.split(os.getcwd())[-1].lower():
    new_path = os.path.join(os.getcwd(),'code')
    os.chdir(new_path)

# these are for caiman-based deconvolution
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# and theres more
import pandas as pd
import rootfun

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

from caiman_funs.caiman_eval import compute_event_exceptionality
import scipy.io as sio

from scipy import stats, special
from scipy.stats import norm, median_abs_deviation
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d

import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from math import ceil


# ---------------------------------------- #
# ----  MAJOR STAND ALONE FUNCTIONS  ----- #
# ---------------------------------------- #

# fast_suite2p is a major function
#imgpath = r"E:\L6 Experiments\L612\FOV1\SEDS_day11_LBC2_p70_FOV1\SEDS_day11_LBC2_p70_FOV1_img\img.tif"
#imgpath = r"F:\John\L6 Experiments\recordings_L5CT\L6-05\FOV3\SEDS_day6_FOV3_noOpto_LBC0_REFERENCE\SEDS_day6_noOpto_noWheels_img_FOV3\img.tif"
def fast_suite2p(imgpath: str, savepath: str = '', gcamp: str ='6f', alt_ops = None, wipe_and_replace: bool = False):
    """
    This code runs suite2p for you if you collected with Spellman lab equipment (thor labs)

    Args:
        >>> datapath: directory, including the .tif file extension
        >>> savepath: path to save your data
        >>> gcamp: default is 6f. 6f and 8f are interchangeable. 8s and 8m have different tau constants.
        >>> zoom_factor: currently dysfunctional, leave alone
        >>> alt_ops: None. Allows user to provide their own ops file.
        >>> wipe_and_erase: Whether to erase the existing suite2p folder and replace

    John Stout

    UPDATES:
    >>> 4/22/25: added a method to try and load a memory mapped file, if it fails, it will load the file into memory and resave as bigtiff
    """

    #___________________________________________#

    # load data lazily
    try:
        images = tf.memmap(imgpath, mode="r")
        # Success — you can use 'images'
        is_mappable = True
    except Exception as e:

        # Failed — fallback method
        is_mappable = False
        print(f"Memory mapping failed: {e}")

        # load into memory, then resave
        print("Reading file into memory, then resaving as bigtiff for memory mapping later")
        images = tf.imread(imgpath) # this is to load the data into memory
        tf.imwrite(imgpath, images, dtype=images.dtype, bigtiff=True)

    # movies and associated frame rates
    root_path = os.path.split(imgpath)[0]
    movie_name = os.path.split(imgpath)[1]

    # whether to wipe and replace the suite2p folder to replace it
    if wipe_and_replace is True:
        s2pFound = len([i for i in os.listdir(root_path) if 'suite2p' in i])
        if s2pFound > 0:
            print("Wiping suite2p folder found in", root_path)
            shutil.rmtree(os.path.join(root_path,'suite2p'))

    # get metadata
    root_contents = os.listdir(root_path)
    metadata_file = [i for i in root_contents if '.xml' in i and 'experiment' in i.lower()][0]
    metadata_path = os.path.join(root_path,metadata_file)
    file = xmltodict.parse(open(metadata_path,"r").read()) # .xml file

    # default ops
    if alt_ops is None:

        # alt_ops['pre_smooth'] = 2, while ops['pre_smooth'] = 0
        # alt_ops['spatial_taper'] = 50 while ops['spatial_taper'] = 40
        # alt_ops['max_overlap'] = 1.0 while ops['max_overlap'] = 0.75 **
        # alt_ops['anatomical_only'] = 1, ops['anatomical_only'] = 0
        # alt_ops['diameter'] = 12, ops['diameter'] = 0
        # alt_ops['soma_crop'] = 1.0, ops['soma_crop'] = True **

        # get ops
        ops = suite2p.default_ops()

        # define frame rate based on metadata
        fr = float(file['ThorImageExperiment']['LSM']['@frameRate'])

        # assuming you are using fast z capture with 3 planes (subtracting the flyback
        # also assuming the flyback was removed
        if 'PlaneZ' in movie_name:
            ops['nplanes'] = 3
            print("Frame rate of",fr,'changed to',fr * (3/4))
            fr =  fr * (3/4) # we tossed the flyback frame and so therefore, the result is 3 plane or 3/4 planes with the fourth having been the flyback
        else:
            print("Frame rate of",fr,'changed to',fr/4)
            fr = fr/4

        # get default suite2p inputs - update on 12/13/2024 after noticing discrepancy in Tims and default params
        # spellOps result in ROI that look overly smoothed out while default ops are not strict enough. This is a play to find middle ground.
        ops['fs'] = fr
        ops['max_overlap'] = 0.75 # 1.0 throws out NO ROIs, 0.75 allows up to 75% overlap
        ops['diameter'] = 12 # was 12, let cellpose figure it out
        ops['soma_crop'] = 1.0
        ops['use_builtin_classifier'] = True

        if images.shape[0] > 5000:
            ops['batch_size'] = 5000 # default is 500 but this machine can handle more
        else:
            ops['batch_size'] = int(np.floor(images.shape[0] / 2))

        # threshold_scaling
        ops['threshold_scaling'] = 2.0
        
        # just precautionary
        ops['do_bidiphase'] = True

        # might as well...
        ops['multiplane_parallel'] = False

        # running suite2p on the max projection after extensive visualization
        # I was not able to examine max_proj/meanImg bc of shaping differences.
        ops['anatomical_only'] = 3 # mean image E

        # denoise
        ops['denoise'] = False

        # allowance for overlap
        ops['allow_overlap'] = False # use distance measurement to identify whether a cell should be merged

        # see here: https://suite2p.readthedocs.io/en/latest/settings.html
        # tau: (float, default: 1.0) The timescale of the sensor (in seconds), used for deconvolution kernel. The kernel is fixed to have this decay and is not fit to the data. We recommend:
        # 0.7 for GCaMP6f
        # 1.0 for GCaMP6m
        # 1.25-1.5 for GCaMP6s
        if '6f' in gcamp or '8f' in gcamp: # check the 8f
            ops['tau'] = 0.7 # gcampe6f
        elif '6m' in gcamp:
            ops['tau'] = 1.0
        elif '6s' in gcamp:
            ops['tau'] = 1.3
            
        # save out the NWB file
        ops['save_NWB']=False # set to false for now

    else:

        # if the user provided an ops file
        ops = alt_ops
        
        # if the provided ops file has a save_path, it is probably wrong
        if len(ops['save_path0'])>0:
            print("save_path0 detected in ops file, rewriting to default")
            ops['save_path0']=''
        if len(ops['fast_disk'])>0:
            print("fast_disk detected in op file, rewriting to default")
            ops['fast_disk']=[]

    # run suite2p algorithm
    if len(savepath) > 0:
        ops['save_path0']=savepath 

    # set db, this overrides the ops variable
    db = {
        'data_path': [root_path],
        'tiff_list': [movie_name],
    }
    db

    # save as .mat
    ops['save_mat']=True
    
    #ops['save_folder']='L6A03_SD1odor_suite2p' # this feature doesn't really work
    if images.shape[0] > 200:

        # run suite2p
        output_ops = suite2p.run_s2p(ops=ops, db=db)

        # load in the data and save out summary images
        F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=output_ops['save_path0'])
        tf.imwrite(os.path.join(output_ops['save_path0'],'meanImg.tif'), output_ops['meanImg'], bigtiff=True)
        tf.imwrite(os.path.join(output_ops['save_path0'],'maxProj.tif'), output_ops['max_proj'], bigtiff=True)
    else:
        print("The total size of your imaging data is <",200," samples. Not performing suite2p.")
        output_ops = False

# read the results from suite2p
def read_s2p(fpath: str):
    '''
    This group of functions might better serve as an object

    read_s2p reads the outputs from s2p

    Args:
        >>> fpath: path to suite2p variables to load

    Returns:
        >>> suite2p outputs + baseline corrected F

    John Stout
    '''

    # get the correct directory
    s2p_found = [i for i in os.listdir(fpath) if 'F.npy' in i]
    
    if len(s2p_found) == 0:
        fpath = parse_fpath(fpath=fpath)
    
    # suite2p results
    F      = np.load(os.path.join(fpath,'F.npy'), allow_pickle=True)
    Fneu   = np.load(os.path.join(fpath,'Fneu.npy'), allow_pickle=True)
    spks   = np.load(os.path.join(fpath,'spks.npy'), allow_pickle=True)
    stat   = np.load(os.path.join(fpath,'stat.npy'), allow_pickle=True)
    ops    =  np.load(os.path.join(fpath,'ops.npy'), allow_pickle=True)
    ops    = ops.item()
    iscell = np.load(os.path.join(fpath,'iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
    blF    = baseline_corrected_F(F=F, Fneu=Fneu, ops=ops) # baseline corrected F

    return F, Fneu, spks, stat, ops, iscell, blF

# a function to make sense of the fpath input
def parse_fpath(fpath: str):
    '''
    function to take an fpath input and return the suite2p folder

    Args:
        >>> fpath: path to suite2p variables to load

    '''
    if os.path.split(fpath)[-1] == 'suite2p':
        # eventually make this code a loop that loops over the various planes
        fpath = os.path.join(fpath,'plane0')
    elif os.path.split(fpath)[-1] != 'plane0' and os.path.split(fpath)[-1] != 'suite2p':
        fpath = os.path.join(fpath,'suite2p','plane0') 
    return fpath     

# ---------------------------------------- #
# --- MAJOR OBJECTS FOR POSTPROCESSING --- #
# ---------------------------------------- #
class postProcess():
    '''
    To run constrained foopsi, you need at least 1001 samples. 
    '''

    def __init__(self, s2ppath: str):

        # read suite2p results
        F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath = s2ppath)

        # store path
        self.s2ppath = s2ppath

        # store suite2p data
        self.F = F; self.Fneu = Fneu; self.spks = spks; self.stat = stat
        self.ops = ops; self.iscell = iscell; self.blF = blF

        # load C and S if they exist
        try: 
            self.C = np.load(os.path.join(s2ppath,'C.npy'), allow_pickle=True)
            self.S = np.load(os.path.join(s2ppath,'S.npy'),)
        except:
            print("C and S not found in",s2ppath)

        if F.shape[1] < 1001:
            print("The size of your F trace is too small to run constrained foopsi. Please see documentation for more information.")
            self.C = np.zeros(shape = F.shape)
            self.S = np.zeros(shape = F.shape)

    # -- method to run OASIS -- #
    def cleanup_raw_traces(self, replace_rename: bool = False):
        '''
        John put this code together based on Tim's MATLAB code and provided an parallelized option 
        thanks to copilot

        Last edit 10/19/2024

        Args: 
            >>> replace_rename: preset to False, set to true if you want to wipe and replace
            >>> suite2p_detrend: set to True, alternative approach is to use sgolay filter and subtract from F, but this 
        
        UPDATES
        1/9/25: @TS devised a denoising method using EMD analysis and @JS implemented this with sgolay to both denoise and detrend data 
                    @JS included code that calculates the std of the noise distribution using the sgolay/mad data
                        then uses this as input to constrained foopsi to prevent constrained foopsi from estimating noise on its own using
                        welch's method because it auto assumes an FS=1
        
        1/10/25: @JS reorganized/cleaned code such that detrended data are processed in a separate method
        '''

        import deconvolution as dc

        # set empty arrays
        C = []; S = []; f_det_all = []; f_emd_all = []
        total_cells = self.F.shape[0]

        # run constrained foopsi
        if self.F.shape[1] > 1001:
            print("runParallel set to False. Iterating through suite2p ROIs...")

            # loop over each cell and perform the operations
            process_start = time.process_time()
            for x in range(self.F.shape[0]):
                print(f'denoising / deconvolving cell {x}')

                # f_detrended = sgolay/mad method (see function and Grosmark 2021). 
                # sn=std of the 'baseline/caevent free/noise' distribution used for constrained_foopsi
                f_detrended, __ = self.sgolay_detrend(f = self.F[x,:] - self.Fneu[x,:])

                # TODO: For some reason, this didn't clean up my signal more
                #q = np.quantile(f_detrended, 0.1)
                #f_detrended = f_detrended - q

                # option to instead use emd analysis to extract high frequency noise for std estimation
                #__, sn = self.emd_denoise(f = f_detrended) # this makes more sense than uses sgolays sn

                # running constrained foopsi
                try:
                    process_start = time.process_time()
                    noise_range = [0.25, 0.5] # noise frequency range
                    deconv_method = 'oasis'   # OASIS
                    solvers = None            # for cvx, but doesn't matter here
                    lags = 5                  # lags==5 appear the most robust which is consistent with their default 
                    sn = None                 # let the code figure out the noise distribution
                    c, bl, c1, g, sn, sp, lam = dc.constrained_foopsi(f_detrended, p = 2, method_deconvolution = deconv_method, bas_nonneg = True,
                                                                noise_range = noise_range, noise_method = 'logmexp', sn=sn,
                                                                lags = lags, fudge_factor = 1, solvers=solvers, verbosity=True)

                    # normalize your spike train
                    # This code takes the median over the median rescaled residual then normalizes this against the spike train
                    # subtracting out the median from the residual is confusing. I dont recall why I put it there. I probably didnt like how the data looked without it.
                    mad = np.median(np.abs( (f_detrended - c) - np.median(f_detrended - c) ))
                    sp = sp / mad
                    
                    print("Time to cleanup raw traces:",(time.process_time() - process_start)/60,"min")    
                except:
                    print("Failed to run constrained foopsi, likely division by zero")
                    c  = np.zeros(shape = self.F[x,:].shape)
                    sp = np.zeros(shape = self.F[x,:].shape)
                C.append(c)
                S.append(sp)
                
                # report on progress
                progress = (x + 1) / total_cells * 100
                print(f"{progress:.2f}% Completed")
        
            # convert to numpy
            C = np.array(C) # denoised flourescence
            S = np.array(S) # deconvolved spiking

            # store in self
            self.C = C
            self.S = S                                      

            # report on timing
            print("Time to cleanup raw traces:",(time.process_time() - process_start)/60,"min")    

            # save traces and return
            print("Saving results to",self.s2ppath)
            if replace_rename is True:
                print("Renaming F to F_s2p and spks to spks_s2p and saving C as F and S as spks...")
                os.rename(os.path.join(self.s2ppath,'F.npy'), os.path.join(self.s2ppath,'F_s2p.npy'))
                os.rename(os.path.join(self.s2ppath,'spks.npy'), os.path.join(self.s2ppath,'spks_s2p.npy'))
                np.save(os.path.join(self.s2ppath,'F.npy'), C); print("Saved denoised flourescence (C)")
                np.save(os.path.join(self.s2ppath,'spks.npy'), S); print("Saved deconvolved spikes (S)")
            else:
                np.save(os.path.join(self.s2ppath,'C.npy'), C); print("Saved denoised flourescence (C)")
                np.save(os.path.join(self.s2ppath,'S.npy'), S); print("Saved deconvolved spikes (S)")
            return C, S

    # -- methods to clean-up the F trace
    def sgolay_detrend(self, f):
        '''
        method that detrends signal f, an input argument representing the users fluorescent trace

        Args:
            >>> f: a single cells fluorescent trace
        
        Returns:
            >>> f_detrended: a detrended version of the input 'f' signal
            >>> sn: standard deviation of the event free 'noise' or 'baseline' signal
        
        UPDATES:
            >>> 4/22/25: added flexibility for window size. Default remains 1001

        John Stout
        '''

        # identify candidate outlier events (signal)
        mad_f = np.median(np.abs(f - np.median(f)))
        ttimes = np.where(f > np.median(f) + 3 * mad_f)[0]
        
        # now replace events with nan
        f2 = f.copy()
        f2[ttimes] = np.nan # nan out the events
        
        # interpolate candidate signal events to estimate noise
        f2 = np.interp(np.arange(len(f2)), np.arange(len(f2))[~np.isnan(f2)], f2[~np.isnan(f2)])
        sn = np.std(f2) # std of the noise (baseline) distribution

        # subtract the underlying trend (detrend) from the noise-reduced signal
        if f2.shape[0] < 1001:
            print("The size of your f trace is < 1000 samples, adjusting dynamically...")
            window = int(f2.shape[1]/1)
            if window % 2 == 0:
                window += 1            
            f3 - savgol_filter(f2, window, 2) # was 1001, but this is too large for small traces
        else:
            f3 = savgol_filter(f2, 1001, 2)
        #@AM changed above line to 101 from 1001
        f_detrended = f - f3
        f_detrended = f_detrended.astype(np.float32)

        return f_detrended, sn

    # method to denoise data, do not use before OASIS
    def emd_denoise(self, f):
        '''
        Runs empirical mode decomposition (EMD) to separate non-sinosoidal 'f' signals into separate mode functions (similar to fourier transform)

        NOTE:
            This should not be run before constrained_foopsi OASIS. Constrained foopsi models the fluorescent trace
            with inclusion of noise as variable and was designed to handle noisy signals
            Indeed, running EMD before constrained_foopsi leads to overfitting of the AR model.
        
        Args:
            >>> f: a single fluorescent trace

        Returns:
            >>> reconstructed_f: reconstructed fluorescent trace after dropping high frequency components (first two)
            >>> sn: standard deviation of noise distribution

        '''

        #import concurrent.futures
        from PyEMD import EMD
        from scipy.signal import find_peaks
        from scipy.ndimage import uniform_filter1d
        import warnings

        ##### Perform EMD decomposition #####
        emd = EMD() # instantiate object

        # parameter setting
        emd.FIXE_H = 10
        emd.MAX_ITERATION = 100
        emd.energy_ratio_thr = 20 # was 20
        emd.DTYPE = np.float32
        emd.spline_kind = 'cubic' # cubic or linear
        emd.range_thr = 0.2
        max_imf = 10 # number of components, was 10
        emd.extrema_detection = 'simple'

        # report to user
        print(f"EMD parameters of: max_imf count={max_imf}, FIXE_H={emd.FIXE_H}, MAX_ITERATION={emd.MAX_ITERATION}, energy_ratio_thr={emd.energy_ratio_thr}, spline_kind={emd.spline_kind}, range_thr={emd.range_thr}, extrema_detection={emd.extrema_detection}")

        # run emd
        imfs = emd(f, max_imf=max_imf) # this will return 11 rows, of which the 11th is the residual (checked)
        #assert imfs.shape[0] == max_imf+1, "emd fitting was did not match the requested number of imfs"
        if imfs.shape[0] != max_imf+1:
            warnings.warn("emd fitting did not match the requested number of imfs, this may result in improper removal of noise.")

        # subtract bottom component - JS
        #f_sub = f-imfs[0:1].sum(axis=0)

        # extract residuals and reconstruct f after dropping first and last imfs
        residual = imfs[-1] # get residual
        imfs = imfs[0:-1,:] # rid imfs of residual

        # standard deviation of noise distribution
        sn = np.std(imfs[0:1].sum(axis=0))

        # drop the high frequency components (first two) and sum the remaining
        reconstructed_f = imfs[2:9, :].sum(axis=0) + residual # was 2:8 in matlab, I had 2:9 after observing lost signal

        # how close is the sum of components to og signal
        #reconstructed_f = imfs[0:9].sum(axis=0)+residual

        '''

        #if imfs.shape[1] < 10:
        #    imfs = np.pad(imfs, ((0, 0), (0, 10 - imfs.shape[1])), mode='constant')

        # combine middle components
        enL  = imfs[3:7, :].sum(axis=0) + residual
        enLL = imfs[4:7, :].sum(axis=0) + residual  

        # Flag timepoints unusable due to residual drift or low SNR
        nanDrift = True  # Set this flag based on your condition
        if nanDrift:
            enH = uniform_filter1d(np.abs(imfs[1, :]), size=2000)
            enM = uniform_filter1d(np.abs(imfs[2:4, :]).sum(axis=0), size=2000)
            rto = enM / enH
            nullTPs = np.where(rto < 2)[0]

        # Scale residual signal for bleaching and slight drift
        pI, pA = find_peaks(reconstructed_f, width=10) # peak events
        q  = np.quantile(reconstructed_f, 0.9) # value associated with 90%tile
        fq = np.where(pA['prominences'] > q)[0] # identify peaks > 90%tile
        pA = pA['prominences'][fq] # filter out peaks < 90%tile 
        pI = pI[fq] # filter out peaks < 90%tile, this is the INDEX of peaks relative to reconstructed_f
        pI = np.concatenate(([0], pI, [len(f) - 1])) # why are we adding duplicates to start and end? Padding?
        pA = np.concatenate((pA[:1], pA, pA[-1:]))

        # Polynomial fitting
        p   = np.polyfit(pI, pA, 2) # fit 2nd order polynomial
        pks = np.polyval(p, np.arange(len(f)))
        pksTmp = -pks # invert trend
        pksTmp = pksTmp - np.min(pksTmp) + 1 # rescale
        fTmp = enL - np.min(enL) # rescale imfs dist to be >0
        fTmp = (fTmp * pksTmp) / np.max(fTmp)

        pksTmp2 = pks - np.min(pks)
        pksTmp2 = pksTmp2 / np.max(pksTmp2) + 1
        fTmp2 = reconstructed_f / pksTmp2
        fTmp2 = fTmp2 - np.min(fTmp2)
        f = fTmp + fTmp2
        '''

        return reconstructed_f, sn

    def save_modified_f(self):
        '''
        This function saves out reconstructed and detrended f traces.

        Note that these traces can be used for analysis of F, rather than for constrained foopsi.
        In fact, constrained foopsi is designed to handle noisy traces and will overfit your signal if EMD is run.
        
        TODO: EMD Analaysis is slow. Parallel processing might be required.

        '''
        print("Denoising data via EMD analysis and detrended via savgolay. Please see documentation.")
        Warning("This code may operate slowly as the EMD analysis per cell would benefit from parallelization which has not been implemented")

        # saving out cleaned traces
        f_clean = []
        process_start = time.process_time()
        for x in range(self.F.shape[0]):

            # run EMD first to denoise trace
            f_reconstructed, sn_emd = self.emd_denoise(f = self.F[x,:] - self.Fneu[x,:])

            # f = neuropil corrected f signal
            f_detrended, sn_golay = self.sgolay_detrend(f = f_reconstructed)

            # cache
            f_clean.append(f_detrended)

            # report
            print(f"{round(((x+1)/self.F.shape[0]), ndigits=3)*100}% complete")

        # convert f_clean to numpy
        f_clean = np.array(f_clean)
        print("Time to denoise and detrend:",(time.process_time() - process_start)/60,"min")    

        # save
        fpath = parse_fpath(fpath=self.s2ppath)

        # save out as .mat
        sio.savemat(file_name = os.path.join(fpath, 'F_clean.mat'), mdict={'f': f_clean, 'info': 'this signal was denoised with EMD by dropping the first two high freq components, then detrended with sgolay'})

        # TO FACT CHECK FOR YOURSELF
        '''
        # run these lines

        # instantiate object
        self = postProcess(s2ppath = r"path/to/your/folder/with/suite2p/folder")

        # choose a cell
        x = 0

        # run EMD first to denoise trace
        f_reconstructed, sn_emd = self.emd_denoise(f = self.F[x,:] - self.Fneu[x,:])

        # f = neuropil corrected f signal
        f_detrended, sn_golay = self.sgolay_detrend(f = f_reconstructed)

        plt.close()
        %matplotlib widget
        plt.plot(self.F[x,:] - self.Fneu[x,:], 'k', linewidth=0.5) # plot f-fneu
        plt.plot(f_reconstructed,'b',linewidth=0.5)                # plot emd reconstructed f
        plt.plot(f_detrended, 'r', linewidth=0.5)                  # plot the emd reconstructed, sgolay detrended signal
        plt.legend(['f-fneu','f_emd','f_emd_sgolay'])
                
        '''

    # -- these can probably be deleted -- #
    def estimate_foopsi_lags(self):
        import numpy as np
        from sklearn.metrics import mean_squared_error

        def foopsi(f_detrended, sn, lags):
            # Fit the AR model with the specified lag
            c, bl, c1, g, sn, sp, lam = dc.constrained_foopsi(f_detrended, p = 2, method_deconvolution = 'oasis', bas_nonneg = True,
                                                            noise_range = [0.25, 0.5], noise_method = 'logmexp', lags = lags, 
                                                            fudge_factor = 1, solvers=None, verbosity=True)
            print(f"lags={lags}")
            return c
        
        # a function to estimation BIC
        def calculate_bic(f_detrended, sn, lags):
            '''
            CoPilot put this together with tweaks by JS :)
            '''
            print(f"Running constrained foopsi on lags={lags}")

            # Fit the AR model with the specified lag
            c, bl, c1, g, sn, sp, lam = dc.constrained_foopsi(f_detrended, p = 2, method_deconvolution = 'oasis', bas_nonneg = True,
                                                            noise_range = [0.25, 0.5], noise_method = 'logmexp', lags = lags, sn=sn,
                                                            fudge_factor = 1, solvers=None, verbosity=True)
            
            # Calculate the mean squared error
            mse = mean_squared_error(f_detrended, c + bl)

            # Number of parameters in the model (lags + 1 for the intercept)
            num_params = lags + 1

            # Calculate the BIC
            n = len(f_detrended)
            bic = n * np.log(mse) + num_params * np.log(n)

            return bic

        # hardcoded max lags
        max_lags = 6 # visual inspection reveal some overfitting at lag=20, so cap it here

        # loop across cells, detrend, calculate BIC with via constrained_foopsi
        process_start = time.process_time(); best_lags = []; bic_vals_all = []
        for x in range(self.F.shape[0]):

            # f_detrended = sgolay/mad method (see function and Grosmark 2021). 
            # sn=std of the 'baseline/caevent free/noise' distribution used for constrained_foopsi
            f_detrended, sn = self.sgolay_detrend(f = self.F[x,:] - self.Fneu[x,:])

            # Range of possible lags to test
            lag_range = range(1, max_lags+1)

            # Calculate BIC for each lag
            process_start = time.process_time()
            #bic_values = [calculate_bic(f_detrended, sn, lag) for lag in lag_range]
            Cout = [foopsi(f_detrended=f_detrended, sn=sn, lags=lagi) for lagi in lag_range]
            Cout = np.array(Cout)
            print("Time to estimate BIC per each lag:",(time.process_time() - process_start)/60,"min")    

            

            # Find the lag with the lowest BIC
            #best_lag = lag_range[np.argmin(bic_values)]
            #print(f"The best lag value is: {best_lag}")

            # save
            #best_lags.append(best_lag)
            #bic_vals_all.append(bic_values)

    def random_peak_generator():
        import numpy as np
        import matplotlib.pyplot as plt

        # Length of the signal
        length = 60000

        # Create a base random signal
        signal = np.random.randn(length)

        # Parameters for peak events
        num_peaks = 50
        peak_duration = 60
        initial_peak_magnitude = 20  # Starting magnitude of peaks
        final_peak_magnitude = 1     # Ending magnitude of peaks

        # Linear decay factor for peak magnitudes
        peak_magnitudes = np.linspace(initial_peak_magnitude, final_peak_magnitude, num_peaks)

        # Evenly spaced peak start indices
        start_indices = np.linspace(0, length - peak_duration, num_peaks).astype(int)

        # Generate peak events with linearly decaying magnitudes
        for i, start_idx in enumerate(start_indices):
            # Create a peak event with linearly decaying magnitude
            peak_event = np.ones(peak_duration) * peak_magnitudes[i]
            
            # Add the peak event to the signal
            signal[start_idx:start_idx + peak_duration] += peak_event

        # Plot the resulting signal
        plt.figure(figsize=(12, 4))
        plt.plot(signal, 'k', linewidth=0.5)
        plt.title('Random Signal with Linearly Decaying Peak Events')
        plt.xlabel('Sample Index')
        plt.ylabel('Signal Amplitude')
        plt.show()

        return signal

    # saves the maxprojection output generated by suite2p
    def save_maxproj_s2p(fpath: str):
        '''
        Function that saves out the max projection image from suite2p, saved as a .tif.
        The saved location will be outside of the suite2p folder

        Args:
            >>> fpath: path to your suite2p folder
        
        '''
        # read data
        F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath)

        # get summary images
        max_proj  = ops['max_proj']
        mean_img  = ops['meanImg']
        mean_imgE = ops['meanImgE']

        # save outputs
        if os.path.split(fpath)[-1] == 'suite2p':
            # eventually make this code a loop that loops over the various planes
            fpath = os.path.split(fpath)[0]
        elif os.path.split(fpath)[-1] == 'plane0':
            fpath = os.path.split(os.path.split(fpath)[0])[0]

        print("Writing summary images to",fpath)
        tf.imwrite(os.path.join(fpath,'max_proj.tif'),
                max_proj)
        tf.imwrite(os.path.join(fpath,'mean_img.tif'),
                mean_img)
        tf.imwrite(os.path.join(fpath,'mean_imgE.tif'),
                mean_imgE)

# can inherit the __init__ from postProcess
#self = cellClassifier(training_sessions_directory=os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierBuildSuite2p'))
#self.build_classifier(auto_feature_select=False, preset_features=False)
#feature_list = ['comp_SNR', 'skewF', 'npix', 'compact', 'asymmetry', 'sd_r']
#self = cellClassifier(load_classifier=True, model_path=os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier.pkl'))
# an excellent feature list

#feature_list = ['comp_SNR', 'skewF', 'fitness', 'corr', 'compact', 'npix_norm', 'npix']
#self.build_classifier(auto_feature_select=False, preset_features=False, feature_list=feature_list)
#self.save_model(filepath=os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier.pkl'))
#self.classify(session_path=r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L608_F_LeftPFC_L6Chr_PFCgcamp6f_L6PAN\SEDS_day2_FOV1_LBC0_optoRec_img\suite2p\plane0")

# self2.save_model(filepath=os.path.join(rf.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier.pkl'))
# self2.classify()
#preferred_features = ['comp_SNR', 'skewF', 'npix', 'compact', 'asymmetry']
#self2.build_classifier(auto_feature_select=False, preset_features=True, preferred_features=preferred_features)
#self = cellClassifier(load_classifier=True, model_path=os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier.pkl'))

# close. The classifier is having trouble on L605 and L607, which it hasn't been trained on as extensively and might be noisier
# when it can't classify anything, it rreturns an error
#
self = cellClassifier(training_sessions_directory=os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierBuildSuite2p'))
#self.check_classifier_loso(auto_feature_select=False, preset_features=False, feature_list=None)
self.build_classifier(preset_features=False, grid_search=True)
#self.save_model(filepath=os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier_svm_pca.pkl'))
#self.classify(session_path=r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L608_F_LeftPFC_L6Chr_PFCgcamp6f_L6PAN\SEDS_day2_FOV1_LBC0_optoRec_img\suite2p\plane0")
#self.classify(session_path=r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L605_M_RightPFC_L6Chr_PFCgcamp6f_L6L5\SEDS_day11_FOV4_optoRec_noProbe_LBC0_img\suite2p\plane0")

# tester sessions
#sessions = os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierTestSuite2p')
#subsessions = os.listdir(sessions)
#[self.classify(session_path=os.path.join(sessions,i,'suite2p','plane0')) for i in subsessions]

#sessions = os.path.join(rootfun.dropbox_root(),'OtherData','John\EXPERIMENTS\LAYER6\Subjects\Imaging')
#subsessions = os.listdir(sessions)

#import os

#imaging_root = os.path.join(
#    rootfun.dropbox_root(),
#    'OtherData', 'John', 'EXPERIMENTS', 'LAYER6', 'Subjects', 'Imaging'
#)

#for subj in os.listdir(imaging_root):
#    subj_path = os.path.join(imaging_root, subj)
#    # skip anything that isn't a folder
#    if not os.path.isdir(subj_path): 
#        continue

#    # now look for any run‐folders under that subject
#    for run in os.listdir(subj_path):
#        run_path = os.path.join(subj_path, run, 'suite2p', 'plane0')
#        if os.path.isdir(run_path):
#            print("Classifying:", subj, "→", run_path)
#            self.classify(session_path=run_path)


#self = cellClassifier(load_classifier=True, model_path=os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier_gridSearch_addedParams_sessionScale.pkl'))
class cellClassifier():
    '''
    Classifier for calcium imaging ROI.
    '''

    def __init__(self, load_classifier: bool = False, model_path: str = None, training_sessions_directory: str = None):
        """
        Args:
            load_classifier: if True, load a previously-saved classifier instead of retraining.
            model_path: path to .pkl file to load/save the classifier.
            training_sessions_directory: where to find raw data for training.
            save_classifier: after training, if True, save the object to model_path.
        """

        if load_classifier:
            if model_path is None or not os.path.isfile(model_path):
                raise ValueError(f"No such model file: {model_path}")
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            # Copy all attributes from the saved instance into self
            self.__dict__.update(saved.__dict__)
            print(f"Loaded classifier from {model_path}")
            return

        # 1) Gather data and train as before
        all_dir = rootfun.list_all_subdirs(training_sessions_directory)
        self.training_sessions = [i for i in all_dir if 'suite2p' in i and 'plane' in i]
        self.df_train = self.gather_classifier_data(self.training_sessions)
        
        #
        #(self.svc,
        # self.scaler,
        # self.selected_features,
        # self.pca,
        # self.n_components,
        # self.idx_rem) = self.build_classifier(
        #    auto_feature_select=False,
        #    preset_features=True
        #)

    def save_model(self, filepath: str):
        """Serialize this entire object to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath: str) -> "cellClassifier":
        """Load a previously-saved classifyCells object from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Pickle file does not contain a {cls.__name__}")
        return obj

    def check_classifier_loso(self,
                        auto_feature_select:    bool = False,
                        preset_features:        bool = False,
                        feature_list:           list = None,
                        ):
        """
        LOSO validation for the classifier. This allows us to check that performance is consistent across animals.
        """
        from imblearn.pipeline    import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm         import SVC
        from sklearn.ensemble    import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import (
            GridSearchCV, GroupShuffleSplit, GroupKFold, cross_val_score
        )

        # 0) clean up and tag each row with its session
        df_clean, idx_rem = self.cleanup_classifier_data(self.df_train)
        # ← you need to have added `session_id` in gather_classifier_data() so it's a column here
        groups = df_clean['mouseName']                     ## ← NEW
        X      = df_clean.drop(columns=['iscell','mouseName'])
        y      = df_clean['iscell']

        # 1) feature‐subset logic (unchanged)
        if preset_features:
            flist = feature_list or ['comp_SNR','skewF','fitness','corr','compact','npix_norm','npix']
            X = X[flist]
        elif feature_list:
            X = X[feature_list]
        self.selected_features = list(X.columns)

        # 2) Optional RFE feature selection
        if auto_feature_select and not preset_features:
            print("Running RFE to select top features…")
            svc_temp = SVC(kernel='linear', class_weight='balanced')
            # try selecting from 1 → all features
            best_score = -1; best_support = None
            for k in range(1, X_train.shape[1]+1):
                rfe = RFE(svc_temp, n_features_to_select=k)
                score = cross_val_score(rfe, X_train, y_train, cv=5).mean()
                if score > best_score:
                    best_score = score
                    best_support = rfe.fit(X_train,y_train).support_
            X_train = X_train.loc[:, best_support]
            X_test  = X_test.loc[:,  best_support]
            print(f"RFE kept {best_support.sum()} features (CV={best_score:.3f})")

        # 3) LOSO sanity check
        loso = GroupKFold(n_splits=groups.nunique())
        loso_scores = cross_val_score(
            SVC(kernel='linear', class_weight='balanced'),
            X, y,
            groups=groups,
            cv=loso,
            scoring='accuracy',
            n_jobs=-1
        )
        print(f"LOSO CV (linear‑SVC) per‑mouse accuracies: {loso_scores.round(3)}  → mean {loso_scores.mean():.3f}")

    def build_classifier(self,
                        auto_feature_select:    bool = False,
                        preset_features:        bool = False,
                        feature_list:           list = None,
                        grid_search:            bool = True,
                        per_session_scaling:    bool = True   ## ← NEW
                        ):
        """
        Train a cell/non-cell classifier, with options for:
        • RFE feature‑selection
        • full pipeline GridSearch over SMOTE/PCA/classifier
        • per‑session scaling + LOSO validation  
        """
        from imblearn.pipeline    import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm         import SVC
        from sklearn.ensemble    import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import (
            GridSearchCV, GroupShuffleSplit, GroupKFold, cross_val_score
        )

        # 0) clean up and tag each row with its session
        df_clean, idx_rem = self.cleanup_classifier_data(self.df_train)
        # ← you need to have added `session_id` in gather_classifier_data() so it's a column here
        groups = df_clean['mouseName']                     ## ← NEW
        X      = df_clean.drop(columns=['iscell','mouseName'])
        y      = df_clean['iscell']

        # 1) feature‐subset logic (unchanged)
        if preset_features:
            flist = feature_list or ['comp_SNR','skewF','fitness','corr','compact','npix_norm','npix']
            X = X[flist]
        elif feature_list:
            X = X[feature_list]
        self.selected_features = list(X.columns)

        # 2) Optional RFE feature selection
        if auto_feature_select and not preset_features:
            print("Running RFE to select top features…")
            svc_temp = SVC(kernel='linear', class_weight='balanced')
            # try selecting from 1 → all features
            best_score = -1; best_support = None
            for k in range(1, X_train.shape[1]+1):
                rfe = RFE(svc_temp, n_features_to_select=k)
                score = cross_val_score(rfe, X_train, y_train, cv=5).mean()
                if score > best_score:
                    best_score = score
                    best_support = rfe.fit(X_train,y_train).support_
            X_train = X_train.loc[:, best_support]
            X_test  = X_test.loc[:,  best_support]
            print(f"RFE kept {best_support.sum()} features (CV={best_score:.3f})")

        # 3) train/test split — either grouped or plain
        if per_session_scaling:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_train, X_test   = X.iloc[train_idx],    X.iloc[test_idx]
            y_train, y_test   = y.iloc[train_idx],    y.iloc[test_idx]
            grp_train, grp_test = groups.iloc[train_idx], groups.iloc[test_idx]

            # per‑session scalers
            self.session_scalers = {}
            Xtr_scaled = pd.DataFrame(index=X_train.index, columns=X_train.columns)
            for sess in grp_train.unique():
                mask = grp_train == sess
                sc = StandardScaler().fit(X_train.loc[mask])
                Xtr_scaled.loc[mask] = sc.transform(X_train.loc[mask])
                self.session_scalers[sess] = sc

            # global fallback scaler
            self.global_scaler = StandardScaler().fit(X_train)
            # transform any sessions not seen in train with global scaler
            Xte_scaled = pd.DataFrame(index=X_test.index, columns=X_test.columns)
            for sess in grp_test.unique():
                mask = grp_test == sess
                sc = self.session_scalers.get(sess, self.global_scaler)
                Xte_scaled.loc[mask] = sc.transform(X_test.loc[mask])

            X_train, X_test = Xtr_scaled, Xte_scaled

        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0, stratify=y
            )

        # 4) Grid search pipeline
        if grid_search:

            # build the base pipeline that GridSearch will tune
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote',   SMOTE(random_state=0)),     # will be overridden to ‘passthrough’ by GridSearch
                ('pca',     PCA()),                     # likewise can be switched off
                ('clf',     SVC(class_weight='balanced', probability=True))
            ])

            param_grid = [
                # ── Linear SVC, PCA ON ────────────────────────────────────────────────────
                {
                'smote': [SMOTE(random_state=0), None],
                'pca':   [PCA()],
                'pca__n_components': [None, min(X_train.shape[1], 5), min(X_train.shape[1], 10)],
                'clf':   [SVC(kernel='linear', class_weight='balanced', probability=True)],
                'clf__C': [0.1, 1, 10]
                },
                # ── Linear SVC, PCA OFF ───────────────────────────────────────────────────
                {
                'smote': [SMOTE(random_state=0), None],
                'pca':   [None],   # no PCA step
                'clf':   [SVC(kernel='linear', class_weight='balanced', probability=True)],
                'clf__C': [0.1, 1, 10]
                },

                # ── RBF SVC, PCA ON ───────────────────────────────────────────────────────
                {
                'smote': [SMOTE(random_state=0), None],
                'pca':   [PCA()],
                'pca__n_components': [None, min(X_train.shape[1], 5)],
                'clf':   [SVC(kernel='rbf', class_weight='balanced', probability=True)],
                'clf__C':     [0.1, 1, 10],
                'clf__gamma': ['scale','auto']
                },
                # ── RBF SVC, PCA OFF ──────────────────────────────────────────────────────
                {
                'smote': [SMOTE(random_state=0), None],
                'pca':   [None],
                'clf':   [SVC(kernel='rbf', class_weight='balanced', probability=True)],
                'clf__C':     [0.1, 1, 10],
                'clf__gamma': ['scale','auto']
                },

                # ── RandomForest (no PCA tuning) ─────────────────────────────────────────
                {
                'smote': [SMOTE(random_state=0), None],
                'pca':   [None],  # skip PCA entirely for trees
                'clf':   [RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=0)],
                'clf__n_estimators': [100, 200],
                'clf__max_depth':    [None, 5, 10]
                },

                # ── LogisticRegression, PCA ON ───────────────────────────────────────────
                {
                'smote': [SMOTE(random_state=0), None],
                'pca':   [PCA()],
                'pca__n_components': [None, min(X_train.shape[1], 5)],
                'clf':   [LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)],
                'clf__C': [0.01, 0.1, 1, 10]
                },
                # ── LogisticRegression, PCA OFF ──────────────────────────────────────────
                {
                'smote': [SMOTE(random_state=0), None],
                'pca':   [None],
                'clf':   [LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)],
                'clf__C': [0.01, 0.1, 1, 10]
                }
            ]

            # gridSearch
            gs = GridSearchCV(
                pipeline, param_grid,
                cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            gs.fit(X_train, y_train)

            print("Best pipeline configuration:", gs.best_params_)
            best_pipe = gs.best_estimator_

            # unpack
            self.scaler     = best_pipe.named_steps['scaler']
            self.smote_used = best_pipe.named_steps['smote'] is not None
            self.pca        = best_pipe.named_steps['pca']
            if self.pca is None:
                print("PCA was skipped")
            self.svc        = best_pipe.named_steps['clf']
            self.pipeline   = best_pipe

            # final test
            y_pred = best_pipe.predict(X_test)
            print("Test accuracy:", accuracy_score(y_test, y_pred))
            print(classification_report(y_test, y_pred))

        # 5) Fallback manual flow
        else:
            print("Running manual scaler → SMOTE → PCA → linear SVC…")
            # scale
            scaler = StandardScaler().fit(X_train)
            Xt = scaler.transform(X_train)
            Xv = scaler.transform(X_test)

            # SMOTE
            sm = SMOTE(random_state=0)
            Xt, yt = sm.fit_resample(Xt, y_train)

            # PCA
            pca = PCA(n_components=min(Xt.shape[1]-1, 5))
            Xt = pca.fit_transform(Xt)
            Xv = pca.transform(Xv)

            # train
            svc = SVC(kernel='linear', C=1, class_weight='balanced', probability=True)
            svc.fit(Xt, yt)
            self.scaler   = scaler
            self.smote    = sm
            self.pca      = pca
            self.svc      = svc
            self.pipeline = None

            # test
            y_pred = svc.predict(Xv)
            print("Test accuracy:", accuracy_score(y_test, y_pred))
            print(classification_report(y_test, y_pred))

        # return for convenience
        return self.svc

    # Initialize an empty DataFrame
    def gather_classifier_data(self, classifier_sessions):

        # load and organize data
        df_all = pd.DataFrame(); F_all = []
        for fpath in classifier_sessions:
            #fpath = sessi
            print("Collecting and organizing data from:",fpath)

            # squirrel mouse name
            mouseName = os.path.split(os.path.split(os.path.split(fpath)[0])[0])[-1].split('_')[0]

            # load data
            F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)
            C = np.load(os.path.join(fpath, 'C.npy'), allow_pickle=True)
            #S = np.load(os.path.join(fpath, 'suite2p', 'plane0', 'S.npy'), allow_pickle=True)

            # make mouseName variable 
            mouseName_array = np.array([mouseName] * F.shape[0])

            # assert shape
            assert C.shape==F.shape, "The shape of your C and F variables do not match. Rerun cleanup raw traces"

            # get morphological features
            morph_features = self.cell_morphology(stat=stat, ops=ops)

            # get physiology features
            cell_textures = self.cell_texture(F, Fneu)
            cell_dynamics = self.cell_dynamics(F, Fneu, C, ops)

            # baseline operation
            Ndet = dcnv.preprocess(
                F=Fneu,
                baseline=ops['baseline'],
                win_baseline=ops['win_baseline'],
                sig_baseline=ops['sig_baseline'],
                fs=ops['fs'],
                prctile_baseline=ops['prctile_baseline']
            )

            # Run skew
            skF = stats.skew(F-Fneu, axis=1)
            skN = stats.skew(Ndet, axis=1)

            # SNR
            varF = np.var(F-Fneu, axis=1)
            varN = np.var(Ndet, axis=1)
            #snr = varF / varN

            # Correlate F to C
            smoothF = gaussian_filter1d(F-Fneu, sigma=7.5*2, axis=1)
            correlation = np.array([np.corrcoef(smoothF[i], C[i])[0, 1] for i in range(smoothF.shape[0])])

            # Fitness, traces == C
            fitness, erfc, sd_r, md = compute_event_exceptionality(traces=F-Fneu)

            # Number of timesteps to consider when testing new neuron candidates
            min_SNR        = 2.5
            min_SNR_reject = 0.5
            decay_time     = 0.7
            frate          = 7.5
            N_samples      = np.ceil(frate * decay_time).astype(int)

            # -- this is from CaImAns code to estimate an SNR -- #

            # Inclusion probability of noise transient
            thresh_fitness_raw = special.log_ndtr(-min_SNR) * N_samples

            # Threshold on time variability
            fitness_min = special.log_ndtr(-min_SNR) * N_samples

            # Components with SNR lower than 0.5 will be rejected
            thresh_fitness_raw_reject = special.log_ndtr(-min_SNR_reject) * N_samples
            comp_SNR = -norm.ppf(np.exp(fitness / N_samples))

            # --------------------------------------------------- #

            # PNR
            #pnr_data = pnr(F=F-Fneu, Fneu=Ndet)

            # Get classifier stats
            npix_norm = np.array([i['npix_norm'] for i in stat])
            npix      = np.array([i['npix'] for i in stat])
            compact   = np.array([i['compact'] for i in stat])
            aspect    = np.array([i['aspect_ratio'] for i in stat]) # aspect ratio is how elongated a component is

            # --- Some important filtering and rescuing steps -- #

            # calculate the median asymmetry of the F signal
            #asymmetry = get_asymmetry(F = F, Fneu = Fneu, C = C, fs = 7.5)

            # identify "not cells" that have low compactness and high skew, then send them to "is cell" category
            #iscell = reject_cell(iscell=iscell, stat=stat, compact=compact, skF=skF, asymmetry=None)
            
            # replacement for 'reject_cell', where a ratio of pixel values are used to identify cells. This is flexible
            iscell = self.ROI_pixel_filter(iscell, ops, stat)

            # find cells near one another amongst the true positives, then if strongly correlated, remove the weaker signal
            #iscell = reject_overlapping_roi(stat=stat, F=F, Fneu=Fneu, C=C, iscell=iscell, fs=7.5)

            # Create a DataFrame for the current session
            df_session = pd.DataFrame(data={
                'mouseName': mouseName_array,
                'iscell': iscell,  # Ensure correct shape
                'skewF': skF,
                'skewN': skN,
                'corr': correlation,
                'fitness': fitness,
                'sd_r': sd_r,
                'md': md,
                'npix_norm': npix_norm, #added
                'npix': npix,
                'compact': compact, # added
                'comp_SNR': comp_SNR,
            })

            # combine with other dataframes
            df_session = pd.concat([df_session, morph_features], axis=1)
            df_session = pd.concat([df_session, cell_textures], axis=1)
            df_session = pd.concat([df_session, cell_dynamics], axis=1)

            # Append the current session DataFrame to the main DataFrame
            df_all = pd.concat([df_all, df_session], ignore_index=True)

        # cache
        return df_all

    # function to remove nan/inf values
    def cleanup_classifier_data(self, df_all):

        # identify nan or inf values
        df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        idx_rem = df_all.index[df_all.isna().any(axis=1)]

        for i in idx_rem:
            print(f'Detected and removed NaN at: {i}')

        # remove them
        df_clean = df_all.drop(idx_rem)

        # store
        df_clean = df_clean
        idx_rem = idx_rem

        return df_clean, idx_rem

    @staticmethod
    def cell_morphology(stat, ops):
        from skimage.measure import regionprops
        import numpy as np

        # read data
        #F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)
        H, W = ops['Ly'], ops['Lx']
        dims = (H, W)

        morph_features = []
        for s in stat:
            mask = np.zeros(dims, dtype=bool)
            mask[s['ypix'], s['xpix']] = True
            props = regionprops(mask.astype(np.uint8))[0]
            morph_features.append({
                'area':              props.area,
                'perimeter':         props.perimeter,
                'eccentricity':      props.eccentricity,
                'solidity':          props.solidity,
                'extent':            props.extent,
                'major_axis_length': props.major_axis_length,
                'minor_axis_length': props.minor_axis_length,
                'orientation':       props.orientation,
                'convex_area':       props.convex_area,
                'bbox_area':         props.bbox_area,
            })

        morph_dataframe = pd.DataFrame(morph_features)
        return morph_dataframe

    @staticmethod
    def cell_texture(F, Fneu):

        from scipy.stats import skew, kurtosis
        #F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)

        texture_features = []
        for i in range(F.shape[0]):
            trace = F[i]
            neu   = Fneu[i]
            corr  = np.corrcoef(trace, neu)[0,1]

            texture_features.append({
                'mean_fluo':         np.mean(trace),
                'median_fluo':       np.median(trace),
                'std_fluo':          np.std(trace),
                'cv_fluo':           np.std(trace) / np.mean(trace),
                'max_fluo':          np.max(trace),
                'min_fluo':          np.min(trace),
                #'skew_fluo':         skew(trace),
                'kurtosis_fluo':     kurtosis(trace),
                'neuropil_corr':     corr,
            })

        texture_dataframe = pd.DataFrame(texture_features)
        return texture_dataframe

    @staticmethod
    def cell_dynamics(F, Fneu, C, ops):

        from scipy.signal import find_peaks, peak_widths
        #F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)
        #C = np.load(os.path.join(fpath, 'C.npy'), allow_pickle=True)

        ts_features = []
        for i in range(C.shape[0]):
            print(f"Calculating time series features for cell {i+1} of {C.shape[0]}")
            
            # temp vars
            c        = C[i]
            f        = F[i]-Fneu[i]
            fs       = ops['fs']
            duration = len(c) / fs

            # get cell calcium event properties
            peaks, widths_s, rise_s, fall_s, iei, amps, props, asymmetry = calcium_events(c=c, Fc=f, fs = fs, detrend_data = True, plot_progress = False)

            ts_features.append({
                'event_count':        len(peaks),
                'event_rate':         len(peaks) / duration,   
                'mean_amp':           np.mean(amps),
                'cv_amp':             np.std(amps)    / np.mean(amps),
                'mean_width_s':       np.mean(widths_s),
                'cv_width_s':         np.std(widths_s)  / np.mean(widths_s),
                'mean_iei_s':         np.mean(iei),
                'cv_iei':             np.std(iei)     / np.mean(iei),
                'median_rise_time':   np.median(peaks - props['right_bases']) / fs,
                'medial_fall_time':   np.median(props['left_bases'] - peaks) / fs,
                'asymmetry':          asymmetry,
            })

        ts_dataframe = pd.DataFrame(ts_features)
        return ts_dataframe

    @staticmethod
    def ROI_pixel_filter(iscell, ops, stat, min_pixel_ratio: float = 100/(512*512) ):
        '''
        Filters out ROIs based on a minimum pixel ratio.

        Args:
            fpath (str): Path to the suite2p data folder.
            min_pixel_ratio (float): Minimum pixel ratio for filtering ROIs.

        Returns:
            iscell: boolean array filtered for ROIs based on pixel ratio.
        '''

        # Get the number of pixels in each ROI
        num_pixels = np.array([len(s['ypix']) for s in stat])

        # get the shape of the current matrix
        matrix_divisor = ops['Ly'] * ops['Lx']

        # any ROI with less than the min_pixel_ratio, set iscell==False
        iscell[ (num_pixels / matrix_divisor) < min_pixel_ratio] = False

        return iscell

    def predict_cell(self, df_predict):
        """
        Predict 'iscell' on a new DataFrame using whichever pipeline
        was chosen in build_classifier (grid_search OR manual flow).

        Parameters
        ----------
        df_predict : pd.DataFrame
            Must contain all the same feature columns (and 'mouseName', 'iscell')
            that were used in training.

        Returns
        -------
        predictions : np.ndarray of bool
        probabilities : np.ndarray, shape (n_samples, n_classes)
        decision_scores : np.ndarray, shape (n_samples,) or None if unavailable
        """
        import numpy as np

        # 1) Clean out NaNs/Infs and remember who got dropped
        df_clean, idx_rem = self.cleanup_classifier_data(df_predict)
        
        # 2) Subset to the features we trained on
        X = df_clean.drop(columns=['mouseName','iscell'])
        X = X[self.selected_features]

        # 3) Run through the saved pipeline if it exists
        if getattr(self, 'pipeline', None) is not None:

            # we have a fitted ImbPipeline → use it
            predictions   = self.pipeline.predict(X)
            probabilities = self.pipeline.predict_proba(X)
            if hasattr(self.pipeline, 'decision_function'):
                decision_scores = self.pipeline.decision_function(X)
            else:
                decision_scores = np.zeros(len(predictions))

        else:
            # manual flow: scale → (no SMOTE) → PCA → SVC
            #  a) scale
            X_scaled = self.scaler.transform(X)
            #  b) PCA if it was used
            if self.pca is not None:
                X_proc = self.pca.transform(X_scaled)
            else:
                X_proc = X_scaled
            #  c) classify
            predictions     = self.svc.predict(X_proc)
            probabilities   = self.svc.predict_proba(X_proc)
            decision_scores = self.svc.decision_function(X_proc)

        # 4) re‑insert dropped rows as False/0
        if len(idx_rem) > 0:
            n_total = len(predictions) + len(idx_rem)
            # assume binary: probabilities.shape[1] gives number of classes
            final_preds     = np.zeros(n_total,             dtype=bool)
            final_probs     = np.zeros((n_total, probabilities.shape[1]))
            final_scores    = np.zeros(n_total)
            
            pred_i = 0
            for i in range(n_total):
                if i in idx_rem:
                    # leave zeros
                    continue
                final_preds[i]  = predictions[pred_i]
                final_probs[i]  = probabilities[pred_i]
                final_scores[i] = decision_scores[pred_i]
                pred_i += 1

            predictions     = final_preds
            probabilities   = final_probs
            decision_scores = final_scores

        print("Classification complete.")
        return predictions, probabilities, decision_scores

    # function that calls in the iscell variable and rewrites it according to predict_cell predictions
    def rewrite_data(self, predict_sessions, predictions, probabilities):
        import scipy.io as sio
        from datetime import datetime

        for i in predict_sessions:
            
            # read og iscell
            F, Fneu, spks, stat, ops, iscell0, __ = read_s2p(fpath=i)
            C = np.load(os.path.join(i, 'C.npy'), allow_pickle=True)
            S = np.load(os.path.join(i, 'S.npy'), allow_pickle=True)
            iscell = np.load(os.path.join(i,'iscell.npy'), allow_pickle=True)

            # rewrite
            iscell_og = iscell.copy()
            del iscell

            # rewrite iscell
            iscell = np.zeros(iscell_og.shape)
            iscell[predictions, 0] = 1.0
            iscell[:, 1] = probabilities[:, 1]

            # --- Some important filtering and rescuing steps -- #
            iscell_in = iscell[:,0].astype(bool)

            # compactness
            compact = np.array([i['compact'] for i in stat])

            # skew
            skFc = stats.skew(F-Fneu, axis=1)
            skF = stats.skew(F, axis=1)

            # identify "not cells" that have low compactness and high skew, then send them to "is cell" category
            iscell_out = self.ROI_pixel_filter(iscell=iscell_in, ops=ops, stat=stat)

            # in really noisy recordings, skew filter needs to be added and applied. This is a really loose definition. In reality, cells don't usually fall below 1.0 or 1.2
            iscell_out[ (skF < .9) | (skFc < .9) ] = False

            # rescue - attempting new functionality 5/6/2025
            #iscell_out = rescue_cell(fpath=i, iscell_in=iscell_out)

            # find cells near one another amongst the true positives, then if strongly correlated, remove the weaker signal
            #iscell_out = iscell_in.copy()
            iscell_out = reject_overlapping_roi(stat=stat, F=F, Fneu=Fneu, C=C, iscell=iscell_out, fs=7.5)

            # regenerate iscell. Note that we are replacing the iscell with iscell_out because iscell_out is processed probabilities from the classifier
            del iscell
            iscell = np.zeros(iscell_og.shape)
            iscell[iscell_out, 0] = 1.0
            iscell[:, 1] = probabilities[:, 1]

            # save
            np.save(os.path.join(i, 'iscell.npy'), iscell, allow_pickle=True)
            np.save(os.path.join(i, 'iscell_og.npy'), iscell_og, allow_pickle=True)

            # save to .mat
            ops_matlab = ops.copy()
            if ops_matlab.get("date_proc"):
                try:
                    ops_matlab["date_proc"] = str(
                        datetime.strftime(ops_matlab["date_proc"], "%Y-%m-%d %H:%M:%S.%f"))
                except:
                    pass        
            sio.savemat(os.path.join(i, 'Fall.mat'), mdict = {'F': F, 'Fneu': Fneu, 'iscell': iscell, 'stat': stat, 'C': C, 'S': S, 'ops': ops_matlab, 's2pSpk': spks})
            print("Saved iscell to", os.path.join(i, 'iscell.npy'))
            print("Saved old 'iscell' to", os.path.join(i, 'iscell_og.npy'))

    # classify
    def classify(self, session_path: str):

        # gather variables
        df_predict = self.gather_classifier_data([session_path])

        # test classifier
        predictions, probabilities, decision_scores = self.predict_cell(df_predict = df_predict)


        self.rewrite_data(predict_sessions = [session_path], predictions=predictions, probabilities = probabilities)


# ---------------------------------------- #
# --  HELPER AND STAND ALONE FUNCTIONS  -- #
# ---------------------------------------- #

def caiman_snr(fpath: str):

    # load data
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)
    C = np.load(os.path.join(fpath, 'C.npy'), allow_pickle=True)
    #S = np.load(os.path.join(fpath, 'suite2p', 'plane0', 'S.npy'), allow_pickle=True)

    # assert shape
    assert C.shape==F.shape, "The shape of your C and F variables do not match. Rerun cleanup raw traces"

    # Fitness, traces == C
    fitness, erfc, sd_r, md = compute_event_exceptionality(traces=F-Fneu)

    # Number of timesteps to consider when testing new neuron candidates
    min_SNR        = 2.5
    min_SNR_reject = 0.5
    decay_time     = 0.7
    frate          = 7.5
    N_samples      = np.ceil(frate * decay_time).astype(int)

    # -- this is from CaImAns code to estimate an SNR -- #

    # Inclusion probability of noise transient
    thresh_fitness_raw = special.log_ndtr(-min_SNR) * N_samples

    # Threshold on time variability
    fitness_min = special.log_ndtr(-min_SNR) * N_samples

    # Components with SNR lower than 0.5 will be rejected
    thresh_fitness_raw_reject = special.log_ndtr(-min_SNR_reject) * N_samples
    comp_SNR = -norm.ppf(np.exp(fitness / N_samples))

    return comp_SNR

# a function to add files to the fall.mat variable
def add_to_fall(fpath: str, var_names: list = ['C.npy', 'S.npy']):
    '''
    A function that takes the Fall.mat variable and adds whatever variables
    in the var_names list

    Args:
        >>> fpath: path to suite2p variables to load
        >>> var_names: list of variables to add to the Fall.mat
                        - The default is C.npy and S.npy as they aren't natively saved to Fall.mat

    John Stout
    '''

    # get path data
    fpath = parse_fpath(fpath = fpath)

    # search for var_names
    dir_contents = os.listdir(fpath)
    vars_found = [i for i in dir_contents if i in var_names]
    print("Discovered", vars_found)

    # load the vars_found data into workspace - thanks copilot :)
    data_dict = {os.path.splitext(i)[0]: np.load(os.path.join(fpath, i)) for i in vars_found}
    
    # load the Fall var
    fall_found = 'Fall.mat' in dir_contents
    if fall_found:

        # load data
        from scipy import io as sio
        mat_data = sio.loadmat(os.path.join(fpath,'Fall.mat'))

        # append
        mat_data.update(data_dict)

        # save
        print("Saving variables as", os.path.join(fpath, 'Fall.mat'))
        sio.savemat(os.path.join(fpath, 'Fall.mat'), mat_data)

# mechanism to read binary output from suite2p
def read_binary(fpath: str, Lx: int = 512, Ly: int = 512):

    # get the correct directory
    fpath = parse_fpath(fpath=fpath)

    # read file for the ops
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)

    # now search for binary file
    fbin = [i for i in os.listdir(fpath) if '.bin' in i]
    if len(fbin) > 0:
        print(f'Discovered and reading .binary file: {fbin}')
        bpath = os.path.join(fpath,fbin[0])

    # Create a BinaryFile object
    n_frames, Ly, Lx = ops["nframes"], ops["Ly"], ops["Lx"]

    # read
    print("Reading BinaryFile, please wait.....")
    binFile = BinaryFile(Ly=Ly, Lx=Lx, filename=bpath, n_frames=n_frames)

    # Read the data into a numpy array
    data = binFile.data

    return data

# makes empty suite2p folders in subdirectories. The fpaths input is a
def make_empty_suite2p(fpaths):
    '''
    makeEmptySuite2p

    Creates empty suite2p folders in subdirectories.

    Args:
        >>> fpaths: Master directory with a bunch of sessions that you want empty suite2p folders created in
    '''

    # get subdirectories
    fpath = os.listdir(fpaths)
    for i in fpath:

        # make temporary directory
        temp_dir = os.path.join(fpaths,i)
        newPath = os.path.join(fpaths,i,'suite2p','plane0')

        # create paths
        try:
            os.makedirs(newPath)
            print("Created path",newPath)
        except:
            pass

# function to rescue cells
def reject_cell(iscell, stat, compact, skF, asymmetry):
    '''
    This code rescues candidate false rejection cells based on high skew and low compactness,
    then rejects cells with low asymmetry

    John Stout
    '''
    
    print("Removing cells with 1) low asymmetrym 2) low pixel counts and 3) low skew")

    # if there are <100 pixels in a cell, toss it
    npix = np.array([i['npix'] for i in stat])
    iscell[npix<100]=False

    # if there are <100 pixels in a cell, toss it
    #skew = np.array([i['skew'] for i in stat])
    #iscell[skew<1.2]=False   

    # rejection time - this actually seems to capture non-asymettrical cells but also
    # really noisy cells because the event peaks are poorly estimated
    #asymmetry_cutoff = 0.4 # after finding cases of good cells > .4 but bad <=.4
    #rejected_cells = np.where(asymmetry <= asymmetry_cutoff)[0]
    #iscell[rejected_cells]=False

    # rescue non-cells - these criterion are largely good
    #rescued_cells = np.where(np.logical_and.reduce([iscell==False, compact <= 1.05, skF > 2.0]))[0]
    #iscell[rescued_cells]=True 

    return iscell

# function to rescue cells
def rescue_cell(fpath, iscell_in):
    '''
    This code rescues candidate false rejection cells based on high skew and low compactness,
    then rejects cells with low asymmetry

    John Stout
    '''
    
    print("Rescuing cells...")

    # read data
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)
    C = np.load(os.path.join(fpath, 'C.npy'), allow_pickle=True)
    skew = np.array([i['skew'] for i in stat])

    # comp_snr > 99th percentile asymmetry > 1
    snr = caiman_snr(fpath=fpath)

    # get the snr percentile
    percentile_90 = np.percentile(snr, 90)

    # rescue a cell if its comp_SNR > 95th percentile and asymmetry > 1.0
    asymmetry = get_asymmetry(F, Fneu, C, fs = 7.5)

    # npix
    npix = np.array([i['npix'] for i in stat])

    # the issue using a percentile is if there are only a few cells, 90th percentile might not be a good cell. So this is hard coded.
    meant_for_true = np.where( ( (snr > 5) | (skew > 2.0) ) & (asymmetry >= 0.4) & (npix > 100) & (iscell_in==False))[0]
    print("Rescuing cells", [i for i in meant_for_true])

    # rescue cells
    iscell_in[meant_for_true] = True

    return iscell_in

# rejecting overlapping ROI
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
                idx_peaks_i  = calcium_events(c = C[i], Fc = Fcor[i])[0]
                idx_peaks_ii = calcium_events(c = C[ii], Fc = Fcor[ii])[0]
                
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
                        iscell[ii
                               ] = False
                    elif skew[i] < skew[ii]:
                        print(f'Sending cell{i} to the "Not Cell" category')
                        iscell[i] = False
    return iscell

# updated calcium events function, converted to python for improved peak detection
# F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath)
# C = np.load(os.path.join(fpath, 'C.npy'), allow_pickle=True)
#c = C[0]; Fc = F[0]-Fneu[0]
def calcium_events(c, Fc,
                   fs: float = 7.5,
                   detrend_data: bool = True,
                   plot_progress: bool = False):
    """
    Detect calcium‐event peaks in a calcium trace `c` and (optionally) detrended fluorescence `Fc`.

    Steps:
      1. Optionally detrend `Fc` via sgolay_detrend(Fc).
      2. Detect peaks in `c` above its 99th percentile, with a minimum spacing of 15 s.
      3. Compute half‑widths at half‑height for each peak to get rise/fall times.
      4. Compute inter‑event intervals (IEI) and peak amplitudes.
      5. Compute an asymmetry metric: median |fall_time – rise_time|.

    Parameters
    ----------
    c : 1D array
        Raw calcium trace.
    Fc : 1D array
        Fluorescence trace (will be detrended if `detrend_data=True`).
    fs : float, default=7.5
        Sampling frequency (Hz).
    detrend_data : bool, default=True
        Whether to run Fc through sgolay_detrend before anything else.
    plot_progress : bool, default=False
        If True, shows diagnostic plots of peaks and half‑height crossings.

    Returns
    -------
    peaks      : 1D int array
        Indices of detected calcium peaks.
    widths_s   : 1D float array
        Full width at half‑height, in seconds.
    rise_s     : 1D float array
        Time from half‑height crossing up to the peak, in seconds.
    fall_s     : 1D float array
        Time from peak down to half‑height crossing, in seconds.
    iei        : 1D float array
        Inter‑event intervals, in seconds.
    amps       : 1D float array
        Peak amplitudes (height above baseline).
    props      : dict
        The properties dict returned by `find_peaks` (e.g. `'peak_heights'`).
    asymmetry  : float
        Median absolute difference between fall_s and rise_s.
    """

    # 1) Detrend fluorescence if requested
    if detrend_data:
        Fc = sgolay_detrend(Fc)

    # 2) Peak detection parameters
    height_thr = np.percentile(c, 99)
    min_dist   = int(fs * 15)  # at least 15 seconds between peaks

    peaks, props = find_peaks(
        c,
        height=height_thr,
        distance=min_dist,
        prominence=0.5
    )
    amps = props['peak_heights']

    # 3) Half‑width at half‑height and crossing points
    widths, _, left_ips, right_ips = peak_widths(c, peaks, rel_height=0.5)
    widths_s = widths / fs
    rise_s   = (peaks - left_ips)  / fs
    fall_s   = (right_ips - peaks) / fs

    # 4) Inter‑event intervals
    iei = np.diff(peaks) / fs

    # 5) Asymmetry metric
    asymmetry = np.median(np.abs(fall_s - rise_s))

    # Diagnostic plotting
    if plot_progress:
        plt.figure()
        plt.plot(c,                 color='k', label='Calcium (c)')
        plt.scatter(peaks, c[peaks],color='r', label='Peaks')
        y_left  = np.interp(left_ips,  np.arange(len(c)), c)
        y_right = np.interp(right_ips, np.arange(len(c)), c)
        plt.scatter(left_ips,  y_left,  color='g', label='Rise @50%')
        plt.scatter(right_ips, y_right, color='m', label='Fall @50%')
        plt.xlabel('Sample index')
        plt.ylabel('Amplitude')
        plt.title('Calcium Peaks & Half‑Height Crossings')
        plt.legend()
        plt.show()

    return peaks, widths_s, rise_s, fall_s, iei, amps, props, asymmetry

# function to compile cell parameters
def compile_cell_parameters(fpath: str):
    '''
    Compiles a handful of useful parameters from suite2p and caiman estimates into a dataframe, 'df_all'

    Args:
        >>> fpath: path to suite2p data
    Returns:    
        >>> df_all: dataframe with all the parameters of interest
    
    John Stout
    '''

    print("Working on",fpath)

    # squirrel mouse name
    mouseName = os.path.split(os.path.split(os.path.split(fpath)[0])[0])[-1].split('_')[0]

    # load data
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)
    C = np.load(os.path.join(fpath, 'C.npy'), allow_pickle=True)
    #S = np.load(os.path.join(fpath, 'suite2p', 'plane0', 'S.npy'), allow_pickle=True)

    # make mouseName variable 
    mouseName_array = np.array([mouseName] * F.shape[0])

    # assert shape
    assert C.shape==F.shape, "The shape of your C and F variables do not match. Rerun cleanup raw traces"

    # baseline operation
    Ndet = dcnv.preprocess(
        F=Fneu,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )

    # Run skew
    skF = stats.skew(F-Fneu, axis=1)
    skN = stats.skew(Ndet, axis=1)

    # SNR
    varF = np.var(F-Fneu, axis=1)
    varN = np.var(Ndet, axis=1)
    snr = varF / varN

    # Correlate F to C
    smoothF = gaussian_filter1d(F-Fneu, sigma=7.5*2, axis=1)
    correlation = np.array([np.corrcoef(smoothF[i], C[i])[0, 1] for i in range(smoothF.shape[0])])

    # Fitness, traces == C
    fitness, erfc, sd_r, md = compute_event_exceptionality(traces=F-Fneu)

    # Number of timesteps to consider when testing new neuron candidates
    min_SNR        = 2.5
    min_SNR_reject = 0.5
    decay_time     = 0.7
    frate          = 7.5
    N_samples      = np.ceil(frate * decay_time).astype(int)

    # -- this is from CaImAns code to estimate an SNR -- #

    # Inclusion probability of noise transient
    thresh_fitness_raw = special.log_ndtr(-min_SNR) * N_samples

    # Threshold on time variability
    fitness_min = special.log_ndtr(-min_SNR) * N_samples

    # Components with SNR lower than 0.5 will be rejected
    thresh_fitness_raw_reject = special.log_ndtr(-min_SNR_reject) * N_samples
    comp_SNR = -norm.ppf(np.exp(fitness / N_samples))

    # --------------------------------------------------- #

    # PNR
    pnr_data = pnr(F=F-Fneu, Fneu=Ndet)

    # Get classifier stats
    npix_norm = np.array([i['npix_norm'] for i in stat])
    npix      = np.array([i['npix'] for i in stat])
    compact   = np.array([i['compact'] for i in stat])
    aspect    = np.array([i['aspect_ratio'] for i in stat]) # aspect ratio is how elongated a component is

    # --- Some important filtering and rescuing steps -- #

    # calculate the median asymmetry of the F signal
    asymmetry = get_asymmetry(F = F, Fneu = Fneu, C = C, fs = 7.5)

    # identify "not cells" that have low compactness and high skew, then send them to "is cell" category
    iscell = reject_cell(iscell=iscell, stat=stat, compact=compact, skF=skF, asymmetry=asymmetry)

    # consider a rescue step
    iscell = rescue_cell(fpath=fpath, iscell_in=iscell)

    # find cells near one another amongst the true positives, then if strongly correlated, remove the weaker signal
    iscell = reject_overlapping_roi(stat=stat, F=F, Fneu=Fneu, C=C, iscell=iscell, fs=7.5)

    # Create a DataFrame for the current session
    df_session = pd.DataFrame(data={
        'mouseName': mouseName_array,
        'iscell': iscell,  # Ensure correct shape
        'snr': snr,
        'skewF': skF,
        'skewN': skN,
        'corr': correlation,
        'fitness': fitness,
        'sd_r': sd_r,
        'md': md,
        'pnr': pnr_data,
        'npix_norm': npix_norm, #added
        'npix': npix,
        'compact': compact, # added
        #'aspect_ratio': aspect, # added
        'comp_SNR': comp_SNR,
        'asymmetry': asymmetry # added
    })

    # Append the current session DataFrame to the main DataFrame
    df_all = pd.concat([df_all, df_session], ignore_index=True)

    return df_all

# pnr
def pnr(F, Fneu):

    # Calculate 95th percentiles
    pk95F = np.percentile(np.sort(np.abs(F), axis=1), 95, axis=1)
    pk95N = np.percentile(np.sort(np.abs(Fneu), axis=1), 95, axis=1)

    # Calculate PNR (peak noise ratio)
    pnr = pk95F / pk95N

    return pnr

# this is essentially F0
def baselineF(fpath: str, baseline = None):

    # read outputs
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)

    # suite2p params
    if baseline is None:
        baseline = ops['baseline']
    win_baseline     = ops['win_baseline']
    sig_baseline     = ops['sig_baseline']
    fs               = ops['fs']
    prctile_baseline = ops['prctile_baseline']

    # baseline F
    win = int(win_baseline * fs)
    if baseline == "maximin":
        Flow = gaussian_filter(F, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    elif baseline == "constant":
        Flow = gaussian_filter(F, [0., sig_baseline])
        Flow = np.amin(Flow)
    # this is essentially F0 as obtained from percentiles, so a singular value
    elif baseline == "constant_prctile":
        Flow = np.percentile(F, prctile_baseline, axis=1)
        Flow = np.expand_dims(Flow, axis=1)

    return Flow

# baseline corrected F
def baseline_corrected_F(F, Fneu, ops):

    """
    This code was taken from the suite2p website to provide baseline subtracted estimates of F

    https://suite2p.readthedocs.io/en/latest/deconvolution.html

    Args:
        >>> F: your F output from suite2p
        >>> Fneu: Your neuropil
        >>> ops: your options variable
    """

    # load traces and subtract neuropil
    Fc = F - ops['neucoeff'] * Fneu

    # baseline operation
    Fc = dcnv.preprocess(
        F=Fc,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )
    return Fc

    # get spikes
    #spks = dcnv.oasis(F=Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])

# dysfunctional
def spk_to_bool(spk, z_transform = True, spk_thresh = 1):
    '''
    Converts estimated spike data to boolean

    Args:
        >>> spk: the `spk` variable from suite2p
        >>> z_transform: whether or not to z-score your signal. You should only set this to false if you are using a z-scored signal already.
        >>> spk_thresh: std above the mean

    Returns:
        >>> spkbool: boolean variable (0s/1s) for each ROI representing spike or no spike

    John Stout
    '''
    if z_transform:
        spk = stats.zscore(spk, axis=1)

    spkbool = np.zeros(shape=spk.shape)
    for celli in range(len(spk)):
        peaks = np.where(spk[celli] > spk_thresh)
        spkbool[celli,peaks]=1
    return spkbool

# dysfunctional
def snr_pnr(fpath):

    # suite2p outputs
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)

    # get baseline F
    F0 = baselineF(fpath=fpath, baseline = "constant_prctile")

    # F/F0
    Ff0 = F/F0
    zFf0 = stats.zscore(Ff0,axis=1)
    zspk = stats.zscore(spks,axis=1)

    snr = []; pnr = []
    for celli in range(len(blF)):

        # signal-noise ratio estimate not working for me so well

        # not a good metric
        blFf0 = blF[celli]/F0[celli]
        snr.append(np.var(blFf0)*100)
        #snr.append(np.var(blF[celli])/F0[celli])
        #snr.append(np.var(blF[celli]/F0[celli]) * 100)
        
        qnt=np.quantile(np.sort(np.abs(F[celli])),[100])

        # snr < 0.25

        # 
        # blF[celli]/F0[celli] # gives me good values rescaled to -.25 and 1.25
    snr = np.array(snr)
    import pandas as pd
    df = pd.DataFrame((snr.T,iscell.T))

    np.where(snr < .1)

    #snr(x)=var(neuron.C(x,:))./abs(var(neuron.C_raw(x,:)-neuron.C(x,:)));
    #qnt=quantile(sort(abs(neuron.C_raw(x,:))),100);
    #pk95(x)=qnt(end);
    #pnr(x)=pk95(x)./std(neuron.C_raw(x,:));

# -- methods to clean-up the F trace
def sgolay_detrend(f, window=1001):
    '''
    method that detrends signal f, an input argument representing the users fluorescent trace

    Args:
        >>> f: a single cells fluorescent trace
    
    Returns:
        >>> f_detrended: a detrended version of the input 'f' signal
        >>> sn: standard deviation of the event free 'noise' or 'baseline' signal
    
    UPDATES:
        >>> 4/22/25: added flexibility for window size. Default remains 1001

    John Stout
    '''

    # identify candidate outlier events (signal)
    mad_f = np.median(np.abs(f - np.median(f)))
    ttimes = np.where(f > np.median(f) + 3 * mad_f)[0]
    
    # now replace events with nan
    f2 = f.copy()
    f2[ttimes] = np.nan # nan out the events
    
    # interpolate candidate signal events to estimate noise
    f2 = np.interp(np.arange(len(f2)), np.arange(len(f2))[~np.isnan(f2)], f2[~np.isnan(f2)])
    sn = np.std(f2) # std of the noise (baseline) distribution

    # subtract the underlying trend (detrend) from the noise-reduced signal
    if f2.shape[0] < 1001:
        print("The size of your f trace is < 1000 samples, adjusting dynamically...")
        window = int(f2.shape[1]/1)
        if window % 2 == 0:
            window += 1            
        f3 - savgol_filter(f2, window, 2) # was 1001, but this is too large for small traces
    else:
        f3 = savgol_filter(f2, window, 2)

    #@AM changed above line to 101 from 1001
    f_detrended = f - f3
    f_detrended = f_detrended.astype(np.float32)

    return f_detrended
