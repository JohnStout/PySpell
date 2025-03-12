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


# ---------------------------------------- #
# ----  MAJOR STAND ALONE FUNCTIONS  ----- #
# ---------------------------------------- #

# fast_suite2p is a major function
#imgpath = r"E:\L6 Experiments\L612\FOV1\SEDS_day11_LBC2_p70_FOV1\SEDS_day11_LBC2_p70_FOV1_img\img.tif"
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
    """

    #___________________________________________#

    # load data lazily
    images = tf.memmap(imgpath, mode="r")

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
        ops['batch_size'] = 5000 # default is 500 but this machine can handle more

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
        output_ops = suite2p.run_s2p(ops=ops, db=db)
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

# ---------------------------------------- #
# --- MAJOR OBJECTS FOR POSTPROCESSING --- #
# ---------------------------------------- #
class postProcess():

    def __init__(self, s2ppath: str):
        import deconvolution as dc

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

        print("runParallel set to False. Iterating through suite2p ROIs...")

        # set empty arrays
        C = []; S = []; f_det_all = []; f_emd_all = []
        total_cells = self.F.shape[0]

        # run constrained foopsi
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
        f3 = savgol_filter(f2, 1001, 2)
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
class classifyCells():

    def __init__(self, load_svm: bool = False, training_sessions_directory: str = None):
        '''
        Args:
            >>> training_session_directory: contains a directory of folders with suite2p/plane0 files to train the model on
        '''

        # TODO add a mechanism to load an svm model
        if load_svm == False:
            all_dir = rootfun.list_all_subdirs(training_sessions_directory)
            self.training_sessions = [i for i in all_dir if 'suite2p' in i and 'plane' in i]

            # gather data to train svm
            self.df_train = self.gather_classifier_data(classifier_sessions=self.training_sessions)

            # build the classifier
            self.svc, self.scaler, self.selected_features, self.pca, self.n_components, self.idx_rem = self.build_classifier(auto_feature_select = False, preset_features = True)

    # build classifier
    def build_classifier(self, auto_feature_select = True, preset_features = False, preferred_features = None):
        '''
        Args:
            >>> training_session_directory: contains a directory of folders with suite2p/plane0 files to train the model on
        '''

        # sanity check
        assert auto_feature_select != preset_features, "You cannot set auto_feature_select and preset_features as both True or both False"

        # cleanup data
        df_clean, idx_rem = self.cleanup_classifier_data(df_all=self.df_train)

        # Assuming df_all is already created and has the necessary columns
        # Split into features (X) and labels (y)
        X = df_clean.drop(columns=['iscell', 'mouseName'])
        y = df_clean['iscell']

        # preselected features
        if preset_features == True and preferred_features is None:

            # using this feature list, extract from df
            feature_list = ['comp_SNR', 'skewF', 'fitness', 'sd_r', 'corr', 'compact', 'npix_norm']
            print("Using preset feature_list:",feature_list)
            selected_features = feature_list
            X = X[feature_list]

        # get number of components
        n_components = X.shape[1]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Synthetic Minority Oversampling Technique: creates synthetic values by interpolating between data points in
        # order to balance out the training dataset. Note that this isn't used for testing.
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # rescale the data
        print("Rescaling data...")
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_resampled) # fit and transform X_train using mean and std of X-mean
        X_test  = scaler.transform(X_test) # transform X_test using mean and std of X_train

        # use recursive feature elimination to identify the best candidate features for classification
        if auto_feature_select == True:
            print("Automatically detecting which features to use for classification")

            # train a temporary classifier on the initial scaled dataset
            svc_rfe = SVC(kernel='linear', C=1, gamma=0.1, class_weight='balanced')

            # use cross-validation to identify the number of features to use for recursive feature elimination
            scores = []

            for n in range(1, X_train.shape[1] + 1):
                rfe = RFE(estimator=svc_rfe, n_features_to_select=n)
                rfe.fit(X_train, y_train)
                score = cross_val_score(rfe, X_train, y_train, cv=5).mean()
                scores.append(score)
                print(f'Performance at {n} features: {score}')

            # Plot cross-validation scores
            plt.plot(range(1, X_train.shape[1] + 1), scores)    
            plt.axhline(y=np.max(scores), color='r', linestyle='--')
            plt.axvline(x=np.argmax(scores)+1, color='r', linestyle='--')
            plt.xlabel('# Features')
            plt.ylabel('Cross-Validation Score')
            plt.show()

            # Optimal number of features
            optimal_n_features = scores.index(max(scores)) + 1
            print(f'Optimal number of features: {optimal_n_features}')

            # using recursive feature elimination, identify the most relevant features
            rfe = RFE(estimator=svc_rfe, n_features_to_select=optimal_n_features)
            rfe.fit(X_train, y_train)
            selected_features = X.columns[rfe.support_]
            print("Selected features:", selected_features)
            X_train = X_train[:, rfe.support_]
            X_test  = X_test[:, rfe.support_]

        # run PCA to transform the dataset into non-correlated variables
        print("Running PCA...")
        n_components = X_train.shape[1]-1 #n-1 PCs
        pca          = PCA(n_components=n_components)
        X_train      = pca.fit_transform(X_train) # calculate PCs from training data
        X_test       = pca.transform(X_test)      # uses the same PCs from the training data, applied to testing data

        # estimate # of PCs to use based on cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(cumulative_variance)
        plt.xlabel('# Principal Components')
        plt.ylabel('Cum. Variance')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.axvline(x=np.where(cumulative_variance >= 0.95)[0][0], color='r', linestyle='--')
        plt.show()

        # Number of components to retain 95% variance
        n_components = np.where(cumulative_variance >= 0.95)[0][0] + 1
        print(f'# of components to retain 95% variance: {n_components}')

        # PCs to keep
        print("Cleaned up X_train and X_test accordingly...")
        X_train = X_train[:, 0:n_components]
        X_test  = X_test[:, 0:n_components]

        # Initialize the SVC classifier
        svc = SVC(kernel='linear', C=1, gamma=0.1, class_weight='balanced', probability=True)

        # Train the model
        svc.fit(X_train, y_train_resampled) # X_train is the resampled version

        # Get support vectors
        sv = svc.support_vectors_
        sv_labels = svc.dual_coef_.ravel() > 0

        # Make predictions on the test set
        y_pred = svc.predict(X_test)

        # Evaluate the model on the entire test set
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy on entire test set: {accuracy}')

        # Evaluate accuracy on accepted components (true)
        accepted_indices = np.where(y_test == True)[0]
        accuracy_accepted = accuracy_score(y_test.iloc[accepted_indices], y_pred[accepted_indices])
        print(f'Accuracy on accepted components (true): {accuracy_accepted}')

        # performance on rejected components
        rejected_indices = np.where(y_test == False)[0]
        accuracy_rejected = accuracy_score(y_test.iloc[rejected_indices], y_pred[rejected_indices])
        print(f'Accuracy on rejected components (false): {accuracy_rejected}')

        # Evaluate the model
        #print(confusion_matrix(y_test, y_pred))
        print("________________________________________________________________")
        print(classification_report(y_test, y_pred))
        print("________________________________________________________________")

        return svc, scaler, selected_features, pca, n_components, idx_rem

    # Initialize an empty DataFrame
    def gather_classifier_data(self, classifier_sessions):

        # load and organize data
        df_all = pd.DataFrame(); F_all = []
        for sessi in classifier_sessions:
            fpath = sessi
            print("Working on",sessi)

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

            # save F
            F_all.append(F)

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

    # Example of using the trained model on new unseen data
    def predict_cell(self, df_predict, svc, scaler, selected_features, pca, n_components):
        '''
        Using the training svm from build_classifier, predict whether a cell is a cell
        '''

        # TODO check the n_component
        # only run the classifier on cells already deemed "cells"
        #df_predict_cell = df_predict[df_predict['iscell']==True]

        # cleanup data
        df_clean, idx_rem = self.cleanup_classifier_data(df_all = df_predict)

        # Assuming df_all is already created and has the necessary columns
        # Split into features (X) and labels (y)
        X = df_clean.drop(columns=['iscell', 'mouseName'])
        X = X[selected_features]

        # standardize
        X_scaled = scaler.transform(X)
    
        # filter out for features not selected using rfe on the training set
        idx_features = [np.where(X.columns == i)[0][0] for i in selected_features]
        X_filt = X_scaled[:, idx_features]

        # pca
        X_pca = pca.transform(X_filt)
        X_pca_filt = X_pca[:, 0:n_components]

        # binary predictions
        predictions = svc.predict(X_pca_filt)

        # the decision-scores variable represents the distance of each feature from the hyperplane and can be interpreted as a confidence score
        decision_scores = svc.decision_function(X_pca_filt)

        # probability represent the likelihood of a class 
        probabilities = svc.predict_proba(X_pca_filt)

        # replace the removed cells and set them to false
        if len(idx_rem) > 0:
            print(f'Detected removed cells {idx_rem}, replacing with 0.0 or False in classification array')
            final_probabilities = np.empty((len(probabilities) + len(idx_rem), probabilities.shape[1]), dtype=object)
            final_predictions   = np.empty(len(predictions) + len(idx_rem), dtype=object)
            final_scores        = np.empty(len(decision_scores) + len(idx_rem), dtype=object)

            # Initialize pointers for the probabilities and final_probabilities arrays
            prob_idx = 0; final_idx = 0

            # Fill the final_probabilities array with shifting
            for i in range(final_probabilities.shape[0]):
                if final_idx in idx_rem:
                    final_probabilities[final_idx] = 0.0
                    final_scores[final_idx] = 0.0
                    final_predictions[final_idx] = False
                else:
                    final_probabilities[final_idx] = probabilities[prob_idx]
                    final_scores[final_idx] = decision_scores[prob_idx]
                    final_predictions[final_idx] = predictions[prob_idx]
                    prob_idx += 1
                final_idx += 1

            # change to bool
            final_probabilities=final_probabilities.astype(bool)

            # rename
            del predictions, probabilities, decision_scores
            predictions = final_predictions
            probabilities = final_probabilities
            decision_scores = final_scores

        return predictions.astype(bool), probabilities, decision_scores

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
            skF = stats.skew(F-Fneu, axis=1)

            # calculate the median asymmetry of the F signal
            asymmetry = get_asymmetry(F = F, Fneu = Fneu, C = C, fs = 7.5)

            # identify "not cells" that have low compactness and high skew, then send them to "is cell" category
            iscell_out = reject_cell(iscell=iscell_in, stat=stat, compact=compact, skF=skF, asymmetry=asymmetry)

            # rescue - adds too much noise
            #iscell_out = rescue_cell(iscell=iscell_out, stat=stat, skF=skF)

            # find cells near one another amongst the true positives, then if strongly correlated, remove the weaker signal
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
            sio.savemat(os.path.join(i, 'Fall_classified.mat'), mdict = {'F': F, 'Fneu': Fneu, 'iscell': iscell, 'stat': stat, 'C': C, 'S': S, 'ops': ops_matlab, 's2pSpk': spks})

    # classify
    def classify(self, session_path: str):

        # gather variables
        df_predict = self.gather_classifier_data([session_path])

        # test classifier
        predictions, probabilities, decision_scores = self.predict_cell(df_predict = df_predict, svc=self.svc, scaler=self.scaler, selected_features=self.selected_features,
                                                                    pca=self.pca, n_components=self.n_components)


        self.rewrite_data(predict_sessions = [session_path], predictions=predictions, probabilities = probabilities)


# ---------------------------------------- #
# --  HELPER AND STAND ALONE FUNCTIONS  -- #
# ---------------------------------------- #

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

    # rejection time - this actually seems to capture non-asymettrical cells but also
    # really noisy cells because the event peaks are poorly estimated
    asymmetry_cutoff = 0.4 # after finding cases of good cells > .4 but bad <=.4
    rejected_cells = np.where(asymmetry <= asymmetry_cutoff)[0]
    iscell[rejected_cells]=False

    # rescue non-cells - these criterion are largely good
    #rescued_cells = np.where(np.logical_and.reduce([iscell==False, compact <= 1.05, skF > 2.0]))[0]
    #iscell[rescued_cells]=True 

    # if there are <100 pixels in a cell, toss it
    npix = np.array([i['npix'] for i in stat])
    iscell[npix<100]=False

    # if there are <100 pixels in a cell, toss it
    skew = np.array([i['skew'] for i in stat])
    iscell[skew<1.2]=False    

    return iscell

# function to rescue cells
def rescue_cell(iscell, stat, skF):
    '''
    This code rescues candidate false rejection cells based on high skew and low compactness,
    then rejects cells with low asymmetry

    John Stout
    '''
    
    print("Rescuing cells if they have >100 pixels and skew > 2.0")

    # if there are <100 pixels in a cell, toss it
    npix = np.array([i['npix'] for i in stat])
    skew = np.array([i['skew'] for i in stat])

    # iscell
    iscell[(npix > 100) & (skew > 2.0)]=True

    return iscell

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

# get asymmetry
def get_asymmetry(F, Fneu, C, fs = 7.5):
    from scipy.signal import find_peaks
    from scipy.stats import zscore

    # neuropil correct
    Fcor = F-Fneu

    asymmetry = []
    for celli in range(C.shape[0]):

        # get example traces
        c = C[celli]
        f = Fcor[celli]

        # detect event peaks
        #cZ              = zscore(c)
        #idx_peaks, prop = find_peaks(cZ)
        #c_peaks         = cZ[idx_peaks]          # find C trace events that are also peaks
        #c_filt_idx      = np.where(c_peaks > 1)  # find C trace events that are also peaks but also greater than 1std
        #idx_peaks       = idx_peaks[c_filt_idx]  # 
        #c_peaks         = c_peaks[c_filt_idx]    # 
        #F_peaks         = f[idx_peaks] # event peaks

        # get event peaks
        idx_peaks, __, __ = calcium_events(c = c)
        F_peaks = f[idx_peaks]

        # Initialize decay times
        decay_left_times = []
        decay_right_times = []
        cell_asymmetry = []

        # Loop over events and detect decay
        for idx, peak_value in zip(idx_peaks, F_peaks):

            # Define the decay threshold (e.g., 50% of peak value)
            decay_threshold = peak_value * 0.5

            # Search for decay before the peak
            left_indices = np.where(f[:idx] <= decay_threshold)[0]
            if len(left_indices) > 0:
                decay_left = left_indices[-1]
                decay_left_times.append(decay_left)

            # Search for decay after the peak
            right_indices = np.where(f[idx:] <= decay_threshold)[0]
            if len(right_indices) > 0:
                decay_right = idx + right_indices[0]
                decay_right_times.append(decay_right)

            # offset
            right_sided = np.abs((idx-decay_right) / fs)
            left_sided = np.abs((idx-decay_left) / fs)
            cell_asymmetry.append(right_sided-left_sided)

        # take the median of the asymmetry metric
        asymmetry.append(np.median(np.array(cell_asymmetry)))

    # make into numpy
    asymmetry = np.array(asymmetry)
    return asymmetry

    # Plot Fcor signal with detected peaks and decay points
    #plt.close()
    #plt.plot(f, label='Fcor Signal')
    #plt.plot(idx_peaks, F_peaks, 'x', label='Peaks')
    #if decay_left is not None:
    #    plt.axvline(x=decay_left, color='r', linestyle='--', label='Decay Left')
    #if decay_right is not None:
    #    plt.axvline(x=decay_right, color='g', linestyle='--', label='Decay Right')
    #plt.legend()
    #plt.show()

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

# smooth data
def smooth(data, window_len=11, window='hanning'):
    if window_len < 3:
        return data
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len//2-1):-(window_len//2)]

# TODO needs testing
# updated calcium_events code that accounts for overalapping evernts
def calcium_events_new(c, Fc, zscore_threshold=1, fs=7.5):
    """
    Calcium event detection using the C trace output from constrained foopsi OASIS.

    Args:
        c: a single c trace
        Fc: filtered signal
        zscore_threshold: default = 1 std which appears to be pretty good
        fs: sampling frequency, default = 7.5

    Returns:
        idx_peaks_C: peak indices in C
        idx_peaks_F: peak indices in Fc
    """
    import numpy as np
    from scipy.stats import zscore
    from scipy.signal import find_peaks
    UserWarning("This code has not been tested. The old calcium_events_old was used in Python. This code was converted from MATLAB")

    # Compute z-score of the signal
    cZ = zscore(c)

    # Find peaks in the z-scored signal
    idx_peaks_C, _ = find_peaks(cZ)
    pks = cZ[idx_peaks_C]

    # Identify peaks that are within 5s of each other and keep the strongest peak
    time_range = np.linspace(0, len(c) / fs, len(c))
    peak_times = time_range[idx_peaks_C]

    next_flag = False
    while not next_flag:
        # Get difference in peak times
        peak_offset = np.diff(peak_times)

        # Identify peaks that are too close to each other
        double_peak_candidate = np.where(peak_offset < 5)[0]

        if double_peak_candidate.size == 0:
            next_flag = True
        else:
            # Examine each peak and keep the max event
            for evi in range(len(double_peak_candidate)):
                # Get the events from each and identify which events are best (difference between events)
                event_diff = pks[double_peak_candidate[evi]:double_peak_candidate[evi]+2]
                
                # Take the minima and remove it
                event_toss = np.argmin(event_diff)
                
                # Set to NaN so we don't change the size of the array during the loop
                idx_peaks_C[double_peak_candidate[evi] + event_toss] = np.nan
                peak_times[double_peak_candidate[evi] + event_toss] = np.nan
                pks[double_peak_candidate[evi] + event_toss] = np.nan

            # Remove NaNs
            idx_peaks_C = idx_peaks_C[~np.isnan(idx_peaks_C)]
            peak_times = peak_times[~np.isnan(peak_times)]
            pks = pks[~np.isnan(pks)]

    # Search through peak_times for max events in Fc
    Fc_sm = smooth(Fc, window_len=int(np.ceil(fs)*4), window='gaussian')

    # Now correct the peak offset
    idx_peaks_F = []
    for evi in range(len(idx_peaks_C)):
        # Get 5s surrounding idx_peak
        idx_around = np.arange(max(0, idx_peaks_C[evi] - int(fs*5)), min(len(c), idx_peaks_C[evi] + int(fs*5)))

        # Now get data surrounding time point of interest
        temp_F = Fc_sm[idx_around]

        # Get peak of the temp variable
        max_temp_F = np.max(temp_F)
        idx_max_temp_F = np.argmax(temp_F)

        # Reset the idx_peak
        idx_peaks_F.append(idx_around[idx_max_temp_F])

    # Convert idx_peaks_F to numpy array
    idx_peaks_F = np.array(idx_peaks_F)

    # Now filter for z-score threshold
    idx_peaks_C = idx_peaks_C[cZ[idx_peaks_C.astype(int)] > zscore_threshold]
    idx_peaks_F = idx_peaks_F[cZ[idx_peaks_F.astype(int)] > zscore_threshold]

    return idx_peaks_C, idx_peaks_F

