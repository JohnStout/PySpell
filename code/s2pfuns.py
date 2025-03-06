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
import deconvolution as dc
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter

# -- A handful of these functio ns might better serve as an object -- #
# to quickly run suite2p
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

# -- postprocessing code -- #
# use this code as such: 
#       postProcess(s2ppath = r"path/to/your/folder").cleanup_raw_traces
# TODO: build a mechanism to rename and replace the F and spks variables with C and S for visualization purposes
#s2ppath = r"E:\L6 Experiments\L612\FOV1\SEDS_day11_LBC2_p70_FOV1\SEDS_day11_LBC2_p70_FOV1_img"
#s2ppath = r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L612_SEDS_day11_updatedParameters\suite2p_r1\plane0"
class postProcess():

    def __init__(self, s2ppath: str):

        # read suite2p results
        F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath = s2ppath)

        # store path
        self.s2ppath = s2ppath

        # store suite2p data
        self.F = F; self.Fneu = Fneu; self.spks = spks; self.stat = stat
        self.ops = ops; self.iscell = iscell; self.blF = blF

    def roi_cleanup():
        '''
        On plane3D signals, this code will be designed to 
            1) identify overlapping ROI in each plane
                Lets say ROIA and ROIB are present in all planes, overlapping 70%
            2) Use CaImAns intersection over union technique to co-label cells in multiple planes
            3) identify which plane the ROI should belong based on its peak activity
                If ROIA is maximally active in plane0, then remove it from plane1 and 2. Same for ROIB
                Preserve the activity traces by pushing them from plane1 and plane2 to plane0 via max projection over the ROI
                    Likewise, we will need to subtract the space of ROIA that is being explained by ROIB
            3) Use grid interpolation
                Interpolate the removed component by identifying times of its peak activity and NaN remove, then interpolate
                
        '''
        pass


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


    # -- methods to rescue and reject cells -- #
    def rescue_and_reject(iscell, compact, skF, asymmetry):
        '''
        This code rescues candidate false rejection cells based on high skew and low compactness,
        then rejects cells with low asymmetry

        John Stout
        '''
        
        # rescue non-cells - these criterion are largely good
        rescued_cells = np.where(np.logical_and.reduce([iscell==False, compact <= 1.05, skF > 2.0]))[0]

        # send forward
        iscell[rescued_cells]=True

        # rejection time - this actually seems to capture non-asymettrical cells but also
        # really noisy cells because the event peaks are poorly estimated
        asymmetry_cutoff = 0.4 # after finding cases of good cells > .4 but bad <=.4
        rejected_cells = np.where(asymmetry <= asymmetry_cutoff)[0]
        iscell[rejected_cells]=False

        return iscell

    # method to detect event half-width
    def event_decay(F, Fneu, C, fs = 7.5):
        from scipy.signal import find_peaks
        from scipy.stats import zscore

        # neuropil correct
        Fcor = F-Fneu

        assymmetry = []
        for celli in range(C.shape[0]):

            # get example traces
            c = C[celli]
            f = Fcor[celli]

            # detect event peaks
            cZ              = zscore(c)
            idx_peaks, prop = find_peaks(cZ)
            c_peaks         = cZ[idx_peaks]          # find C trace events that are also peaks
            c_filt_idx      = np.where(c_peaks > 1)  # find C trace events that are also peaks but also greater than 1std
            idx_peaks       = idx_peaks[c_filt_idx]  # 
            c_peaks         = c_peaks[c_filt_idx]    # 
            F_peaks         = f[idx_peaks] # event peaks

            # Initialize decay times
            decay_left_times = []
            decay_right_times = []
            cell_assymmetry = []

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
                cell_assymmetry.append(right_sided-left_sided)

            # take the median of the assymmetry metric
            assymmetry.append(np.median(np.array(cell_assymmetry)))

        # make into numpy
        assymmetry = np.array(assymmetry)
        return assymmetry

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






        def event_width(self, c):
            pass

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

    # method for ROI classification
    def classify_roi(self):
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from caimanfuns import compute_event_exceptionality

        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score
        from scipy.ndimage import gaussian_filter, gaussian_filter1d
        from scipy import stats
        from scipy.stats import norm
        from scipy import special

        # pnr
        def pnr(F, Fneu):

            # Calculate 95th percentiles
            pk95F = np.percentile(np.sort(np.abs(F), axis=1), 95, axis=1)
            pk95N = np.percentile(np.sort(np.abs(Fneu), axis=1), 95, axis=1)

            # Calculate PNR (peak noise ratio)
            pnr = pk95F / pk95N

            print(f'PNR: {pnr}')
            return pnr

        # Initialize an empty DataFrame
        def gather_classifier_data(classifier_sessions: list):

            df_all = pd.DataFrame(); F_all = []
            for sessi in classifier_sessions:
                fpath = sessi
                print("Working on",sessi)

                # squirrel mouse name
                mouseName = os.path.split(fpath)[-1].split('_')[0]

                # load data
                F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)
                C = np.load(os.path.join(fpath, 'suite2p', 'plane0', 'C.npy'), allow_pickle=True)
                S = np.load(os.path.join(fpath, 'suite2p', 'plane0', 'S.npy'), allow_pickle=True)

                # make mouseName variable 
                mouseName_array = np.array([mouseName] * F.shape[0])

                # Detrend F and neuropil
                #Fdet = detrend_signal(F=F)
                #Ndet = detrend_signal(F=Fneu)

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
                skF = stats.skew(blF, axis=1)
                skN = stats.skew(Ndet, axis=1)

                # SNR
                varF = np.var(blF, axis=1)
                varN = np.var(Ndet, axis=1)
                snr = varF / varN

                # Correlate F to C
                smoothF = gaussian_filter1d(blF, sigma=7.5*2, axis=1)
                correlation = np.array([np.corrcoef(smoothF[i], C[i])[0, 1] for i in range(smoothF.shape[0])])

                # Fitness, traces == C
                fitness, erfc, sd_r, md = compute_event_exceptionality(traces=blF)

                # Number of timesteps to consider when testing new neuron candidates
                min_SNR = 2.5
                min_SNR_reject = 0.5
                decay_time = 0.7
                frate = 7.5
                N_samples = np.ceil(frate * decay_time).astype(int)

                # Inclusion probability of noise transient
                thresh_fitness_raw = special.log_ndtr(-min_SNR) * N_samples

                # Threshold on time variability
                fitness_min = special.log_ndtr(-min_SNR) * N_samples

                # Components with SNR lower than 0.5 will be rejected
                thresh_fitness_raw_reject = special.log_ndtr(-min_SNR_reject) * N_samples
                comp_SNR = -norm.ppf(np.exp(fitness / N_samples))

                # PNR
                pnr_data = pnr(F=blF, Fneu=Ndet)

                # Get classifier stats
                npix_norm = np.array([i['npix_norm'] for i in stat])
                compact = np.array([i['compact'] for i in stat])
                aspect = np.array([i['aspect_ratio'] for i in stat]) # aspect ratio is how elongated a component is

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
                    'compact': compact, # added
                    'aspect_ratio': aspect, # added
                    'comp_SNR': comp_SNR
                })

                # Append the current session DataFrame to the main DataFrame
                df_all = pd.concat([df_all, df_session], ignore_index=True)

                # save F
                F_all.append(F)

            return df_all

        # function to remove nan/inf values
        def cleanup_classifier_data(df_all):

            # identify nan or inf values
            df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
            idx_rem = df_all.index[df_all.isna().any(axis=1)]

            for i in idx_rem:
                print(f'Detected and removed NaN at: {i}')

            # remove them
            df_clean = df_all.drop(idx_rem)

            return df_clean

        # build classifier
        def build_classifier(df_all, auto_feature_select = True, preset_features = False):
            '''
            Build classifier
            '''
            from sklearn.feature_selection import RFE
            from sklearn.metrics import classification_report, confusion_matrix
            import matplotlib.pyplot as plt

            # sanity check
            assert auto_feature_select != preset_features, "You cannot set auto_feature_select and preset_features as both True or both False"

            # cleanup data
            df_clean = cleanup_classifier_data(df_all = df_all)

            # Assuming df_all is already created and has the necessary columns
            # Split into features (X) and labels (y)
            X = df_clean.drop(columns=['iscell', 'mouseName'])
            y = df_clean['iscell']

            # preselected features
            if preset_features == True:

                # using this feature list, extract from df
                feature_list = ['comp_SNR', 'skewF', 'fitness', 'sd_r', 'corr', 'compact', 'npix_norm', 'aspect_ratio']
                print("Using preset feature_list:",feature_list)
                selected_features = feature_list
                X = X[feature_list]

            # get number of components
            n_components = X.shape[1]

            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # rescale the data
            print("Rescaling data...")
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train) # fit and transform X_train using mean and std of X-mean
            X_test  = scaler.transform(X_test) # transform X_test using mean and std of X_train

            # Evaluate RFE for a range of n_features_to_select values
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
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.axhline(y=0.95, color='r', linestyle='--')
            plt.axvline(x=np.where(cumulative_variance >= 0.95)[0][0], color='r', linestyle='--')
            plt.title('Explained Variance by Principal Components')
            plt.show()

            # Number of components to retain 95% variance
            n_components = np.where(cumulative_variance >= 0.95)[0][0] + 1
            print(f'Number of components to retain 95% variance: {n_components}')

            # PCs to keep
            print("Cleaned up X_train and X_test accordingly...")
            X_train = X_train[:, 0:n_components]
            X_test  = X_test[:, 0:n_components]

            # Initialize the SVC classifier
            svc = SVC(kernel='linear', C=1, gamma=0.1, class_weight='balanced', probability=True)

            # Train the model
            svc.fit(X_train, y_train)

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

            return svc, scaler, selected_features, pca, n_components

        # todo:
        def build_activity_classifier():
            pass

        def build_anatomy_classifier():
            pass

        # Example of using the trained model on new unseen data
        def predict_cell(df_predict, svc, scaler, selected_features, pca, n_components):
            '''
            Using the training svm from build_classifier, predict whether a cell is a cell
            '''

            # TODO check the n_component

            # cleanup data
            df_clean = cleanup_classifier_data(df_all = df_predict)

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

            return predictions, probabilities, decision_scores

        # function that calls in the iscell variable and rewrites it according to predict_cell predictions
        def rewrite_iscell(predict_sessions, predictions, probabilities):
            for i in predict_sessions:

                # read og iscell
                iscell = np.load(os.path.join(i,'suite2p','plane0','iscell.npy'), allow_pickle=True)
                
                # rewrite
                iscell_og = iscell.copy()
                del iscell

                # rewrite iscell
                iscell = np.zeros(iscell_og.shape)
                iscell[predictions, 0] = 1.0
                iscell[:, 1] = probabilities[:, 1]

                # save
                np.save(os.path.join(i, 'suite2p', 'plane0', 'iscell.npy'), iscell, allow_pickle=True)
                np.save(os.path.join(i, 'suite2p', 'plane0', 'iscell_og.npy'), iscell_og, allow_pickle=True)

            pass

        # overlap remove
        def overlap_trash(predicted_session):

            '''
            Use the correlation between C variables and overlap variables to identify candidate cells that should be tossed or merged.

            Rather than merging, keep the cell with the strongest skew and most compactness

            
            '''
            from scipy.stats import pearsonr

            # predicted session
            predicted_session = sessi

            # load the stat and C variables
            F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=predicted_session)
            C = np.load(os.path.join(predicted_session, 'suite2p', 'plane0', 'C.npy'), allow_pickle=True)
            S = np.load(os.path.join(predicted_session, 'suite2p', 'plane0', 'S.npy'), allow_pickle=True)

            # only include stat that are cells
            stat = stat[iscell==True]

            # search for cells with overlap
            cells = len(stat)
            for i in range(cells):

                # y and x pixels of comparative roi
                ypix, xpix = stat[i]['ypix'], stat[i]['xpix']
                pixels = set(zip(xpix, ypix))

                # get signal
                c_cell = C[i,:]

                # loop over cells, but ignore the current cell
                for ii in range(i + 1, cells):

                    # calculate the percentage of overlap
                    ypix_comp, xpix_comp = stat[ii]['ypix'], stat[ii]['xpix']

                    # get signal
                    c_comp = C[ii,:]

                    # comparative pixels
                    pixels_comp = set(zip(xpix_comp, ypix_comp))

                    # identify overlapping pixels
                    overlap = len(pixels & pixels_comp)

                    # calculate percent overlap using the smaller roi
                    p_overlap = overlap / min(len(pixels), len(pixels_comp))
                        
                    # if the p_overlap > 0.5, check for strong signal correlation
                    if p_overlap > .1:

                        # check for signal correlation
                        r, p = pearsonr(x = c_cell, y = c_comp)

                        # calculate r^2
                        r2 = r ** 2
                        print(f'r2={r2}')

                        if r2 >= 0.7:
                            print("Strongly correlated cell detected")
                            input()
                        
            pass

        # classifier_sessions
        # TODO: FIX THIS
        training_sessions = [
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L1_SD1_odor_day9_FOV3_optoRec_LBC0_img",
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L607T4_SDswitch_day1_noOpto_FOV2_img",
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L608_SEDS_day8_FOV1_LBC0_noOpto_img",
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L612_SEDS_day3_LBC2_p70_optoRec_FOV1_img",
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L613_CD1_odor_day1_optoRec_LBC2_FOV2_p70_img",
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L614_CD2_odor_day1_FOV3_LBC2_optoRec_p70_img",
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L615_CD_odor_day1_optoRec_FOV1_LBC2_p70_img",
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L616_SD1_whisker_day8_optoRec_FOV1_LBC2_img_001",
            r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\T30_SEDS_day25_FOV6_optoRec_LBC2_img_000"
        ]

        # gather data to train svm
        df_train = gather_classifier_data(training_sessions)

        # train svm
        svc, scaler, selected_features, pca, n_components = build_classifier(df_train, auto_feature_select = False, preset_features = True)

        # gather variables
        df_predict = gather_classifier_data([self.s2ppath])

        # test classifier
        predictions, probabilities, decision_scores = predict_cell(df_predict = df_predict, svc=svc, scaler=scaler, selected_features=selected_features,
                                                                    pca=pca, n_components=n_components)

        rewrite_iscell(predict_sessions = [self.s2ppath], predictions=predictions, probabilities = probabilities)

    # method to merge ROI
    def automerge_roi(self):

        '''
        Automatically merge cells if they are 70% overlapping and exhibiting > 0.9 temporal correlation

        '''

        # get ROI
        from scipy.stats import pearsonr

        # search for cells with overlap
        stat = self.stat
        cells = len(stat)
        for i in range(cells):

            # loop over cells, but ignore the current cell
            for ii in range(i + 1, cells):

                # get distance between centriods
                med_dist = (((stat[ii]['med'][0]-stat[i]['med'][0]) ** 2) + ((stat[ii]['med'][1]-stat[i]['med'][1]) ** 2)) ** 0.5
                    
                # if the ROI are less than 20 pixels apart, check their temporal correlation
                if med_dist < 20:

                    # check for signal correlation
                    r, p = pearsonr(x = self.F[i,:], y = self.F[ii,:])
                    if r >= 0.5:
                        print("Strongly correlated cell detected")
                        # merge ROI

        def merge_activity_masks(self):
            print("merging activity... this may take some time")
            i0 = int(1 - self.iscell[parent.ichosen])
            ypix = np.zeros((0,), np.int32)
            xpix = np.zeros((0,), np.int32)
            lam = np.zeros((0,), np.float32)
            footprints = np.array([])
            F = np.zeros((0, parent.Fcell.shape[1]), np.float32)
            Fneu = np.zeros((0, parent.Fcell.shape[1]), np.float32)
            if parent.hasred:
                F_chan2 = np.zeros((0, parent.Fcell.shape[1]), np.float32)
                Fneu_chan2 = np.zeros((0, parent.Fcell.shape[1]), np.float32)
                if not hasattr(parent, "F_chan2"):
                    parent.F_chan2 = np.load(os.path.join(parent.basename, "F_chan2.npy"))
                    parent.Fneu_chan2 = np.load(os.path.join(parent.basename, "Fneu_chan2.npy"))

            probcell = []
            probredcell = []
            merged_cells = []
            remove_merged = []
            for n in np.array(parent.imerge):
                if len(parent.stat[n]["imerge"]) > 0:
                    remove_merged.append(n)
                    for k in parent.stat[n]["imerge"]:
                        merged_cells.append(k)
                else:
                    merged_cells.append(n)
            merged_cells = np.unique(np.array(merged_cells))

            for n in merged_cells:
                ypix = np.append(ypix, parent.stat[n]["ypix"])
                xpix = np.append(xpix, parent.stat[n]["xpix"])
                lam = np.append(lam, parent.stat[n]["lam"])
                footprints = np.append(footprints, parent.stat[n]["footprint"])
                F = np.append(F, parent.Fcell[n, :][np.newaxis, :], axis=0)
                Fneu = np.append(Fneu, parent.Fneu[n, :][np.newaxis, :], axis=0)
                if parent.hasred:
                    F_chan2 = np.append(F_chan2, parent.F_chan2[n, :][np.newaxis, :], axis=0)
                    Fneu_chan2 = np.append(Fneu_chan2, parent.Fneu_chan2[n, :][np.newaxis, :],
                                        axis=0)
                probcell.append(parent.probcell[n])
                probredcell.append(parent.probredcell[n])

            probcell = np.array(probcell)
            probredcell = np.array(probredcell)
            pmean = probcell.mean()
            prmean = probredcell.mean()

            # remove overlaps
            ipix = np.concatenate((ypix[:, np.newaxis], xpix[:, np.newaxis]), axis=1)
            _, goodi = np.unique(ipix, return_index=True, axis=0)
            ypix = ypix[goodi]
            xpix = xpix[goodi]
            lam = lam[goodi]

            ### compute statistics of merges
            stat0 = {}
            stat0["imerge"] = merged_cells
            if "iplane" in parent.stat[merged_cells[0]]:
                stat0["iplane"] = parent.stat[merged_cells[0]]["iplane"]
            stat0["ypix"] = ypix
            stat0["xpix"] = xpix
            stat0["med"] = median_pix(ypix, xpix)
            stat0["lam"] = lam / lam.sum()

            if "aspect" in parent.ops:
                d0 = np.array([int(parent.ops["aspect"] * 10), 10])
            else:
                d0 = parent.ops["diameter"]
                if isinstance(d0, int):
                    d0 = [d0, d0]

            # red prob
            stat0["chan2_prob"] = -1
            # inmerge
            stat0["inmerge"] = -1

            ### compute activity of merged cells
            F = F.mean(axis=0)
            Fneu = Fneu.mean(axis=0)
            if parent.hasred:
                F_chan2 = F_chan2.mean(axis=0)
                Fneu_chan2 = Fneu_chan2.mean(axis=0)
            dF = F - parent.ops["neucoeff"] * Fneu
            # activity stats
            stat0["skew"] = stats.skew(dF)
            stat0["std"] = dF.std()

            spks = oasis(F=dF[np.newaxis, :], batch_size=parent.ops["batch_size"],
                        tau=parent.ops["tau"], fs=parent.ops["fs"])

            ### remove previously merged cell from FOV (do not replace)
            for k in remove_merged:
                masks.remove_roi(parent, k, i0)
                np.delete(parent.stat, k, 0)
                np.delete(parent.Fcell, k, 0)
                np.delete(parent.Fneu, k, 0)
                np.delete(parent.F_chan2, k, 0)
                np.delete(parent.Fneu_chan2, k, 0)
                np.delete(parent.Spks, k, 0)
                np.delete(parent.iscell, k, 0)
                np.delete(parent.probcell, k, 0)
                np.delete(parent.probredcell, k, 0)
                np.delete(parent.redcell, k, 0)
                np.delete(parent.notmerged, k, 0)

            # add cell to structs
            parent.stat = np.concatenate((parent.stat, np.array([stat0])), axis=0)
            parent.stat = roi_stats(parent.stat, parent.Ly, parent.Lx,
                                    aspect=parent.ops.get("aspect", None),
                                    diameter=parent.ops.get("diameter", None),
                                    do_crop=parent.ops.get("soma_crop", 1))
            parent.stat[-1]["lam"] = parent.stat[-1]["lam"] * merged_cells.size
            parent.Fcell = np.concatenate((parent.Fcell, F[np.newaxis, :]), axis=0)
            parent.Fneu = np.concatenate((parent.Fneu, Fneu[np.newaxis, :]), axis=0)
            if parent.hasred:
                parent.F_chan2 = np.concatenate((parent.F_chan2, F_chan2[np.newaxis, :]),
                                                axis=0)
                parent.Fneu_chan2 = np.concatenate(
                    (parent.Fneu_chan2, Fneu_chan2[np.newaxis, :]), axis=0)
            parent.Spks = np.concatenate((parent.Spks, spks), axis=0)
            iscell = np.array([parent.iscell[parent.ichosen]], dtype=bool)
            parent.iscell = np.concatenate((parent.iscell, iscell), axis=0)
            parent.probcell = np.append(parent.probcell, pmean)
            parent.probredcell = np.append(parent.probredcell, -1)
            parent.redcell = np.append(parent.redcell, False)
            parent.notmerged = np.append(parent.notmerged, False)

            ### for GUI drawing
            ycirc, xcirc = utils.circle(parent.stat[-1]["med"], parent.stat[-1]["radius"])
            goodi = ((ycirc >= 0) & (xcirc >= 0) & (ycirc < parent.ops["Ly"]) &
                    (xcirc < parent.ops["Lx"]))
            parent.stat[-1]["ycirc"] = ycirc[goodi]
            parent.stat[-1]["xcirc"] = xcirc[goodi]

            # * add colors *
            masks.make_colors(parent)
            # recompute binned F
            parent.mode_change(parent.activityMode)

            for n in merged_cells:
                parent.stat[n]["inmerge"] = len(parent.stat) - 1
                masks.remove_roi(parent, n, i0)
            masks.add_roi(parent, len(parent.stat) - 1, i0)
            masks.redraw_masks(parent, ypix, xpix)


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

# dysfunctional
#fpath = r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L1_F_LeftPFC_L6Chr_PFCgcamp6f_L6PAN\SD1_odor_day9_FOV3_optoRec_LBC0_img"
def automatic_cell_sorter(fpath: str):
    '''
    This function is going to use automatic algorithms that take into account
    the width, the SNR, and number of peak events to further classify suite2p cells
    as cells or not cells.

    Maybe we could use various features to train an algorithm


    Criterion: decay function x symmetry of calcium event x calcium event peak >2x the noise peak
    
    '''
    import matplotlib.pyplot as plt
    import matplotlib
    
    # suite2p outputs
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)

    # idx for cells
    idxCell   = np.where(iscell==True)[0]
    idxNoCell = np.where(iscell==False)[0]

    # zscore the signals
    blFz = stats.zscore(blF, axis=1)

    # threshold out events as 1std from mean
    caEvents = np.where(blFz > 2)

    #%matplotlib widget
    #matplotlib.use('Agg')
    fig, ax = plt.subplots(nrows=10,ncols=2)
    for i in range(9):
        ax[i,0].plot(blFz[idxCell[i]], color='b', linewidth=0.5)
        ax[i,1].plot(blFz[idxNoCell[i]], color='r', linewidth=0.5)
    plt.plot(blFz[0])
    plt.show()

    # calculate signal:noise ratio on the baseline corrected F
    

    # compute peak-to-noise ratio
    #data_filtered -= data_filtered.mean(axis=0)
    #data_max = np.max(data_filtered, axis=0)
    #data_std = get_noise_fft(data_filtered.T, noise_method='mean')[0].T
    #pnr = np.divide(data_max, data_std)
    #pnr[pnr < 0] = 0   



    pass


# --- grosmark method --- #

# 1) motion correct movie
# 2) correlation coefficients of individual frames to the avg motion correted frame is calculated
# 3) low correlation frames are excluded and censored

# 1) Align sessions using a cross-correlation maximizing two dimensional affine transform, then concatenate
# 2) Only pixels observed across all imaging sessions included

# 1) run the concatenated film through suite2p
'''
class preProcess():
    def __init__(self):
        pass

    # fast_suite2p above motion corrects using suite2ps algorithms. This code
    # is to ONLY motion correct, but not run suite2p in case the user wants to 
    def motion_correct(self, imgpath: str, savepath: str = '', gcamp: str ='6f', alt_ops = None):
        """
        This code motion corrects and saves your img.tif file

        Args:
            >>> datapath: directory, including the .tif file extension
            >>> savepath: path to save your data
            >>> gcamp: default is 6f. 6f and 8f are interchangeable. 8s and 8m have different tau constants.
            >>> zoom_factor: currently dysfunctional, leave alone
            >>> alt_ops: None. Allows user to provide their own ops file.

        John Stout
        """

        # load data lazily
        images = tf.memmap(imgpath, mode="r")

        # movies and associated frame rates
        root_path = os.path.split(imgpath)[0]
        movie_name = os.path.split(imgpath)[1]

        # get metadata
        root_contents = os.listdir(root_path)
        metadata_file = [i for i in root_contents if '.xml' in i and 'experiment' in i.lower()][0]
        metadata_path = os.path.join(root_path,metadata_file)
        file = xmltodict.parse(open(metadata_path,"r").read()) # .xml file

        # default ops
        if alt_ops is None:

            # define frame rate based on metadata
            fr = float(file['ThorImageExperiment']['LSM']['@frameRate'])
            print("Frame rate of",fr,'changed to',fr/4)
            fr = fr/4

            # get default suite2p inputs
            ops = suite2p.default_ops()
            ops['fs']=fr

            # The default is actually 1.0, but I heard Carsen Stringer mention using 0.7 is using fast gcamp
            if '6f' in gcamp or '8f' in gcamp: # check the 8f
                ops['tau'] = 0.7 # gcampe6f
            elif '6m' in gcamp:
                ops['tau'] = 1.0
            elif '6s' in gcamp:
                ops['tau'] = 1.3

            # if the shape of your images data is > 3, then you have a z-plane
            if len(images.shape) > 3 and len(images.shape) < 5:
                print("z-plane detected. If this is not true, stop and troubleshoot")
                ops['nplanes']=images.shape[-1]
            else:
                ops['nplanes']=1
        
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

        # set roidetect to false so that you only get the motion corrected video as an output
        ops['roidetect'] = False

        # save mot corrected video
        ops['reg_tif'] = True

        # running this will only save the motion corrected video
        output_ops = suite2p.run_s2p(ops=ops, db=db)

    # grosmark identifies poor motion corrected frames
    def badMotionCorrect():
        pass

'''