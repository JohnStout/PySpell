# Go through code and snake_case the functions, camelCase the methods

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

# these are for caiman-based deconvolution
import deconvolution as dc
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter

# -- A handful of these functions might better serve as an object -- #
# to quickly run suite2p
def fast_suite2p(imgpath: str, savepath: str = '', gcamp: str ='6f', zoom_factor: float = 2.0, alt_ops = None, wipe_and_replace: bool = False):
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

        # define frame rate based on metadata
        fr = float(file['ThorImageExperiment']['LSM']['@frameRate'])
        print("Frame rate of",fr,'changed to',fr/4)
        fr = fr/4

        # get default suite2p inputs
        ops = suite2p.default_ops()
        ops['fs'] = fr

        # added on 12/5/2024
        #ops['use_builtin_classifier'] = True

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

        # if the shape of your images data is > 3, then you have a z-plane
        if len(images.shape) > 3 and len(images.shape) < 5:
            print("z-plane detected. If this is not true, stop and troubleshoot")
            ops['nplanes']=images.shape[-1]
        else:
            ops['nplanes']=1
            
        # save out the NWB file
        ops['save_NWB']=True # set to false for now

        # Code that adjust the various suite2p parameters to account for your zooming during recording
        # zoom factor is set to 2.0 based on fastZCapture script
        if zoom_factor != 2.0:
            pass
        if zoom_factor == 1.0:
            ops['denoise'] = True
       
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
class postProcess():

    def __init__(self, s2ppath: str):

        # read suite2p results
        F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath = s2ppath)

        # store path
        self.s2ppath = s2ppath

        # store suite2p data
        self.F = F; self.Fneu = Fneu; self.spks = spks; self.stat = stat
        self.ops = ops; self.iscell = iscell; self.blF = blF

    def cleanup_raw_traces(self, run_parallel: bool = False, replace_rename: bool = False):
        import concurrent.futures

        '''
        John put this code together based on Tim's MATLAB code and provided an parallelized option 
        thanks to copilot

        Last edit 10/19/2024

        Args: 
            >>> run_parallel: set to False if you want to run iteratively.
            >>> replace_rename: preset to False, set to true if you want to wipe and replace
            >>> suite2p_detrend: set to True, alternative approach is to use sgolay filter and subtract from F, but this 
        '''

        if run_parallel == True:
            print("run_parallel set to True, running cleanup_raw_traces using parallel processing...")
            def process_cell(index, F, ops, Fneu):

                # neuropil corrected f
                f = F - ops['neucoeff'] * Fneu

                # identify candidate outlier events (noise)
                ttimes = np.where(f > np.median(f) + 3 * median_abs_deviation(f))[0]
                f2 = f.copy()
                f2[ttimes] = np.nan  # nan out the events

                # interpolate candidate noise events
                f2 = np.interp(np.arange(len(f2)), np.arange(len(f2))[~np.isnan(f2)], f2[~np.isnan(f2)])

                # denoised
                f3 = savgol_filter(f2, 1001, 2)
                f4 = f - f3

                # noise constrained deconvolution using default parameters from matlab
                c, bl, c1, g, sn, sp, lam = dc.constrained_foopsi(f4, p=2, method='cvx', bas_nonneg=True,
                                                                noise_range=[0.25, 0.5], noise_method='logmexp',
                                                                lags=2, fudge_factor=1)
                return index, c, sp

            # Assuming self.F is defined and dc.constrained_foopsi is available
            def parallel_cleanup(self, threads: bool = False):
                total_cells = self.F.shape[0]
                results = [None] * total_cells  # Pre-allocate list to store results

                if threads is True:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                        futures = {executor.submit(process_cell, x, self.F[x, :]): x for x in range(total_cells)}
                        
                        for future in concurrent.futures.as_completed(futures):
                            index, c, sp = future.result()
                            results[index] = (c, sp)  # Store the results in the correct order
                            print(f"{(sum(1 for result in results if result is not None) / total_cells) * 100:.2f}% Completed")
                else:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                        futures = {executor.submit(process_cell, x, self.F[x, :]): x for x in range(total_cells)}
                        
                        for future in concurrent.futures.as_completed(futures):
                            index, c, sp = future.result()
                            results[index] = (c, sp)
                            print(f"{(sum(1 for result in results if result is not None) / total_cells) * 100:.2f}% Completed")

                C, S = zip(*results)  # Unpack results
                C = np.array(C)
                S = np.array(S)
                
                # Ensure clean up of resources
                executor.shutdown(wait=True)

                return C, S

            # Example usage
            process_start = time.process_time()
            C, S = parallel_cleanup(self)
            print("Time to cleanup raw traces:",(time.process_time() - process_start)/60,"min")
            #print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

        # else if you do not want to run in parallel
        else:
            print("runParallel set to False. Iterating through suite2p ROIs...")

            # set empty arrays
            C = []
            S = []
            total_cells = self.F.shape[0]

            # loop over each cell and run the cleanup code
            process_start = time.process_time()
            for x in range(self.F.shape[0]):
                print(f'denoising / deconvolving cell {x}')

                f = []; f2 = []; f3 = []; f4 = []

                # neuropil corrected f, this is important for background trend removal
                f = self.F[x,:] - self.ops['neucoeff'] * self.Fneu[x,:]

                # identify candidate outlier events (signal)
                ttimes = np.where(f > np.median(f) + 3 * median_abs_deviation(f))[0]
                f2 = f.copy()
                f2[ttimes] = np.nan # nan out the events
                
                # interpolate candidate signal events to estimate noise
                f2 = np.interp(np.arange(len(f2)), np.arange(len(f2))[~np.isnan(f2)], f2[~np.isnan(f2)])
                
                # subtract the underlying trend (detrend) from the noise-reduced signal
                f3 = savgol_filter(f2, 1001, 2)
                f4 = f - f3

                # parallel computing
                try:
                    noise_range = [0.25, 0.5]
                    deconv_method = 'cvx' # was cvx
                    c, bl, c1, g, sn, sp, lam = dc.constrained_foopsi(f4, p = 2, method = deconv_method, bas_nonneg = True,
                                                                noise_range = noise_range, noise_method = 'logmexp',
                                                                lags = 2, fudge_factor = 1)
                except:
                    print("Failed to run constrained foopsi, likely division by zero")
                    c  = np.zeros(shape = f4.shape)
                    sp = np.zeros(shape = f4.shape)
                C.append(c)
                S.append(sp)
                
                # report on progress
                progress = (x + 1) / total_cells * 100
                print(f"{progress:.2f}% Completed")
            
            # convert to numpy
            C = np.array(C) # denoised flourescence
            S = np.array(S) # deconvolved spiking

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


# -- restructuring-based code -- #

# method to detrend your flourescent signal
def detrend_signal(fpath: str):
    '''
    Detrends your input signal, F by:
        1) Subtracting the neuropil signal
        2) Identifying calcium events as 3mad, then setting those values to NaN
        3) Interpolating NaN values to obtain a "baseline" or "non-event" signal
        4) Subtracting the "baseline" or "non-event" signal from the neuropil corrected F to further detrend local activity

    Args: 
        >>> fpath: path to your suite2p data or path to the folder with suite2p data
    
    Returns:
        >>> detF: detrended F
    '''

    # read in suite2p
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath = fpath)

    # detrend signal
    f = []; f2 = []; f2_store = np.zeros(shape=F.shape); 
    f3 = np.zeros(shape=F.shape); detF = np.zeros(shape=F.shape)
    for x in range(F.shape[0]):
        print(f'denoising / deconvolving cell {x}')

        # start by subtracting the neuropil from F
        f = []
        f = F - ops['neucoeff'] * Fneu

        # identify candidate outlier events (signal)
        ttimes = np.where(f > np.median(f) + 3 * median_abs_deviation(f))[0]
        f2 = f.copy()
        f2[ttimes] = np.nan # nan out the events
        
        # interpolate candidate noise events
        f2 = np.interp(np.arange(len(f2)), np.arange(len(f2))[~np.isnan(f2)], f2[~np.isnan(f2)])
        f2_store = f2

        # secondary denoising
        f3 = savgol_filter(f2, 1001, 2)
        detF[x, :] = f - f3

    return detF


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
