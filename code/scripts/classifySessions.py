# classifier

# load modules
import os; import matplotlib.pyplot as plt; import tifffile as tf
path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)
#from s2pfuns import detrend_signal
from pathlib import Path
import numpy as np
import time
import suite2p
from suite2p.extraction import dcnv
import csv
import rootfun as rf # we can import this if our cwd is local

from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from s2pfuns import read_s2p
from scipy import stats
from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter

from caiman import components_evaluation as comp_eval
from scipy.stats import norm
from scipy import special

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

import pandas as pd

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from suite2p.classification import classify, builtin_classfile
builtin_classfile = builtin_classfile

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
        fitness, erfc, sd_r, md = comp_eval.compute_event_exceptionality(traces=F-Fneu)

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
        pnr_data = pnr(F=F-Fneu, Fneu=Ndet)

        # Get classifier stats
        npix_norm = np.array([i['npix_norm'] for i in stat])
        compact = np.array([i['compact'] for i in stat])
        aspect = np.array([i['aspect_ratio'] for i in stat]) # aspect ratio is how elongated a component is

        # --- Some important filtering and rescuing steps -- #

        # calculate the median asymmetry of the F signal
        asymmetry = event_decay(F = F, Fneu = Fneu, C = C, fs = 7.5)

        # identify "not cells" that have low compactness and high skew, then send them to "is cell" category
        iscell = rescue_and_reject(iscell=iscell, stat=stat, compact=compact, skF=skF, asymmetry=asymmetry)

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
            'compact': compact, # added
            #'aspect_ratio': aspect, # added
            'comp_SNR': comp_SNR,
            'asymmetry': asymmetry # added
        })

        # Append the current session DataFrame to the main DataFrame
        df_all = pd.concat([df_all, df_session], ignore_index=True)

        # save F
        F_all.append(F)

    return df_all

# function to rescue cells
def rescue_and_reject(iscell, stat, compact, skF, asymmetry):
    '''
    This code rescues candidate false rejection cells based on high skew and low compactness,
    then rejects cells with low asymmetry

    John Stout
    '''
    
    print("Rescuing cells with 1) strong skew and removing cells with 1) low asymmetry and 2) low pixel counts")

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

# detect calcium event decay
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
                
            # if the ROI are less than 20 pixels apart between two classified cells, check for event overlap
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

# updated calcium_events code that accounts for overalapping evernts
def calcium_events(c, Fc, zscore_threshold=1, fs=7.5):
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
    Fc_sm = smooth(Fc, window_len=int(ceil(fs)*4), window='gaussian')

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


# Example usage
#c = np.random.randn(1000)
#Fc = np.random.randn(1000)
#idx_peaks_C, idx_peaks_F = calcium_events(c, Fc)
#print("C peaks:", idx_peaks_C)
#print("Fc peaks:", idx_peaks_F)


# function to find calcium event peaks
def calcium_events_old(c, zscore_threshold = 1):
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

    cZ              = zscore(c)
    idx_peaks, prop = find_peaks(cZ)
    cZ_peaks        = cZ[idx_peaks]          # find C trace events that are also peaks
    c_raw_peaks     = c[idx_peaks]
    c_filt_idx      = np.where(cZ_peaks > 1)  # find C trace events that are also peaks but also greater than 1std
    idx_peaks       = idx_peaks[c_filt_idx]  # 
    cZ_peaks        = cZ_peaks[c_filt_idx]    # zscored C peaks
    c_raw_peaks     = c_raw_peaks[c_filt_idx] # C peaks

    return idx_peaks, c_raw_peaks, cZ_peaks

# function to remove nan/inf values
def cleanup_classifier_data(df_all):

    # identify nan or inf values
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    idx_rem = df_all.index[df_all.isna().any(axis=1)]

    for i in idx_rem:
        print(f'Detected and removed NaN at: {i}')

    # remove them
    df_clean = df_all.drop(idx_rem)

    return df_clean, idx_rem

# build classifier
def build_classifier(df_all, auto_feature_select = True, preset_features = False):
    '''
    Build classifier
    '''
    from sklearn.feature_selection import RFE
    from sklearn.metrics import classification_report, confusion_matrix

    # sanity check
    assert auto_feature_select != preset_features, "You cannot set auto_feature_select and preset_features as both True or both False"

    # cleanup data
    df_clean, idx_rem = cleanup_classifier_data(df_all = df_all)

    # Assuming df_all is already created and has the necessary columns
    # Split into features (X) and labels (y)
    X = df_clean.drop(columns=['iscell', 'mouseName'])
    y = df_clean['iscell']

    # preselected features
    if preset_features == True:

        # using this feature list, extract from df
        feature_list = ['comp_SNR', 'skewF', 'fitness', 'sd_r', 'corr', 'compact', 'npix_norm']
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

    return svc, scaler, selected_features, pca, n_components, idx_rem

# Example of using the trained model on new unseen data
def predict_cell(df_predict, svc, scaler, selected_features, pca, n_components):
    '''
    Using the training svm from build_classifier, predict whether a cell is a cell
    '''

    # TODO check the n_component
    # only run the classifier on cells already deemed "cells"
    #df_predict_cell = df_predict[df_predict['iscell']==True]

    # cleanup data
    df_clean, idx_rem = cleanup_classifier_data(df_all = df_predict)

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
def rewrite_data(predict_sessions, predictions, probabilities):
    import scipy.io as sio
    from datetime import datetime

    for i in predict_sessions:

        # read og iscell
        F, Fneu, spks, stat, ops, iscell0, __ = read_s2p(fpath=i)
        C = np.load(os.path.join(i, 'suite2p', 'plane0', 'C.npy'), allow_pickle=True)
        S = np.load(os.path.join(i, 'suite2p', 'plane0', 'S.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(i,'suite2p','plane0','iscell.npy'), allow_pickle=True)

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
        asymmetry = event_decay(F = F, Fneu = Fneu, C = C, fs = 7.5)

        # identify "not cells" that have low compactness and high skew, then send them to "is cell" category
        iscell_out = rescue_and_reject(iscell=iscell_in, stat=stat, compact=compact, skF=skF, asymmetry=asymmetry)

        # find cells near one another amongst the true positives, then if strongly correlated, remove the weaker signal
        iscell_out = reject_overlapping_roi(stat=stat, F=F, Fneu=Fneu, C=C, iscell=iscell_out, fs=7.5)

        # regenerate iscell. Note that we are replacing the iscell with iscell_out because iscell_out is processed probabilities from the classifier
        del iscell
        iscell = np.zeros(iscell_og.shape)
        iscell[iscell_out, 0] = 1.0
        iscell[:, 1] = probabilities[:, 1]

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

# classifier_sessions
training_sessions = [
    #r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L1_SD1_odor_day9_FOV3_optoRec_LBC0_img",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L607T4_SDswitch_day1_noOpto_FOV2_img",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L608_SEDS_day8_FOV1_LBC0_noOpto_img",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L612_SEDS_day3_LBC2_p70_optoRec_FOV1_img",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L613_CD1_odor_day1_optoRec_LBC2_FOV2_p70_img",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L614_CD2_odor_day1_FOV3_LBC2_optoRec_p70_img",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L615_CD_odor_day1_optoRec_FOV1_LBC2_p70_img",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L616_SD1_whisker_day8_optoRec_FOV1_LBC2_img_001",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\T30_SEDS_day25_FOV6_optoRec_LBC2_img_000",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L612_SEDS_day11_updatedParameters",
    r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierBuildSuite2p\L614_SEDS_day2_optoRec_LBC2_p70_FOV3_img"
]

# gather data to train svm
df_train = gather_classifier_data(training_sessions)

# train svm
svc, scaler, selected_features, pca, n_components, idx_rem = build_classifier(df_train, auto_feature_select = False, preset_features = True)

# -- classifier testing -- #
Datafolder = r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects"

# get all subdirs
subdirs = rf.list_all_subdirs(phile_name = Datafolder)
predict_sessions = [i for i in subdirs if 'plane0' in i] # filter out for suite2p
predict_sessions = [os.path.split(os.path.split(i)[0])[0] for i in predict_sessions] # get root

for sessi in predict_sessions:

    # gather variables
    df_predict = gather_classifier_data([sessi])

    # test classifier
    predictions, probabilities, decision_scores = predict_cell(df_predict = df_predict, svc=svc, scaler=scaler, selected_features=selected_features,
                                                                pca=pca, n_components=n_components)

    rewrite_data(predict_sessions = [sessi], predictions=predictions, probabilities = probabilities)
