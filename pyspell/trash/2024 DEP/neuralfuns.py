import numpy as np


def calcium_events(c, Fc, zscore_threshold=6, fs=7.5, close_events='keep', detrend_data=True, plot_progress = False, detect_method='scipy'):
    """
    Identify calcium event peaks in a signal with optional detrending and event clustering.
    Includes diagnostic plots for peak detection and alignment.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    from math import ceil
    from scipy import stats
    from scipy.signal import find_peaks, peak_widths

    # -- detrend if requested --
    print("Detecting calcium events...")
    if detrend_data:
        print("Input F detrended")
        Fc = sgolay_detrend(Fc)

    if detect_method == 'scipy':
    
        # calcium event peaks
        percentile_99 = np.percentile(c, 99)
        peaks, props = find_peaks(c, height=percentile_99, distance=fs*15, prominence=0.5)
        amps       = props['peak_heights']
        widths, h, left_ips, right_ips = peak_widths(c, peaks, rel_height=0.5)

        # convert everything to seconds
        widths_s    = widths / fs
        rise_s      = (peaks - left_ips)  / fs
        fall_s      = (right_ips - peaks) / fs

        # interevent interval
        iei = np.diff(peaks) / fs

        # output
        asymmetry = np.median(np.abs(fall_s-rise_s))

        if plot_progress:
            %matplotlib qt
            # a) Calcium trace + peaks + half‑height points
            plt.figure()
            plt.plot(c,      label='C trace (raw)', color='k')
            plt.scatter(peaks, c[peaks],         color='r', label='Peaks')
            # interpolate y‑values at left_ips/right_ips
            y_left  = np.interp(left_ips,  np.arange(len(c)), c)
            y_right = np.interp(right_ips, np.arange(len(c)), c)
            plt.scatter(left_ips,  y_left,  color='g', label='Half‑height rise')
            plt.scatter(right_ips, y_right, color='m', label='Half‑height fall')
            plt.title('Calcium peaks + 50% crossing points')
            plt.xlabel('Sample index')
            plt.ylabel('Signal amplitude')
            plt.legend()
            plt.show()

    else:
        # -- detrend the signal --
        cZ = stats.zscore(c)

        # -- initial peak detection --
        raw_idx_peaks, _ = find_peaks(cZ)

        # cast to float so we can mark removals with np.nan
        idx_peaks_C = raw_idx_peaks.astype(float)

        # -- diagnostic plot of raw calcium peaks --
        if plot_progress:
            %matplotlib qt
            plt.figure()
            plt.plot(cZ, label='z-scored calcium')
            plt.scatter(raw_idx_peaks, cZ[raw_idx_peaks], color='r', label='initial peaks')
            plt.title('Raw calcium peaks')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(Fc, label='Detrended F')
            plt.scatter(raw_idx_peaks, Fc[raw_idx_peaks], color='r', label='initial peaks')
            plt.title('Raw calcium peaks')
            plt.legend()
            plt.show()    

        # -- time vector for peak times --
        time_range = np.linspace(0, len(c) / fs, len(c))

        # -- cluster/remove peaks closer than 10 s --
        while True:
            # get valid indices
            valid_idx = idx_peaks_C[~np.isnan(idx_peaks_C)].astype(int)
            if valid_idx.size < 2:
                break
            # corresponding times and amplitudes
            peak_times = time_range[valid_idx]
            pks_valid = cZ[valid_idx]
            # find close pairs
            offsets = np.diff(peak_times)
            close_pairs = np.where(offsets < 10)[0]
            if close_pairs.size == 0:
                break
            for evi in close_pairs:
                i1 = valid_idx[evi]
                i2 = valid_idx[evi + 1]
                if close_events == 'keep':
                    # preserve the larger amplitude peak
                    if pks_valid[evi] >= pks_valid[evi + 1]:
                        idx_peaks_C[np.where(idx_peaks_C == i2)] = np.nan
                    else:
                        idx_peaks_C[np.where(idx_peaks_C == i1)] = np.nan
                else:
                    # remove both too-close peaks
                    idx_peaks_C[np.where(idx_peaks_C == i1)] = np.nan
                    idx_peaks_C[np.where(idx_peaks_C == i2)] = np.nan

        # -- cleaned calcium peaks --
        clean_idx = idx_peaks_C[~np.isnan(idx_peaks_C)].astype(int)
        if plot_progress:
            plt.figure()
            plt.plot(cZ, label='z-scored calcium')
            plt.scatter(clean_idx, cZ[clean_idx], color='g', label='filtered peaks')
            plt.title('Filtered calcium peaks')
            plt.legend()
            plt.show()

        # -- smooth fluorescence signal --
        Fc_sm = gaussian_filter1d(Fc, sigma=ceil(fs) * 4)
        if plot_progress:
            plt.figure()
            plt.plot(Fc, alpha=0.3, label='raw Fc')
            plt.plot(Fc_sm, label='smoothed Fc')
            plt.title('Raw vs smoothed fluorescence')
            plt.legend()
            plt.show()

        # -- align calcium peaks to fluorescence peaks --
        idx_peaks_F = []
        for peak in clean_idx:
            window = np.arange(peak - int(fs * 5), peak + int(fs * 5))
            window = window[(window >= 0) & (window < len(Fc_sm))]
            sub = Fc_sm[window]
            best = window[np.argmax(sub)]
            idx_peaks_F.append(best)
        
        if plot_progress:
            plt.figure()
            plt.plot(Fc_sm, label='smoothed Fc')
            plt.scatter(idx_peaks_F, Fc_sm[idx_peaks_F], color='m', label='aligned peaks')
            plt.title('Fluorescence peaks aligned')
            plt.legend()
            plt.show()

        # 3 MAD
        #mad_f = np.median(np.abs(Fc - np.median(Fc)))
        #ttimes = np.where(Fc > np.median(Fc) + 3 * mad_f)[0]

        # -- apply z-score threshold and build peak array --
        #idx_peaks = clean_idx[Fc[clean_idx] > 6 * mad_f]
        #idx_peaks = clean_idx[Fz[clean_idx] > zscore_threshold]
        idx_peaks_C = clean_idx[cZ[clean_idx] > zscore_threshold]
        idx_peaks_F = np.array(idx_peaks_C)[cZ[np.array(idx_peaks_C)] > zscore_threshold].astype(int)
        peak_array = np.zeros_like(Fc)
        peak_array[idx_peaks_F] = 1
        Fz = stats.zscore(Fc)

        if plot_progress:
            plt.figure()
            plt.plot(Fz, label='z-scored calcium')
            plt.scatter(idx_peaks_F, Fz[idx_peaks_F], color='r', label='filtered peaks')
            plt.title('Filtered calcium peaks')
            plt.legend()
            plt.show()        

    return idx_peaks_C




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
        #idx_peaks, __, __ = calcium_events(c = c)
        idx_peaks = calcium_events_2(c=c, Fc=f, zscore_threshold=3, fs=7.5, close_events='keep', detrend_data=True, plot_progress=False)
        F_peaks = f[idx_peaks]

        # Initialize decay times
        decay_left_times = []
        decay_right_times = []
        cell_asymmetry = []

        # Loop over events and detect decay
        for idx, peak_value in zip(idx_peaks, F_peaks):

            # set to none
            decay_left = None; decay_right = None

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
            if decay_left is not None and decay_right is not None:
                right_sided = np.abs((idx-decay_right) / fs)
                left_sided = np.abs((idx-decay_left) / fs)
                cell_asymmetry.append(right_sided-left_sided)
            else:
                cell_asymmetry.append(0)

        # take the median of the asymmetry metric
        asymmetry.append(np.median(np.array(cell_asymmetry)))


        # should have a metric of variance to capture stability/variability of the asymmetry

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


# morphology features of cells
def cell_morphology(fpath: str):

    from skimage.measure import regionprops
    import numpy as np

    # read data
    F, Fneu, spks, stat, ops, iscell, blF = read_s2p(fpath=fpath)

    # dims = (height, width) of your imaging frames
    dims = (ops['Ly'], ops['Lx'])  

    morph_features = []
    for s in stat:
        mask = np.zeros(dims, dtype=bool)
        mask[s['ypix'], s['xpix']] = True
        props = regionprops(mask.astype(np.uint8))[0]

        morph_features.append({
            'area':               props.area,
            'perimeter':          props.perimeter,
            'eccentricity':       props.eccentricity,
            'solidity':           props.solidity,
            'extent':             props.extent,
            'major_axis_length':  props.major_axis_length,
            'minor_axis_length':  props.minor_axis_length,
            'orientation':        props.orientation,
            'convex_area':        props.convex_area,
            'bbox_area':          props.bbox_area,
        })

    return morph_features



    # build classifier
    def build_classifier(self, auto_feature_select = True, preset_features = False, feature_list: list = [], grid_search: bool = False):
        '''
        Args:
            >>> training_session_directory: contains a directory of folders with suite2p/plane0 files to train the model on
        '''

        # some classifier options
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble  import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm       import SVC

        # sanity check
        #assert auto_feature_select != preset_features, "You cannot set auto_feature_select and preset_features as both True or both False"

        # cleanup data
        df_clean, idx_rem = self.cleanup_classifier_data(df_all=self.df_train)

        # Assuming df_all is already created and has the necessary columns
        # Split into features (X) and labels (y)
        X = df_clean.drop(columns=['iscell', 'mouseName'])
        y = df_clean['iscell']

        # preselected features
        if preset_features == True and len(feature_list)==0:

            # using this feature list, extract from df
            #feature_list = ['comp_SNR', 'skewF', 'fitness', 'sd_r', 'corr', 'compact', 'npix_norm']
            feature_list = ['comp_SNR', 'skewF', 'fitness', 'corr', 'compact', 'npix_norm', 'npix']
            print("Using preset feature_list:",feature_list)
            selected_features = feature_list
            X = X[feature_list]

        elif preset_features == False and len(feature_list) > 0:
            print("Using preferred feature_list:",feature_list)
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
            plt.figure()
            plt.plot(range(1, X_train.shape[1] + 1), scores)    
            plt.axhline(y=np.max(scores), color='r', linestyle='--')
            plt.axvline(x=np.argmax(scores)+1, color='r', linestyle='--')
            plt.xlabel('# Features')
            plt.ylabel('Cross-Validation Score')
            plt.savefig(os.path.join(rootfun.dropbox_root(), 'OtherData', 'ClassifierBuildSuite2p', 'cross_validation_scores.png'))
            plt.close()            
            plt.show()

            # Optimal number of features
            optimal_n_features = scores.index(max(scores)) + 1
            print(f'Optimal number of features: {optimal_n_features}')

            # using recursive feature elimination, identify the most relevant features
            rfe = RFE(estimator=svc_rfe, n_features_to_select=optimal_n_features)
            rfe.fit(X_train, y_train)
            selected_features = X.columns[rfe.support_]
            print("Selected features:", selected_features)
            X_train = X_train[:, rfe.support_] # using the original, non scaled X-train and X-test
            X_test  = X_test[:, rfe.support_]
        else:
            selected_features = X.columns
            print("Using all features:", selected_features)

        # an automated method to run a grid search for best classifier and parameter combinations
        if grid_search:
            print("Running hyperparameter & model selection via GridSearchCV…")

            # 1) Define candidate estimators and their param grids
            estimators = {
                'svc_linear': {
                    'est': SVC(kernel='linear', class_weight='balanced', probability=True),
                    'params': {
                    'C': [0.01, 0.1, 1, 10, 100]
                    }
                },
                'svc_rbf': {
                    'est': SVC(kernel='rbf', class_weight='balanced', probability=True),
                    'params': {
                    'C':     [0.01, 0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
                    }
                },
                'random_forest': {
                    'est': RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=0),
                    'params': {
                    'n_estimators': [100, 200],
                    'max_depth':    [None, 5, 10]
                    }
                },
                'logistic': {
                    'est': LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1),
                    'params': {
                    'C': [0.01, 0.1, 1, 10]
                    }
                }
            }

            best_score = -np.inf
            best_clf   = None
            best_name  = None

            # 2) Loop through each and run a GridSearchCV
            for name, cfg in estimators.items():
                print(f"→ Tuning {name} …")
                grid = GridSearchCV(
                    cfg['est'],
                    cfg['params'],
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                print(f"   Best {name} score = {grid.best_score_:.3f} with {grid.best_params_}")

                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_clf   = grid.best_estimator_
                    best_name  = name

            # 3) Use the winning estimator
            print(f"Selected {best_name} as the final classifier (CV = {best_score:.3f})")
            svc = best_clf

        else:
            # fallback: use default SVC on all features
            svc = SVC(kernel='linear', C=1, gamma=0.1,
                    class_weight='balanced', probability=True)
            print("Using default linear SVC (no auto‐select)")

        # Synthetic Minority Oversampling Technique: creates synthetic values by interpolating between data points in
        # order to balance out the training dataset. Note that this isn't used for testing.
        #from imblearn.over_sampling import SMOTE
        #smote = SMOTE(random_state=42)
        #X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # run PCA to transform the dataset into non-correlated variables
        #print("Running PCA...")
        #n_components = X_train_resampled.shape[1]-1 #n-1 PCs
        #pca          = PCA(n_components=n_components)
        #X_train      = pca.fit_transform(X_train_resampled) # calculate PCs from training data
        #X_test       = pca.transform(X_test)      # uses the same PCs from the training data, applied to testing data

        # 4) Finally, fit svc on your full (resampled + PCA) training set
        svc.fit(X_train, y_train)
        self.svc = svc

        # estimate # of PCs to use based on cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.figure()
        plt.plot(cumulative_variance)
        plt.xlabel('# Principal Components')
        plt.ylabel('Cum. Variance')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.axvline(x=np.where(cumulative_variance >= 0.95)[0][0], color='r', linestyle='--')
        plt.savefig(os.path.join(rootfun.dropbox_root(), 'OtherData', 'ClassifierBuildSuite2p', 'cumulative_variance.png'))
        plt.show()
        plt.close()

        # Number of components to retain 95% variance
        n_components = np.where(cumulative_variance >= 0.95)[0][0] + 1
        print(f'# of components to retain 95% variance: {n_components}')

        # PCs to keep
        print("Cleaned up X_train and X_test accordingly...")
        X_train = X_train[:, 0:n_components]
        X_test  = X_test[:, 0:n_components]

        # Initialize the SVC classifier
        #svc = SVC(kernel='linear', C=1, gamma=0.1, class_weight='balanced', probability=True)

        # Train the model
        svc.fit(X_train, y_train) # X_train is the resampled version

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
        print(confusion_matrix(y_test, y_pred))
        print("________________________________________________________________")
        print(classification_report(y_test, y_pred))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        classification_text = classification_report(y_test, y_pred)
        ax.text(0.5, 0.5, classification_text, fontsize=12, ha='center', va='center', wrap=True)
        plt.title("Classification Report")
        plt.savefig(os.path.join(rootfun.dropbox_root(), 'OtherData', 'ClassifierBuildSuite2p', 'classification_report.png'))
        plt.close()        
        print("________________________________________________________________")

        # Save the classifyCells object
        #if save_classifier == True:
        #    self.savepath = os.path.join(rootfun.dropbox_root(),'OtherData','ClassifierBuildSuite2p','cellClassifier.pkl')
        #    self.save_model(filepath = self.savepath)            
        self.svc = svc
        self.scaler = scaler
        self.selected_features = selected_features
        self.pca = pca
        self.n_components = n_components
        self.idx_rem = idx_rem
        return svc, scaler, selected_features, pca, n_components, idx_rem
    

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
    def build_classifier(self,
                        auto_feature_select:    bool = False,
                        preset_features:        bool = False,
                        feature_list:           list = None,
                        grid_search:            bool = True,
                        per_session_scaling:    bool = True,   ## ← NEW
                        skew_threshold:         float = 1.0,
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
                ('skfilt',  SkewFilter())               # <-- drops low‑skew cells
                ('scaler', StandardScaler()),
                ('smote',   SMOTE(random_state=0)),     # will be overridden to ‘passthrough’ by GridSearch
                ('pca',     PCA()),                     # likewise can be switched off
                ('clf',     SVC(class_weight='balanced', probability=True))
            ])

            param_grid = [
                # ── Linear SVC, PCA ON ────────────────────────────────────────────────────
                {
                'skfilt__threshold': [0.5, 0.8, 1.0, 1.2],
                'smote': [SMOTE(random_state=0), None],
                'pca':   [PCA()],
                'pca__n_components': [None, min(X_train.shape[1], 5), min(X_train.shape[1], 10)],
                'clf':   [SVC(kernel='linear', class_weight='balanced', probability=True)],
                'clf__C': [0.1, 1, 10]
                },
                # ── Linear SVC, PCA OFF ───────────────────────────────────────────────────
                {
                'skfilt__threshold': [0.5, 0.8, 1.0, 1.2],
                'smote': [SMOTE(random_state=0), None],
                'pca':   [None],   # no PCA step
                'clf':   [SVC(kernel='linear', class_weight='balanced', probability=True)],
                'clf__C': [0.1, 1, 10]
                },

                # ── RBF SVC, PCA ON ───────────────────────────────────────────────────────
                {
                'skfilt__threshold': [0.5, 0.8, 1.0, 1.2],                    
                'smote': [SMOTE(random_state=0), None],
                'pca':   [PCA()],
                'pca__n_components': [None, min(X_train.shape[1], 5)],
                'clf':   [SVC(kernel='rbf', class_weight='balanced', probability=True)],
                'clf__C':     [0.1, 1, 10],
                'clf__gamma': ['scale','auto']
                },
                # ── RBF SVC, PCA OFF ──────────────────────────────────────────────────────
                {
                'skfilt__threshold': [0.5, 0.8, 1.0, 1.2],                    
                'smote': [SMOTE(random_state=0), None],
                'pca':   [None],
                'clf':   [SVC(kernel='rbf', class_weight='balanced', probability=True)],
                'clf__C':     [0.1, 1, 10],
                'clf__gamma': ['scale','auto']
                },

                # ── RandomForest (no PCA tuning) ─────────────────────────────────────────
                {
                'skfilt__threshold': [0.5, 0.8, 1.0, 1.2],                    
                'smote': [SMOTE(random_state=0), None],
                'pca':   [None],  # skip PCA entirely for trees
                'clf':   [RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=0)],
                'clf__n_estimators': [100, 200],
                'clf__max_depth':    [None, 5, 10]
                },

                # ── LogisticRegression, PCA ON ───────────────────────────────────────────
                {
                'skfilt__threshold': [0.5, 0.8, 1.0, 1.2],                    
                'smote': [SMOTE(random_state=0), None],
                'pca':   [PCA()],
                'pca__n_components': [None, min(X_train.shape[1], 5)],
                'clf':   [LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)],
                'clf__C': [0.01, 0.1, 1, 10]
                },
                # ── LogisticRegression, PCA OFF ──────────────────────────────────────────
                {
                'skfilt__threshold': [0.5, 0.8, 1.0, 1.2],                    
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
            skfilt = SkewFilter(threshold=skew_threshold)
            X_train_f = skfilt.fit_transform(X_train)
            y_train_f = y_train.loc[X_train_f.index]
            X_test_f  = skfilt.transform(X_test)
            y_test_f  = y_test.loc[X_test_f.index]  

            # b) scale
            scaler = StandardScaler().fit(X_train_f)
            Xt = scaler.transform(X_train_f)
            Xv = scaler.transform(X_test_f)

            # c) SMOTE
            sm = SMOTE(random_state=0)
            Xt, yt = sm.fit_resample(Xt, y_train_f)

            # d) PCA
            pca = PCA(n_components=min(Xt.shape[1]-1,5))
            Xt = pca.fit_transform(Xt)
            Xv = pca.transform(Xv)

            # e) SVC
            svc = SVC(kernel='linear', C=1, class_weight='balanced', probability=True)
            svc.fit(Xt, yt)

            # stash
            self.skfilt   = skfilt
            self.scaler   = scaler
            self.smote    = sm
            self.pca      = pca
            self.svc      = svc
            self.pipeline = None

            # evaluate
            y_pred = svc.predict(Xv)
            print("Test accuracy:", accuracy_score(y_test_f, y_pred))
            print(classification_report(y_test_f, y_pred))

        return self.svc
