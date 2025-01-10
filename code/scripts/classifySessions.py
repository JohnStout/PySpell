# classifier

# load modules
import os; import matplotlib.pyplot as plt; import tifffile as tf
path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)
from s2pfuns import detrend_signal
from pathlib import Path
import numpy as np
import time
import suite2p
from suite2p.extraction import dcnv
import csv

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
        fitness, erfc, sd_r, md = comp_eval.compute_event_exceptionality(traces=blF)

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
                
                




        # identify cells with overlap
        overlap = roi['overlap']



        # cells with overlap
        overlapping_cells = np.where(overlap==True)[0]

        # loop over cells with overlap
        for overi in overlapping_cells:
            ypix_comp = stat[overi].ypix

    pass

# TODO?
# use suite2p's classifier first
# in series, compare to inclusion of compactness, and maybe weight more heavily the skewF variable

# classifier_sessions
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

# -- classifier testing -- #
predict_sessions = [
    #r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\ClassifierTestSuite2p\L612_CD2_whisker_day2_p70it_FOV1_LBC2_optoRec_img",

    # for tim
    #r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L612_F_RightPFC_L6Chr_PFCgcamp8f_L6PAN\SEDS_day7_LBC2_p70_optoRec_FOV1_img",
    #r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L614_F_LeftPFC_L6Chr_PFCgcamp6f_L6PAN\SEDS_day2_optoRec_LBC2_p70_FOV3_img",
    #r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L608_F_LeftPFC_L6Chr_PFCgcamp6f_L6PAN\SEDS_day8_FOV1_LBC0_noOpto_img"

    #r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L615_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN\CD1_whisker_day1_optoRec_FOV1_LBC2_img"
   
    #r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L613_F_LeftPFC_L6Chr_PFCgcamp8f_L6PAN\CD1_whisker_day1_LBC2_FOV2_optoRec_img",
    #r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Subjects\Imaging\L615_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN\CD1_odor_day2_optoRec_FOV1_LBC2_p70_img_001"
]

for sessi in predict_sessions:

    # gather variables
    df_predict = gather_classifier_data([sessi])

    # test classifier
    predictions, probabilities, decision_scores = predict_cell(df_predict = df_predict, svc=svc, scaler=scaler, selected_features=selected_features,
                                                                pca=pca, n_components=n_components)

    rewrite_iscell(predict_sessions = [sessi], predictions=predictions, probabilities = probabilities)


'''
# New unseen data to predict
unseen_data = pd.DataFrame({
    # Include your data columns here
})

# Make predictions on the new unseen data
predictions = predict_new_data(unseen_data, svc, scaler)
print(predictions)






# plot
%matplotlib inline
plt.figure()
plt.scatter(sv[:,0], sv[:,1], s=15, edgecolors='k', c='r')
plt.show()

## method with PCA is improved in accuracy

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
n_components = 10  # Number of principal components to keep (you can adjust this value)
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled) # calculate PCs from training data
X_test_pca = pca.transform(X_test_scaled) # uses the same PCs from the training data, applied to testing data

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming X_train_scaled is already defined
pca = PCA()
pca.fit(X_train_scaled)

# Plot cumulative explained variance
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


# Initialize the SVC classifier
svc = SVC(kernel='linear', C=1, gamma=0.1)

# Train the model
svc.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = svc.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Evaluate accuracy on accepted components (true)
accepted_indices = np.where(y_test == True)[0]
accuracy_accepted = accuracy_score(y_test.iloc[accepted_indices], y_pred[accepted_indices])
print(f'Accuracy on accepted components (true): {accuracy_accepted}')




# recursive feature elimination removes the least important features
# and builds model until optimal number of features are reached










r_values_min=0.8,
r_values_lowest=-1,

idx_components_r = np.where(r_values >= r_values_min)[0]
idx_components_raw = np.where(comp_SNR > min_SNR)[0]

bad_comps = np.where((r_values <= r_values_lowest) | (comp_SNR <= min_SNR_reject))[0]
cnn_values = []

idx_components = np.union1d(idx_components, idx_components_r)
idx_components = np.union1d(idx_components, idx_components_raw)
idx_components = np.setdiff1d(idx_components, bad_comps)
idx_components_bad = np.setdiff1d(list(range(len(r_values))), idx_components)



# good inputs
# fitness, npix_norm, compact, 


# Show plot
plt.show()






# non correlated vars
# sd_r, skew, fitness

# classifier loc
class_data = np.load(os.path.join(path_added,'scripts','classifier_stout.npy'), allow_pickle = True).item()

self = Classifier()
class Classifier:
    """ ROI classifier model that uses logistic regression
    
    Parameters
    ----------

    classfile: string (optional, default None)
        path to saved classifier

    keys: list of str (optional, default None)
        keys of ROI stat to use to classify

    """

    def __init__(self, classfile=None, keys=None):
        # stat are cell stats from currently loaded recording
        # classfile is a previously saved classifier file
        if classfile is not None:
            self.load(classfile, keys=keys)
        else:
            self.loaded = False
            self.keys = keys

    def load(self, classfile, keys=None):
        """ data loader

        saved classifier contains stat with classification labels 

        Parameters
        ----------
        
        classfile: string 
            path to saved classifier

        keys: list of str (optional, default None)
            keys of ROI stat to use to classify
         
        """
        try:
            model = np.load(classfile, allow_pickle=True).item()
            if keys is None:
                self.keys = model["keys"]
                self.stats = model["stats"]
            else:
                model["keys"] = np.array(model["keys"])
                ikey = np.isin(model["keys"], keys)
                self.keys = model["keys"][ikey].tolist()
                self.stats = model["stats"][:, ikey]
            self.iscell = model["iscell"]
            self.loaded = True
            self.classfile = classfile
            self._fit()
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
            print("ERROR: incorrect classifier file")
            self.loaded = False

    def run(self, stat, p_threshold: float = 0.5) -> np.ndarray:
        """Returns cell classification thresholded with "p_threshold" and its probability."""
        probcell = self.predict_proba(stat)
        is_cell = probcell > p_threshold
        return np.stack([is_cell, probcell]).T

    def predict_proba(self, stat):
        """ apply logistic regression model and predict probabilities

        model contains stat with classification labels 

        Parameters
        ----------
        
        stat : list of dicts
            needs self.keys keys

        """
        test_stats = np.array([stat[j][k] for j in range(len(stat)) for k in self.keys
                              ]).reshape(len(stat), -1)
        logp = self._get_logp(test_stats)
        y_pred = self.model.predict_proba(logp)[:, 1]
        return y_pred

    def save(self, filename: str) -> None:
        """ save classifier to filename """
        np.save(filename, {
            "stats": self.stats,
            "iscell": self.iscell,
            "keys": self.keys
        })

    def _get_logp(self, stats):
        """ compute log probability of set of stats
        
        Parameters
        --------------

        stats : 2D array
            size [ncells, nkeys]
        
        """
        logp = np.zeros(stats.shape)
        for n in range(stats.shape[1]):
            x = stats[:, n]
            x[x < self.grid[0, n]] = self.grid[0, n]
            x[x > self.grid[-1, n]] = self.grid[-1, n]
            x[np.isnan(x)] = self.grid[0, n]
            ibin = np.digitize(x, self.grid[:, n], right=True) - 1
            logp[:, n] = np.log(self.p[ibin, n] + 1e-6) - np.log(1 - self.p[ibin, n] +
                                                                 1e-6)
        return logp

    def _fit(self):
        """ fit logistic regression model using stats, keys and iscell """
        nodes = 100
        ncells, nstats = self.stats.shape
        ssort = np.sort(self.stats, axis=0)
        isort = np.argsort(self.stats, axis=0)
        ix = np.linspace(0, ncells - 1, nodes).astype("int32")
        grid = ssort[ix, :]
        p = np.zeros((nodes - 1, nstats))
        for j in range(nodes - 1):
            for k in range(nstats):
                p[j, k] = np.mean(self.iscell[isort[ix[j]:ix[j + 1], k]])
        p = gaussian_filter(p, (2., 0))
        self.grid = grid
        self.p = p
        logp = self._get_logp(self.stats)
        self.model = LogisticRegression(C=100., solver="liblinear")
        self.model.fit(logp, self.iscell)



# pnr


np.where(iscell==False)

import pandas as pd
df = pd.DataFrame(data = {
    'iscell': iscell,
    'snr': snr,
    'skewF': skF,
    'skewN': skN
    })

# what if I test their distributions?
plt.hist(Ndet[0])
plt.hist(Fdet[0])


# Map the boolean variable to colors
colors = np.where(df_all['iscell'], 'red', 'blue')

# Create 3D scatter plot
%matplotlib widget
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(df_all['comp_SNR'], df_all['npix_norm'], df_all['compact'], c=colors, marker='o') 

'''