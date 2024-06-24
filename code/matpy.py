# This code is meant to work with .mat files from MATLAB as well as contain matlab equivalent functions
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import mmap
import numpy as np
from scipy.spatial import cKDTree

class matpy():

    def __init__(self, matpath: str):
        """
        Initialize object by entering the path to your matlatb file. This detects cases where only the path is provided.
        
        """
        self.matpath = matpath

        if '.mat' in os.path.split(self.matpath)[-1]:
            self.matfile = os.path.split(self.matpath)[-1]
            print(".mat file discovered:",self.matfile)
        else:
            dir_contents = os.listdir(self.matpath)
            self.matfiles = [i for i in dir_contents if '.mat' in i]
            print("multiple .mat files discovered. This may cause errors. Please define 'matpath' as your specific .mat file to load")

    def parsemat(self):
        """
            This method reads your .mat file blindly, returning the .mat variables and returning a dictionary of arrays.

            John Stout 
            6/10/2024
        """

        # load your .mat file
        self.matdata = sio.loadmat(self.matpath, mat_dtype=True)
        
        # remove your excess variables
        dunder_dels  = [i for i in self.matdata.keys() if '__' in i] # remove headers and dunders
        for key in dunder_dels:
            del self.matdata[key]
        self.varnames = [i for i in self.matdata.keys() if '__' not in i] # remove headers and dunders
        
        return self.matdata, self.varnames
    
    def parseNeuronStruct(self):

        # parse the file
        self.parsemat()

        # get "neuron" and reorganize the embedded inputs
        if 'neuron' in self.varnames:
            self.neuronVarNames = [i for i in self.matdata['neuron'].dtype.names]
            for keys in self.varnames:
                self.matstruct = dict()
                for vars in self.matdata['neuron'].dtype.names: # variable names
                    data = getEmbeddedData(self.matdata['neuron'][vars])  # extract data
                    self.matstruct[vars] = data

        return self.matstruct, self.neuronVarNames
    
    def mat2nwb():
        pass

    def memapmat(self, fsave: str):
        """
        This method takes a matlab file, parses it to bits, then memory maps the file using numpy methods for quick and easy reading

        John Stout 6/10/2024
        """

        # parse the matlab file
        matvar  = self.parsemat() # save as np file
        
        # save what you can
        for i in matvar.keys():
            if 'neuron' in i:

                # create a filename
                filename = os.path.join(os.path.split(self.matpath)[0],'mmap_'+self.matfile.split('.mat')[0]+'_'+i+'.npy')            
                data = matvar[i]

                # map array                
                mmapped_array = np.memmap(filename, dtype='uint16',
                                        mode='w+', shape=len(matvar[i]))
                
                mmapped_array[:] = matvar[i]
                mmapped_array.flush()

    def memapmat_reader(self):
        """
        Decompose the memapmat file to a dictionary and return to user with memory mapped items
        """
        pass

def autoassigndict(datain):
    '''
    Saves information from your dictionary based on the name of the variable
    '''
    # Convert the numpy array to a Python dictionary
    matstruct = dict()
    dict_keys = [i for i in matstruct if '__' not in i]
    for keys in dict_keys:
        if datain[keys].dtype.names is not None:
            matstruct[keys] = dict()
            for field in datain[keys].dtype.names:
                for array in datain[keys][field]:
                    for value in array:
                        matstruct[keys][field] = value
        else:
            if type(datain[keys]) is np.ndarray or type(datain[keys]) is np.array:
                matstruct[keys] = datain[keys]
                datain[keys][0][0].dtype

    # above does the trick to get data into a structure, but I can't save that data as a memory mappable file
    temp = []
    for keys in dict_keys:
        temp.append(datain[keys])

def getEmbeddedData(data: np.ndarray, precision: str = 'float32'):
    """
    Sometimes data comes in many shapes and colors, but when it is embedded, it is really annoying to get out.

    This code extracts embedded data in np.ndarrays

    Args:
        > data: nested numpy array
        > precision: precision of data to convert (dtype). Default is float32

    John Stout
    """

    # extract data from embedded structure
    while np.max(data.shape)==1: 
        data = data[0]

    return np.array(data,dtype='float32')

def dsearchn(points, query_points):
    """
    Find the index of the nearest value in `points` for each `query_point`.

    Args:
        points (numpy.ndarray): Array of points (N-dimensional).
        query_points (numpy.ndarray): Array of query points (N-dimensional).

    Returns:
        numpy.ndarray: Indices of the nearest points in `points` for each query point.

    Thanks CoPilot :)
    """
    tree = cKDTree(points)
    _, indices = tree.query(query_points)
    return indices

def dsearchn2(x, v):
    z=np.atleast_2d(x)-np.atleast_2d(v).T
    return np.where(np.abs(z).T==np.abs(z).min(axis=1))[0]

def generateTrialTimes_spellmanData(matpath):

    # load in behavior
    behdict = sio.loadmat(matpath)

    # now align trials
    trial_start_times = behdict['trialStartTimes'][0]
    trial_end_times   = behdict['trialEndTimes'][0]

    # if there are more end times than there are start times
    if len(trial_end_times) > len(trial_start_times):

        # more common than not, you will find an extra end time
        trial_end_times = trial_end_times[0:-1]
        print("End trial shaved off")
    
    assert len(trial_end_times) == len(trial_start_times), "There are an uneven number of start and stop times"

    # if there are just as many end as there are start times
    if len(trial_end_times) == len(trial_start_times):

        # concatenate
        trials_cat = np.concatenate(([trial_start_times], [trial_end_times]),axis=0).T

        # check rows/columns
        if trials_cat.shape[0] < trials_cat.shape[1]:
            trials_cat = trials_cat.T
        
        # calculate trial durations
        trial_durations = trials_cat[:,1]-trials_cat[:,0]

        # search for negative values
        trials_misaligned = (trial_durations < 0).any()
        assert trials_misaligned==False, "Start and stop times are not aligned. Negative values detected (stop-start) indicating stops that occured after a start."

    # trial duration info
    behdict['trialSampleLength']  = [trial_durations]
    behdict['trialStartTimes']    = [trial_start_times]
    behdict['trialEndTimes']      = [trial_end_times]
    behdict['trials_cat']         = [trials_cat]

    # get rewarded trial times
    trial_times = behdict['trials_cat'][0]
    trialRewardCorrectTimes = []; trialRewardIncorrectTimes = []; trialRewardIdx = []; 
    trialRewardTimesAll = np.empty((trial_times.shape[0],)); trialRewardTimesAll[:]=np.nan
    trialRewardTimeIdx = np.empty((trial_times.shape[0],)); trialRewardTimeIdx[:]=np.nan
    for triali, trialtimes in enumerate(trial_times):
        trial_found = 0
        for rewti in behdict['rewardTimes'][0]:
            if rewti > trialtimes[0] and rewti <= trialtimes[1] and behdict['trialCorrect'][0][triali]==1: # if reward time falls in between trial
                trialRewardCorrectTimes.append(rewti)
                trial_found = 1
                trialRewardTimeIdx[triali]=rewti
                trialRewardTimesAll[triali]=rewti
            elif rewti > trialtimes[0] and rewti <= trialtimes[1] and behdict['trialCorrect'][0][triali]==0:
                trialRewardIncorrectTimes.append(rewti) # times for reward
                trial_found = 1 # marker for whether reward happened or not
                trialRewardTimeIdx[triali]=rewti
                trialRewardTimesAll[triali]=rewti
        trialRewardIdx.append(trial_found)

    # returns the following
    behdict['trialRewardCorrectTimes']   = np.array(trialRewardCorrectTimes) # index of correct reward times
    behdict['trialRewardIncorrectTimes'] = np.array(trialRewardIncorrectTimes) # index of incorrect reward times
    behdict['trialRewardIdx']            = np.array(trialRewardIdx) # boolean 1s/0s per each trial if it was rewarded
    behdict['trialRewardTimesAll']       = np.array(trialRewardTimesAll) # index of all rewarded times organized by trial

    return behdict