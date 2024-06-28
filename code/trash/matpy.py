# Meant for working interchangeably with matlab files
#
# Working on a way to parse matlab files, write them as memory mappable in order to effectively load a lot of data at once
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import mmap
import pynwb

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
    self.matstruct = dict()
    dict_keys = [i for i in self.matstruct if '__' not in i]
    for keys in dict_keys:
        if datain[keys].dtype.names is not None:
            self.matstruct[keys] = dict()
            for field in datain[keys].dtype.names:
                for array in datain[keys][field]:
                    for value in array:
                        self.matstruct[keys][field] = value
        else:
            if type(datain[keys]) is np.ndarray or type(datain[keys]) is np.array:
                self.matstruct[keys] = datain[keys]
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

