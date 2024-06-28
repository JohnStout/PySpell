import numpy as np
from scipy import stats

def mad(data):
    """
    Calculates the Mean Absolute Deviation (M.A.D.) for a given data set.
    Args:
        data (numpy.ndarray or list): Input data.

    Returns:
        float: Mean Absolute Deviation.

    Thanks CoPilot :)
    """
    print("This code was generated with CoPilot")
    return np.mean(np.abs(data - np.mean(data)))

def sem(data: np.array):
    '''
    A single array of data
    '''
    sem_data = np.std(data, ddof=1) / np.sqrt(np.size(data))
    return sem_data