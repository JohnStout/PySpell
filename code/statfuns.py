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

def running_average(data, window_size):
    """
    AI generated code that performs running avg over a window_size denoted by you

    Args:
        >>> data: 1D array of data
        >>> window_size: window to conduct running average over

    Returns:
        >>> running_averages
    """
    running_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        window_average = sum(window) / window_size
        running_averages.append(window_average)
        #print(i,'through',i+window_size)
    return running_averages
