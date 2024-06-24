import matplotlib.pyplot as plt
import numpy as np
import cv2
import nwbfun

# technique to rescale the image easily to maximize viewing capacity
def scaled_imshow(image):
    plt_data = plt.imshow(image, 
                          vmin=np.percentile(np.ravel(image),50), 
                          vmax=np.percentile(np.ravel(image),99.5))
    return plt_data

# gaussian filter over the image
def gauss_filt(image, gSig_filt):
    #gSig_filt = np.array([7,7])
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    ex = cv2.filter2D(np.array(image, dtype=np.float32),-1, ker2D, borderType=cv2.BORDER_REFLECT)
    return ex

def seds_peth(neural_data: np.array, fs: float, behdata: dict, time_around: list = [10, 10]):
    """
    Extracts various combinations of time locked data based on the SEDS task

    This is really valuable for quickly organizing your data into trial x unit information
    
    Args:
        >>> neural_data: np.array of shape (Components X Time)
        >>> fs: sampling rate of neural data
        >>> behdata: behavioral data in dictionary
        >>> time_around: list type variable telling you how much time before and after the primary event marker

    Returns:
        >>> peth: peri-event time histogram in the shape of (EventMarker X component X time)
                    so if your event marker was 'rewardTimes', then EventMarker = rewardTimes and time
                    reflects time around your eventMarker as based on your `time_around` variable.
    
    John Stout
    """

    # these are your time and trial markers for .mat behavioral data
    time_markers_def = ['trialStartTimes', 'rewardTimes', 'stimOnTimes', 'stimOffTimes', 'lickTimesL', 'lickTimesR', 'rewardTimesIdxTrials']
    trial_organizers = ['setIDs', 'trialCorrect', 'trialLRs', 'irrelLRs']

    # search for each time_marker in the behdictionary
    found_string = []
    for i in time_markers_def:
        found = 0
        for keys in behdata.keys():
            if i in keys:
                found = 1
        found_string.append(found)

    # this code ensures that only the behavioral variables found are used below. This code entirely avoids the cheap try/catch
    time_markers = [time_markers_def[i] for i in range(len(found_string)) if found_string[i] == 1]

    # convert your time_around variable to samples around
    samples_around = np.array(time_around)*int(fs)

    # prepare output
    peth = dict()

    # you can only time lock things once, so writing everything out would be redundant
    for time_marker in time_markers:

        if len(behdata[time_marker]) == 1:
            data_primary = behdata[time_marker][0]
        else:
            data_primary = behdata[time_marker]

        # for some reason, the amount of reward times differs from the amount of correct trials
        #data_primary = behdata[time_marker]
        print(time_marker, "detected")

        # here is your dict
        peth[time_marker] = peth_looper(neural_data=neural_data, indexer=data_primary, samples_around=samples_around) 

        if 'trialStartTimes' in time_marker or 'stimOnTimes' in time_marker or 'stimOffTimes' in time_marker or 'rewardTimesIdxTrials' in time_marker:

            # set
            setID_0        = data_primary[behdata['setIDs'][0]==0] # set 0
            setID_1        = data_primary[behdata['setIDs'][0]==1] # set 1

            # accuracy
            trialCorrect   = data_primary[behdata['trialCorrect'][0]==1] # correct
            trialIncorrect = data_primary[behdata['trialCorrect'][0]==0] # incorrect

            # relevant stimulus
            trialLeft      = data_primary[behdata['trialLRs'][0]==0] # left
            trialRight     = data_primary[behdata['trialLRs'][0]==1] # right

            # irrelevant stimulus
            irrelLeft      = data_primary[behdata['irrelLRs'][0]==0] # left
            irrelRight     = data_primary[behdata['irrelLRs'][0]==1] # right

            # set x accuracy
            setID_0_correct   = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==1))]
            setID_0_incorrect = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==0))]
            
            setID_1_correct   = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==1))]
            setID_1_incorrect = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==0))]
            
            # set x accuracy x relevant stimulus
            setID_0_correct_relLeft      = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==1) & (behdata['trialLRs'][0]==0))]
            setID_0_incorrect_relLeft    = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==0) & (behdata['trialLRs'][0]==0))]
            setID_0_correct_relRight     = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==1) & (behdata['trialLRs'][0]==1))]
            setID_0_incorrect_relRight   = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==0) & (behdata['trialLRs'][0]==1))]
            
            setID_1_correct_relLeft      = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==1) & (behdata['trialLRs'][0]==0))]
            setID_1_incorrect_relLeft    = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==0) & (behdata['trialLRs'][0]==0))]
            setID_1_correct_relRight     = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==1) & (behdata['trialLRs'][0]==1))]
            setID_1_incorrect_relRight   = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==0) & (behdata['trialLRs'][0]==1))]

            # set x accuracy x irrelevant
            setID_0_correct_irrelLeft    = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==1) & (behdata['irrelLRs'][0]==0))]
            setID_0_incorrect_irrelLeft  = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==0) & (behdata['irrelLRs'][0]==0))]
            setID_0_correct_irrelRight   = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==1) & (behdata['irrelLRs'][0]==1))]
            setID_0_incorrect_irrelRight = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==0) & (behdata['irrelLRs'][0]==1))]
            
            setID_1_correct_irrelLeft    = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==1) & (behdata['irrelLRs'][0]==0))]
            setID_1_incorrect_irrelLeft  = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==0) & (behdata['irrelLRs'][0]==0))]
            setID_1_correct_irrelRight   = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==1) & (behdata['irrelLRs'][0]==1))]
            setID_1_incorrect_irrelRight = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==0) & (behdata['irrelLRs'][0]==1))]

            # set x accuracy x irrelevant x relevant
            setID_0_correct_irrelLeftRelLeft    = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==1) & (behdata['irrelLRs'][0]==0) & (behdata['trialLRs'][0]==0))]
            setID_0_incorrect_irrelLeftRelLeft  = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==0) & (behdata['irrelLRs'][0]==0) & (behdata['trialLRs'][0]==0))]
            setID_0_correct_irrelRightRelLeft   = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==1) & (behdata['irrelLRs'][0]==1) & (behdata['trialLRs'][0]==0))]
            setID_0_incorrect_irrelRightRelLeft = data_primary[np.where((behdata['setIDs'][0]==0) & (behdata['trialCorrect'][0]==0) & (behdata['irrelLRs'][0]==1) & (behdata['trialLRs'][0]==0))]
            
            setID_1_correct_irrelLeftRelRight    = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==1) & (behdata['irrelLRs'][0]==0) & (behdata['trialLRs'][0]==1))]
            setID_1_incorrect_irrelLeftRelRight  = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==0) & (behdata['irrelLRs'][0]==0) & (behdata['trialLRs'][0]==1))]
            setID_1_correct_irrelRightRelRight   = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==1) & (behdata['irrelLRs'][0]==1) & (behdata['trialLRs'][0]==1))]
            setID_1_incorrect_irrelRightRelRight = data_primary[np.where((behdata['setIDs'][0]==1) & (behdata['trialCorrect'][0]==0) & (behdata['irrelLRs'][0]==1) & (behdata['trialLRs'][0]==1))]

            # get your data combinations
            peth[time_marker+'setID_0']                              = peth_looper(neural_data=neural_data, indexer=setID_0,                             samples_around=samples_around)
            peth[time_marker+'setID_1']                              = peth_looper(neural_data=neural_data, indexer=setID_1,                             samples_around=samples_around)
            
            peth[time_marker+'trialCorrect']                         = peth_looper(neural_data=neural_data, indexer=trialCorrect,                        samples_around=samples_around)
            peth[time_marker+'trialIncorrect']                       = peth_looper(neural_data=neural_data, indexer=trialIncorrect,                      samples_around=samples_around)
            
            peth[time_marker+'relLeft']                              = peth_looper(neural_data=neural_data, indexer=trialLeft,                           samples_around=samples_around)
            peth[time_marker+'relRight']                             = peth_looper(neural_data=neural_data, indexer=trialRight,                          samples_around=samples_around)                        

            peth[time_marker+'irrelLeft']                            = peth_looper(neural_data=neural_data, indexer=irrelLeft,                           samples_around=samples_around)
            peth[time_marker+'irrelRight']                           = peth_looper(neural_data=neural_data, indexer=irrelRight,                          samples_around=samples_around)                        

            peth[time_marker+'setID_0_correct']                      = peth_looper(neural_data=neural_data, indexer=setID_0_correct,                     samples_around=samples_around)
            peth[time_marker+'setID_0_incorrect']                    = peth_looper(neural_data=neural_data, indexer=setID_0_incorrect,                   samples_around=samples_around)
            peth[time_marker+'setID_1_correct']                      = peth_looper(neural_data=neural_data, indexer=setID_1_correct,                     samples_around=samples_around)
            peth[time_marker+'setID_1_incorrect']                    = peth_looper(neural_data=neural_data, indexer=setID_1_incorrect,                   samples_around=samples_around)
            
            peth[time_marker+'setID_0_correct_relLeft']              = peth_looper(neural_data=neural_data, indexer=setID_0_correct_relLeft,             samples_around=samples_around)
            peth[time_marker+'setID_0_incorrect_relLeft']            = peth_looper(neural_data=neural_data, indexer=setID_0_incorrect_relLeft,           samples_around=samples_around)
            peth[time_marker+'setID_0_correct_relRight']             = peth_looper(neural_data=neural_data, indexer=setID_0_correct_relRight,            samples_around=samples_around)
            peth[time_marker+'setID_0_incorrect_relRight']           = peth_looper(neural_data=neural_data, indexer=setID_0_incorrect_relRight,          samples_around=samples_around)

            peth[time_marker+'setID_1_correct_relLeft']              = peth_looper(neural_data=neural_data, indexer=setID_1_correct_relLeft,             samples_around=samples_around)
            peth[time_marker+'setID_1_incorrect_relLeft']            = peth_looper(neural_data=neural_data, indexer=setID_1_incorrect_relLeft,           samples_around=samples_around)
            peth[time_marker+'setID_1_correct_relRight']             = peth_looper(neural_data=neural_data, indexer=setID_1_correct_relRight,            samples_around=samples_around)
            peth[time_marker+'setID_1_incorrect_relRight']           = peth_looper(neural_data=neural_data, indexer=setID_1_incorrect_relRight,          samples_around=samples_around)

            peth[time_marker+'setID_0_correct_irrelLeft']            = peth_looper(neural_data=neural_data, indexer=setID_0_correct_irrelLeft,           samples_around=samples_around)
            peth[time_marker+'setID_0_incorrect_irrelLeft']          = peth_looper(neural_data=neural_data, indexer=setID_0_incorrect_irrelLeft,         samples_around=samples_around)
            peth[time_marker+'setID_0_correct_irrelRight']           = peth_looper(neural_data=neural_data, indexer=setID_0_correct_irrelRight,          samples_around=samples_around)
            peth[time_marker+'setID_0_incorrect_irrelRight']         = peth_looper(neural_data=neural_data, indexer=setID_0_incorrect_irrelRight,        samples_around=samples_around)

            peth[time_marker+'setID_1_correct_irrelLeft']            = peth_looper(neural_data=neural_data, indexer=setID_1_correct_irrelLeft,           samples_around=samples_around)
            peth[time_marker+'setID_1_incorrect_irrelLeft']          = peth_looper(neural_data=neural_data, indexer=setID_1_incorrect_irrelLeft,         samples_around=samples_around)
            peth[time_marker+'setID_1_correct_irrelRight']           = peth_looper(neural_data=neural_data, indexer=setID_1_correct_irrelRight,          samples_around=samples_around)
            peth[time_marker+'setID_1_incorrect_irrelRight']         = peth_looper(neural_data=neural_data, indexer=setID_1_incorrect_irrelRight,        samples_around=samples_around)

            peth[time_marker+'setID_0_correct_irrelLeftRelLeft']     = peth_looper(neural_data=neural_data, indexer=setID_0_correct_irrelLeftRelLeft,    samples_around=samples_around)
            peth[time_marker+'setID_0_incorrect_irrelLeftRelLeft']   = peth_looper(neural_data=neural_data, indexer=setID_0_incorrect_irrelLeftRelLeft,  samples_around=samples_around)
            peth[time_marker+'setID_0_correct_irrelRightRelLeft']    = peth_looper(neural_data=neural_data, indexer=setID_0_correct_irrelRightRelLeft,   samples_around=samples_around)
            peth[time_marker+'setID_0_incorrect_irrelRightRelLeft']  = peth_looper(neural_data=neural_data, indexer=setID_0_incorrect_irrelRightRelLeft, samples_around=samples_around)

            peth[time_marker+'setID_1_correct_irrelLeftRelRight']    = peth_looper(neural_data=neural_data, indexer=setID_1_correct_irrelLeftRelRight,    samples_around=samples_around)
            peth[time_marker+'setID_1_incorrect_irrelLeftRelRight']  = peth_looper(neural_data=neural_data, indexer=setID_1_incorrect_irrelLeftRelRight,  samples_around=samples_around)
            peth[time_marker+'setID_1_correct_irrelRightRelRight']   = peth_looper(neural_data=neural_data, indexer=setID_1_correct_irrelRightRelRight,   samples_around=samples_around)
            peth[time_marker+'setID_1_incorrect_irrelRightRelRight'] = peth_looper(neural_data=neural_data, indexer=setID_1_incorrect_irrelRightRelRight, samples_around=samples_around)

    return peth

def peth_looper(neural_data, indexer, samples_around):
    # get neural data
    peth_data = []
    for i in range(len(indexer)):
        if ~np.isnan(indexer[i]):
            # change that the indexer is of type int
            start = int(indexer[i])-samples_around[0]; end = int(indexer[i])+samples_around[1]
            if start > 0:
                try:
                    peth_data.append(neural_data[:,start:end])
                except:
                    pass  
    return np.array(peth_data)

def sort_activity_by_xmax(neural_data):
    """
    Args:
        >>> neural_data: np.array of neural data (Cells X Activity)

    Returns:
        >>> sorted_data: sorted neural data (cells sorted according to their peak activity on X)
        >>> sorted_x_idx: sorted x information to sort data

    John Stout
    """

    # max of response on the Y axis according to its location on X
    x_maxes = np.argmax(neural_data,axis=1)

    # now sort your spk array based on the sorted x_maxes
    sorted_x_idx = np.argsort(x_maxes)

    # sort
    sorted_data = neural_data[sorted_x_idx]

    return sorted_data, sorted_x_idx

def shadedErrorBar(x_data, y_mean, y_error):

    # Generate some example data
    #x = np.linspace(0, 30, 30)
    #y = np.sin(x / 6 * np.pi)
    #error = np.random.normal(0.1, 0.02, size=y.shape)
    #y += np.random.normal(0, 0.1, size=y.shape)

    # Plot the line
    plt.plot(x_data, y_mean, 'k-')

    # Fill the shaded error region
    plt.fill_between(x_data, y_mean - y_error, y_mean + y_error)

    # Show the plot
    return plt.show()