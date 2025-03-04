# behfuns
#
#
# written by John S on 9/13/2024

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import os
import sys

# changing directory to load functions
os.chdir(os.getcwd())
import statfuns as stata
import plotters as plter

#bpath = r"C:\Users\spell\SpellmanLab Dropbox\OtherData\John\EXPERIMENTS\LAYER6\Panneuronal\T30_M_LeftPFC_L6Chr_PFCgcamp6f\FOV3\SEDS_day8_FOV3_probe_noOpto"

def load_putty(bpath: str):
    """
    Loads your putty table data as a pandas array. You must have saved your putty table as a .csv file

    Args:
        >>> bpath: path to your .csv putty table

    """

    # check for putty file data
    if '.csv' not in os.path.split(bpath)[-1]:
        puttyFiles = [i for i in os.listdir(bpath) if '.csv' in i and 'putty' in i]
        assert len(puttyFiles)==1, "This code needs to be adapted to handle multiple putty tables"
        puttyFile = puttyFiles[0]
        bpath = os.path.join(bpath, puttyFile)

    # load data
    behdata = pd.read_csv(bpath)

    return behdata

def getTrialCongruency(behdata):
    """
    Gets congruent and incongruent trials
    """

    # incongruent trials are trials where stimulus direction is opposite to irrelevant stim
    incong_trials = behdata[behdata['stim'] != behdata['irrel']]

    # congruent trials are trials where stim and irrelevant stim are on the same side
    cong_trials   = behdata[behdata['stim'] == behdata['irrel']]

    return incong_trials, cong_trials

def performancePlot(behdata, window_size: int):
    """
    generates a 'performance plot' with lines denoting block switches
    
    The power of this function is that you can filter your pandas array however you want,
    enter it as `behdata` and then run this function

    Args:
        >>> behdata: pandas dataFrame putty data 
    """

    # calculate a running average
    rAvg = stata.running_average(behdata['corr'], window_size=window_size)

    # smooth result
    smAvg = gaussian_filter1d(rAvg, sigma=2)

    # get block switch indices
    blkswitch = np.where(np.diff(behdata['blknum'])==1)[0]
    blkswitch_corrected = blkswitch-(window_size-1) # lose a sample

    # new trialcount x axis
    trialX = np.linspace(window_size, len(behdata['corr'])-(window_size-1), len(behdata['corr'])-(window_size-1), dtype=int)

    # generate figure
    plt.plot(trialX, smAvg, color = 'k')
    plt.plot(trialX, rAvg, color = 'gray', linewidth=.5, alpha = 1)

    for i in blkswitch_corrected:
        plt.axvline(x=i, color='r', linestyle='--', label='Vertical Line')

    plt.ylabel("Proportion Correct")
    plt.xlabel("Trial")
    plt.show()

def switchTriggeredAvg(behdata, trials_around: int = 20, pre_smooth = True, plot_result = False):
    """
    Generate a session averaged (or trial averaged) event triggered performance curve

    This will extract 10 trials around a switch in both directions, average over your data
    
    Args: 
        >>> behdata: your pandas array of putty data
        >>> trials_around: number of trials to consider around a switch
        >>> pre_smooth: set this to true if you want to calculate switchTriggeredAvg on the running averaged data
                - when False, you are just working with binary numbers to generate an avg and sem
        >>> plot_result: preset to false


    """

    # get block switch indices
    blkswitch = np.where(np.diff(behdata['blknum'])==1)[0]

    # This is a smoothed method that uses a running average
    if pre_smooth:

        # running avg
        window_size = 5

        # calculate a running average
        rAvg = np.array(stata.running_average(behdata['corr'], window_size=window_size))

        # smooth result
        smAvg = gaussian_filter1d(rAvg, sigma=2)
        
        # new blkswitch index
        blkswitch_corrected = blkswitch-(window_size-1) # lose a sample

        # loop over the blkswitch_corrected variable and use the presmoothed avg
        corrTriggered = []
        for i in blkswitch_corrected:
            try:
                corrTriggered.append(smAvg[i-trials_around : i+trials_around])
            except:
                print("Failed to get samples",i-trials_around,":",i+trials_around)

    else:

        # loop over blkswitch and get trials around
        corrTriggered = []
        for i in blkswitch:
            corrTriggered.append(np.array(behdata['corr'][i-trials_around : i+trials_around]))
    
    # identify cases of empty arrays and remove
    corrFilt = [i for i in corrTriggered if len(i) > 0]  
    switchTriggArray = np.array(corrFilt) # convert to numpy

    # get the average over switches
    switchTriggeredAv = np.mean(corrTriggArray,axis=0)
    switchTriggeredSEM = np.std(corrTriggArray, axis=0)/np.sqrt(corrTriggArray.shape[0])

    # Plot the line
    if plot_result:
        trialX = np.linspace(-trials_around, trials_around, len(switchTriggeredAvg), dtype=int)

        fig = plt.figure()
        plt.plot(trialX, switchTriggeredAvg, 'k-')

        # Fill the shaded error region
        plt.fill_between(trialX, switchTriggeredAv - switchTriggeredSEM, switchTriggeredAv + switchTriggeredSEM)

    return switchTriggeredAv, switchTriggeredSEM, switchTriggArray
