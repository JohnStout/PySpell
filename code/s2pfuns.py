# compute deconvolution
from suite2p.extraction import dcnv
import numpy as np
from scipy import stats

def baseline_corrected_F(F, Fneu, ops):

    """
    This code was taken from the suite2p website to provide baseline subtracted estimates of F

    https://suite2p.readthedocs.io/en/latest/deconvolution.html

    This is dF/F

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
