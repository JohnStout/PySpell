# compute deconvolution
from suite2p.extraction import dcnv
import numpy as np
from scipy import stats

def baseline_corrected_F(F, Fneu, ops):

    """
    This code was taken from the suite2p website to provide baseline subtracted estimates of F

    https://suite2p.readthedocs.io/en/latest/deconvolution.html

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
