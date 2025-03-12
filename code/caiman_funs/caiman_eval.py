#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:49:57 2016

@author: agiovann
"""

from builtins import range
from past.utils import old_div

import cv2
import itertools
import logging
import numpy as np
import os
import peakutils
import tensorflow as tf
import scipy
from scipy.sparse import csc_matrix
from scipy.stats import norm
from typing import Any, List, Tuple, Union
import warnings

from .caiman_stats import mode_robust_fast, mode_robust

def compute_event_exceptionality(traces: np.ndarray,
                                 robust_std: bool = False,
                                 N: int = 5,
                                 use_mode_fast: bool = False,
                                 sigma_factor: float = 3.) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    """
    Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Args:
        Y: ndarray
            movie x,y,t

        A: scipy sparse array
            spatial components

        traces: ndarray
            Fluorescence traces

        N: int
            N number of consecutive events (N must be greater than 0)

        sigma_factor: float
            multiplicative factor for noise estimate (added for backwards compatibility)

    Returns:
        fitness: ndarray
            value estimate of the quality of components (the lesser the better)

        erfc: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise

        noise_est: ndarray
            the components ordered according to the fitness
    """
    if N == 0:
        # Without this, numpy ranged syntax does not work correctly, and also N=0 is conceptually incoherent
        raise Exception("FATAL: N=0 is not a valid value for compute_event_exceptionality()")

    T = np.shape(traces)[-1]
    if use_mode_fast:
        md = mode_robust_fast(traces, axis=1)
    else:
        md = mode_robust(traces, axis=1)

    ff1 = traces - md[:, None]

    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:

        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, _ in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(old_div(np.sum(ff1**2, 1), Ns))

    # compute z value
    z = old_div((traces - md[:, None]), (sigma_factor * sd_r[:, None]))

    # probability of observing values larger or equal to z given normal
    # distribution with mean md and std sd_r
    #erf = 1 - norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    #erf = np.log(erf)
    # compute with this numerically stable function
    erf = scipy.special.log_ndtr(-z)

    # moving sum
    erfc = np.cumsum(erf, 1)
    erfc[:, N:] -= erfc[:, :-N]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    return fitness, erfc, sd_r, md
