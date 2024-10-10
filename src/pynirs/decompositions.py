"""
Module handling various data decompositions such as PCA, ICA and SVD for data cleaning and processing.


This module contains functions and wrappers for applying various transformations for data decomposition and cleaning.


Author: Ali Zaidi

Date: 03-JULY-2024
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from pynirs.hb_conv import subtract_ambient, calc_hb_concs, calc_toi


def svd(data: np.ndarray, full_matrices: bool=False) -> list[np.ndarray]:
    """
    Perform Singluar Value Decomposition on NIRS data

    Arguments
    ---------
        data: np.ndarray (shape=(channels, observations))
            The data to be decomposed.

        full_matrices: boolean (default=False)
            Whether to calculate the full U and V matrices (which is rarely a good idea!)

    Returns
    -------
        (U,S,V): tuple
            The matrices resulting from the SVD as a tuple. For more info, lookup numpy.linalg.svd.
    """

    # First check whether the data is the correct type
    if type(data) is not np.ndarray:
        raise Exception('data must be a numpy array')
    
    # if someone accidentally passes data with observations as rows, transpose
    if np.argmax(data.shape) == 0:
        data = data.T

    # Now do the SVD and return the result
    return np.linalg.svd(data, full_matrices=full_matrices)
    

def ica(data):
    "Perform an Independent Component Analysis based decomposition on raw NIRS"
    pass

def svd_clean(data, return_noise=True):
    "Clean raw NIRS 5WL data with SVD, return cleaned raw data"

    if data.shape[0] < data.shape[1]:
        data = data.T

    raw_data = data
    raw_data_means = np.mean(raw_data, axis=0)
    raw_data_mc = raw_data - raw_data_means

    u,s,v = np.linalg.svd(raw_data_mc.T, full_matrices=False)
    
    noise = np.mean(u[:,0])*v[0,:]
    rec = u[:,1:-1] @ np.diag(s[1:-1]) @ v[1:-1,:]
    rec = rec.T
    clean_data = (rec - rec[0,:] + raw_data[0,:]).T

    if return_noise:
        return clean_data, noise
    

def clean_hbc_toi(data):
    "Takes in the 12 channels from raw NIRS data, returns Hb-concs and TOI"

    near, far = subtract_ambient(data.T)
    near_clean, _ = svd_clean(near)
    far_clean, _ = svd_clean(far)

    near_hb_clean, far_hb_clean = calc_hb_concs(near_clean, far_clean)
    toi_clean = calc_toi(near_clean, far_clean)

    return near_hb_clean, far_hb_clean, toi_clean

