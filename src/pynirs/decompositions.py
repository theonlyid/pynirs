"""
Module handling various data decompositions such as PCA, ICA and SVD for data cleaning and processing.


This module contains functions and wrappers for applying various transformations for data decomposition and cleaning.


Author: Ali Zaidi

Date: 03-JULY-2024
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA


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
