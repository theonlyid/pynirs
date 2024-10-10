"""
This sub-module contains all the custom types used in pynirs.

Author: Ali Zaidi
Date: 27-June-2024

"""

import numpy as np
from typing import NamedTuple

class Param:
    """An object to store the parameters returned by the Sigmoid.fit() method.
    
    Attributes
    ----------
    means: list[float]
        The mean values of the slope and intercept, respectively

    cov: numpy.ndarray[float]
        The covariance matrix of the best-fit parameters. Variance of each parameter is on the diagonal. Use vars = np.diag(cov).

    rsquare: float
        The R-square value of the fit

    """
    
    def __init__(self, means: list[float] = None, cov: np.ndarray[float] = None, rsquare: float = None):
        self.means=means
        self.cov=cov
        self.rsquare=rsquare

    def __repr__(self) -> str:
         r = f"Params(means:{self.means}, R-square:{self.rsquare:0.2f})"

# TODO: implement TTestResult Tuple
# TTestResult = NamedTuple('TTestResult', ['T-statistic', 'p-value'])
