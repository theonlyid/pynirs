# -*- coding: utf-8 -*-
"""
Module for handling sigmoid fits and the meta-data associated with them.

author: Ali Zaidi
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from pynirs.nirs_types import Param

class Sigmoid:

    """
    Object that handles the sigmoid fits to cumsum data. Currently fits sigmoid curves bounded between 0 and 1.

    
    Attributes
    ----------
    x : np.ndarray[float]
        Array with timeseries data for interpreting slope and intercept.

    y : np.ndarray[float]
        Data to fit sigmoid curve to.
    
    yhat: np.ndarray[flaot]
        The predicted y-values based on the best-fit parameters.

    params: Params
        An object with the best fit parameters and their statistics. Type help(Params) for more info.

    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, fs = 1):
        """
        Instantiates an object with the relevant x and y data. Will automatically fit a sigmoid to the data.

        Parameters
        ----------
            x_data: np.ndarray
                The timeseries data for the x-axis

            y_data: np.ndarray
                The observed y-data that will be fit

            fs : float
                The sampling rate used to normalize the parameters to seconds.
        """

        self.x_data = x_data
        self.y_data = y_data - y_data[0]
        self.fs = fs
        self.params = self.fit(x_data=x_data, y_data=y_data, plot=True)
        self.yhat = self.predict(x_data=x_data, slope=self.params.means[0], intercept=self.params.means[1], scale=self.params.means[2])
        print(self)

    def fit(self, x_data: np.ndarray = None, y_data: np.ndarray = None, plot=False) -> Param:
        """
        Fit sigmoid curve to data. If no data is passed, it will use the x and y data stored in the object.

        Parameters
        ----------
            x_data: np.ndarray
                The timeseries data for the x-axis

            y_data: np.ndarray
                The observed y-data that will be fit

            plot: boolean (default = True)
                Flag to plot the data after the fit for diagnostics

        Returns
        -------
        params : Param (see nirs_types for parameter object definition)
            The parameters returned by the fit() method, along with their covariance matrix. Has the folling attributes.
                means (list[float]): the mean values of the parameters
                cov (np.ndarray): the covariance matrix of the paramters
                rsquare (float): the R-square for the fit
                yhat (np.ndarray): the y-values predicted from the fit params for the x_data provided
        """

        if x_data is None and y_data is None:
             x_data = self.x_data
             y_data = self.y_data
    
        y_obs = y_data
        # y_obs = np.cumsum(y_obs)/np.sum(y_obs)
        # y_obs = y_data
        p_opt, p_var = curve_fit(self.predict, x_data, y_obs, np.array([1, len(x_data)/2, 1], dtype=np.float64), maxfev=1_000_000_000)
        y_hat = self.predict(x_data, p_opt[0], p_opt[1], p_opt[2])
        rsquare = 1 - (np.var(y_obs - y_hat)/np.var(y_obs))

        if plot:
            plt.figure()
            plt.plot(x_data, y_obs)
            plt.plot(x_data, y_hat, "--")
            plt.grid()
            plt.title(f"Fit results: R^2={rsquare:0.2f}")
            plt.legend(['observed', 'fit'])

        return Param(p_opt, p_var, rsquare)

    def predict(self, x_data: np.ndarray, slope: float, intercept: float, scale: float = 1) -> np.ndarray:
        """
        Predict the y-values of a sigmoid curve for the x_data and params provided.

        Parameters
        ----------
            x_data: np.ndarray
                The x-values for which to calculate the y-values
            slope: float
                The slope of the sigmoid
            intercept: float
                the intercept of the sigmoid

        Returns
        -------
            yhat: np.ndarray
                The predicted values based on the x-data, slope and intercept
        
        """
        y = (1/(1 + np.exp(-slope*(x_data - intercept)))) * scale
        return y
    
    def __repr__(self):
            slope = self.params.means[0]
            intercept = self.params.means[1]
            repr = f"SigmoidFit(slope:{slope:0.2f}, intercept:{intercept:0.2f}, scale:{self.params.means[2]:0.2f})"
            return repr
    

    #TODO: Add t-test functionality for parameters
    def ttest(param1, param2):
         pass

if __name__ == "__main__":

    x = np.arange(-100, 100)
    slope = 0.1
    intercept = 50
    scale = 10
    y = (1/(1 + np.exp(-slope*(x-intercept)))) * scale

    sig = Sigmoid(x, y)
