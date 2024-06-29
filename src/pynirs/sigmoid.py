# -*- coding: utf-8 -*-
"""
@author: Ali Zaidi
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


class Sigmoid:
    """
    Object that handles the sigmoid fits to cumsum data. Currently fits sigmoid curves bounded between 0 and 1.

    
    Attributes
    ----------
    x : np.ndarray[float]
        Array with timeseries data for interpreting slope and intercept.

    y : np.ndarray[float]
        Data to fit sigmoid curve to.

    params : dict
        The parameters returned by the fit() method, along with their covariance matrix. Has the folling attributes.

            means (list[float]): the mean values of the parameters
            cov (np.ndarray): the covariance matrix of the paramters
            rsquare (float): the R-square for the fit
            yhat (np.ndarray): the y-values predicted from the fit params for the x_data provided

    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, fs = 1):
        """
        Instantiates an object with the relevant x and y data.

        Parameters
        ----------
            x_data: np.ndarray
                The timeseries data for the x-axis

            y_data: np.ndarray
                The observed y-data that will be fit

            fs : float
                The sampling rate used to normalize the parameters to seconds.
        """
        self.x = x_data
        self.y = y_data
        self.params = None
        self.yhat = None
        self.fs = fs

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, plot=True):
        """
        Fit sigmoid curve to data.

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
        params : dict
            The parameters returned by the fit() method, along with their covariance matrix. Has the folling attributes.
                means (list[float]): the mean values of the parameters
                cov (np.ndarray): the covariance matrix of the paramters
                rsquare (float): the R-square for the fit
                yhat (np.ndarray): the y-values predicted from the fit params for the x_data provided
        """

        y_obs = self.y - self.y.mean()
        y_obs = np.cumsum(y_obs)/np.sum(y_obs)
        p_opt, p_var = curve_fit(self.pred, self.x, y_obs, np.array([1, -len(self.x)/2], dtype=np.float64), maxfev=1_000_000_000)
        y_hat = self.predict(self.x, p_opt[0], p_opt[1])

        rsquare = 1 - (np.var(y_data - y_hat)/np.var(y_data))

        if plot:
            plt.figure()
            plt.plot(self.x, y_obs)
            plt.plot(self.x, y_hat)
            plt.grid()
            plt.title(f"Fit results: R^2={rsquare}")
            plt.legend(['observed', 'fit'])

        params = {"means": p_opt, "param_variance": p_var, "rsquare": rsquare, "yhat": y_hat}
        self.params = params
        return params

    def predict(self, x_data: np.ndarray, slope: float, intercept: float):
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
        y = 1/(1 + np.exp(-slope*(x_data - intercept)))
        return y
    
    def __repr__(self):
            slope = self.params["means"][0]
            intercept = self.params["means"][1]
            repr = f"SigmoidFit(slope:{slope:0.2f}, intercept:{intercept:0.2f})"
            return repr
    