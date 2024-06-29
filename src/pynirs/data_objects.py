# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:08:32 2024

@author: Ali Zaidi
"""

import pandas as pd
import numpy as np
from scipy import signal, fft, stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle


class Experiment:
    """
    A class for loading and manipulating timeseries data from a CSV file. Has methods to allow resampling, filtering, etc.
    """

    def __init__(self, filename, auto_resample=True, extract_ne=True, fs=100):
        self._log = []
        self.__log("Instantiating Experiment object")
        self.filename = filename
        self.fs = fs
        self.filter_params = None
        self.data, self.comments = self.load_file(filename)
        self._data = self.data.copy()
        if extract_ne:
            self.extract_NE()
        if auto_resample:
            self.resample(1)

    def __log(self, statement):
        self._log.append(statement)
        print(statement)

    def get_log(self):
        return self._log


    def load_file(self, filename: str, repace_data=True) -> list[pd.DataFrame]:
        self.__log(f"Loading {filename}")
        data = pd.read_csv(filename, low_memory=False)
        comments = data.pop('comments')
        data.pop('datetime')
        comments = comments[~comments.isna()].copy()
        return data, comments

    def extract_NE(self):
        tmp = self.comments.str.extract(r'(?P<event>NE.+)(?P<dose>\d\.\d{2}|Off)')
        tmp = tmp[~tmp.iloc[:,0].isna()]

        tmp.iloc[-1,0] = 'NE Off'
        tmp.iloc[-1,-1] = 0
        ne_dose = np.zeros((self.data.shape[0],))
        for i in tmp.index.to_numpy():
            ne_dose[i:] = tmp['dose'][i]
        self.data['NE dose'] = ne_dose


    def resample(self, fs_new):
        self.__log(f"Resampling from {self.fs} to {fs_new} Hz")
        nsamples = self.data.shape[0] * fs_new // self.fs
        d_np = self.data.to_numpy()
        d_r = signal.resample(d_np, nsamples, axis=0)
        df_r = pd.DataFrame(d_r, columns=self.data.columns)
        ts = np.arange(0, d_r.shape[0]/fs_new, 1/fs_new)
        df_r['ts'] = ts
        self.data = df_r
        self._comments = self.comments.copy()

        _comments = self.comments.copy()
        _comments.index = _comments.index.to_numpy() * fs_new // self.fs
        self.comments = _comments
        self.fs = fs_new
        self.extract_NE()

    def get_fft(self, n=2048):
        self.__log(f"Generating FFT with fs={self.fs}")
        fx = fft.fft(self.data.iloc[:,1:].to_numpy(), n, axis=0)
        xf = fft.fftfreq(n, 1/self.fs)

        fft_df = pd.DataFrame(fx[:len(xf)//2,:], columns=self.data.columns[1:])
        fft_df['f'] = xf[:len(xf)//2]
        self.fft = fft_df

    def butter(self, order=2, wn=1/10, fs=1, btype='lowpass'):
        b, a = signal.butter(2, wn, fs=fs, btype=btype)
        self.filter_params = {'type': 'butter', 'order': order, 'Wn': wn, 'fs':fs, 'btype':btype, 'b': b, 'a':a}
        self.__log(f"Generating Butterworth filter. Params:{self.filter_params}")

    def filter(self):
        if self.filter_params is None:
            print("[ERROR] First call Experiment.butter() to generate a filter before filtering data.")
            return

        self.__log(f"Filtering data with {self.filter_params['btype']} {self.filter_params['Wn']} Hz")

        data = self.data.to_numpy()
        ts = data[:,0]

        if (self.filter_params['btype'] != 'lowpass'):
            data = data - data.mean(axis=0)

        b = self.filter_params['b']
        a = self.filter_params['a']

        filt_data = signal.filtfilt(b, a, data, axis=0)
        filt_df = pd.DataFrame(filt_data, columns=self.data.columns)
        filt_df['ts'] = ts
        self.data = filt_df
        self.extract_NE()

    def reset(self):
        self.__log("Reseting data to original values loaded from file.")
        self.data = self._data.copy()
        self.comments = self._comments.copy()
        self.filter_params=None
        self.fs = 100

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def __repr__(self):
        return f"Dataset loaded from {self.filename} with {self.data.shape[0]} samples and fs={self.fs}"

class Event:
    """
    A class for working with instances of physiological challenges such as hypoxias, map-manipulations, etc.
    Has methods that enable the estimation of feautures such as the area under the curve, cumsums and sigmoid-fits.
    """
    def __init__(self, dataset: Experiment = None, start=0, stop=-1, pre=0, post=0, detrend=False):
        print("Instantiating Event object")
        self.fs = dataset.fs
        data = dataset.data.iloc[start-pre:stop+post,:].copy()
        data['ts'] = data['ts'] - data['ts'].iloc[0]
        data.set_index('ts')
        self.data = data
        self.duration = (stop - start)/self.fs
        self.calc_features()
        if detrend:
            self.detrend()



    def detrend(self):
            _d = signal.detrend(self.data.iloc.to_numpy(), axis=0, type='linear')
            self.data.iloc[:,1:] = _d.iloc[:,1:]


    def get_fft(self, n=2048):
        self.log(f"Generating FFT with fs={self.fs}")
        fx = fft.fft(self.data.iloc[:,1:].to_numpy(), n, axis=0)
        xf = fft.fftfreq(n, 1/self.fs)

        fft_df = pd.DataFrame(fx[:len(xf)//2,:], columns=self.data.columns[2:])
        fft_df['f'] = xf[:len(xf)//2]
        self.fft = fft_df


    def calc_features(self):
        z_data = (self.data - self.data.iloc[:10, :].mean(axis=0)).copy()
        f_sum = z_data.sum(axis=0).copy()
        f_cumsum =z_data.cumsum(axis=0).copy()
        self.features = {'data': z_data, 'sum': f_sum, 'cumsum': f_cumsum}
        

    def save(self, object_name):
        with open(object_name, "wb") as f:
            pickle.dump(self, f)

    def __repr__(self):
        return "Event object"

