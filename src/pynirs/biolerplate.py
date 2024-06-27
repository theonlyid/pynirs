# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:08:32 2024

@author: azaidi02
"""


#%% Imports
import pandas as pd
import numpy as np
from scipy import signal, fft, stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle


class Experiment:

    def __init__(self, filename, auto_resample=True, extract_ne=True, fs=100):
        self._log = []
        self.__log("Instantiating Experiment object")
        self.filename = filename
        self.fs = fs
        self.filter_params = None
        self.data, self.comments = self.load_file()
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


    def load_file(self):
        self.__log(f"Loading {self.filename}")
        data = pd.read_csv(self.filename, low_memory=False)
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

    def __init__(self, dataset: Experiment = None, start=0, stop=-1, pre=0, post=0, detrend=False):
        print("Instantiating Event object")
        self.fs = dataset.fs
        data = dataset.data.iloc[start-pre:stop+post,:].copy()
        data['ts'] = data['ts'] - data['ts'].iloc[0]
        data.set_index('ts')
        self.data = data
        self.calc_features()
        if detrend:
            self.detrend()


    def detrend(self):
            _d = signal.detrend(self.data.iloc.to_numpy(), axis=0, type='linear')
            self.data.iloc = _d


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

    def fit_dist(self, col_name, plot=True, return_yhat=True):

        def pred(x: np.float64, a: np.float64, b:np.float64) -> np.ndarray:
            y = 1/(1 + np.exp(-(a*x + b)))
            return y
        
        x_vals = self.data['ts'].to_numpy()
        y_obs = (self.data[col_name] - self.data[col_name].iloc[:10].mean()).to_numpy()
        y_obs = np.cumsum(y_obs)/np.sum(y_obs)
        p_opt, p_var = curve_fit(pred, x_vals, y_obs, np.array([1, -len(x_vals)/2], dtype=np.float64), maxfev=1_000_000_000)
        y_hat = pred(x_vals, p_opt[0], p_opt[1])

        if plot:
            plt.figure()
            plt.plot(x_vals, y_obs)
            plt.plot(x_vals, y_hat)
            plt.grid()
            plt.title(col_name)
            plt.legend(['observed', 'fit'])
        
        if return_yhat:
            return p_opt, y_hat
        else:
            return p_opt

    def save(self, object_name):
        with open(object_name, "wb") as f:
            pickle.dump(self, f)


    def __repr__(self):
        return "Event object"


E_COEFFS = np.array([[0.3194, 2.5713],
                     [0.4383, 1.3029],
                     [0.9291, 0.7987],
                     [1.1596, 0.7861],
                     [1.3514, 0.8968]])


def subtract_ambient(pd_reads: np.array):
    near_vals = pd_reads[:5, :] - pd_reads[5, :]
    far_vals = pd_reads[6:11, :] - pd_reads[11, :]

    return near_vals, far_vals


def calc_hb_concs(near_pd_reads, far_pd_reads):
    # Declare constants
    DPF = 6  # differential pathlength factor (6 is default in PLM).
    wls = (680, 730, 810, 850, 910)  # wavelength of each LED in nm
    d = [1, 1.6]  # specify source-detector positions

    # load the extinction coefficients.
    # grab the extinction coefficients for the wls in the LED.
    e_coef = E_COEFFS

    # Step 1: calculate the relative changes in OD
    near_delta_ODs = -np.log10(near_pd_reads.T / near_pd_reads[:, 0])  # normalize PD reads to first sample
    far_delta_ODs = -np.log10(far_pd_reads.T / far_pd_reads[:, 0])

    # step 2: Scale the extinction coefficient wavelength by assumed pathlength
    A_near = e_coef * d[0] * DPF  # A is e_coef times by assumed pathlength
    A_far = e_coef * d[1] * DPF

    # step 3: calculate hb concentrations.
    concs_near = 1000 * np.linalg.pinv(A_near) @ near_delta_ODs.T
    concs_far = 1000 * np.linalg.pinv(A_far) @ far_delta_ODs.T

    # concs_near = np.transpose(concs_near) * 1000  # convert unit to uM
    #
    # concs_far = np.matmul(np.linalg.pinv(A_far), np.transpose(far_delta_ODs))
    # concs_far = np.transpose(concs_far) * 1000  # convert unit to uM

    return concs_near, concs_far


"""
This function calculates TOI using similar coefficients to the PLM.
This TOI function is preliminary

Function inputs:
 1. Ambient subtracted near PD reads
 2. Ambient subtracted far PD reads

Function outputs:
    toi
"""


def calc_toi(near_pd_reads, far_pd_reads):
    # declare constants
    h = 0.00046  # Wavelength dependence of scattering (1/nm);
    d = np.array([1, 1.6])  # source-detector seperations
    wls = np.array([680, 730, 810, 850, 910])  # LED wavelengths in nm
    abs_offsets = np.array([0.62517, 0.81141, 0.82149, 0.81788, 0.67964])  # calibration coefficients

    # load the extinction coefficients.
    # e_coef = scipy.io.loadmat('e_coef_cope.mat')
    # e_coef = np.asarray(e_coef['e_coef_cope'])
    # # grab the extinction coefficients for the wls in the LED.
    # idx = [];
    # for wl in wls:
    #     idx.append(int(np.where(e_coef[:, 0] == wl)[0]))
    # e_coef = e_coef[idx, 1:3]

    e_coef = E_COEFFS

    # Step 1: Estimate the optical density gradient.
    OD_grads = np.log10(near_pd_reads / far_pd_reads) / (d[1] - d[0])
    OD_grads = OD_grads.T + abs_offsets / (d[1] - d[0])

    # Step 2: estimate the effect of the wavelength dependence of scattering
    usp = 1 - h * wls

    # step 3: Estimate absorbance coefficient
    ua = 1 / (3 * usp) * (np.log(10) * OD_grads - 2 / np.mean(d)) ** 2

    # step 4: estimate hb concentrations
    hb_concs = np.matmul(np.linalg.pinv(e_coef), np.transpose(ua))
    hb_concs = np.transpose(hb_concs)

    # step 5: calculate TOI:
    toi = hb_concs[:, 0] / np.sum(hb_concs, 1) * 100

    return toi


def get_pd_reads(df):
    d = np.empty((len(df), 12))
    d[:, :5] = df.iloc[:, 1:6].to_numpy()
    d[:, 5] = df.iloc[:, 11].to_numpy()
    d[:, 6:11] = df.iloc[:, 6:11].to_numpy()
    d[:, 11] = df.iloc[:, 12].to_numpy()
    return d


def load_data():
    fn = 'sample_data_to_check_neurokey_calcs.csv'
    df = pd.read_csv(fn)
    return df


if __name__ == '__main__':
    df = load_data()
    d = get_pd_reads(df)
    near, far = subtract_ambient(d.T)
    hb_near, hb_far = calc_hb_concs(near, far)
    toi = calc_toi(near, far)
    # plt.subplots(3, 1, sharex=True)
    # plt.subplot(311)
    # plt.plot(hb_near.T)
    # plt.subplot(312)
    # plt.plot(hb_far.T)
    # plt.subplot(313)
    # plt.plot(toi)
    # plt.show()

    print((np.allclose(hb_near[0, :], df['Near O2hb']) & np.allclose(hb_near[1, :], df['Near HHb']) & np.allclose(
        hb_far[1, :], df['Far HHb']) & np.allclose(hb_far[0, :], df['Far O2Hb'])))


if __name__ == '__main__':
    e = Experiment('C:/Users/azaidi02/OneDrive - UBC/Projects/DARPA/data/nirs/Didi Seven Sx alignments/Pre SCI/combined_data_v2.csv')

# %%
