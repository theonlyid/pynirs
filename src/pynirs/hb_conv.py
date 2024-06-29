"""
Python functions for the conversion of raw NIRS data to Hb-concs and TOI
"""

import numpy as np

#The coefficients for conversion of optical density to Hb-concentrations.
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

    return concs_near, concs_far


def calc_toi(near_pd_reads, far_pd_reads):
    """
    Calculates TOI using coefficients similar to the PLM.

    Function inputs:
    1. Ambient subtracted near PD reads
    2. Ambient subtracted far PD reads

    Function outputs:
        toi
    """
    # declare constants
    h = 0.00046  # Wavelength dependence of scattering (1/nm);
    d = np.array([1, 1.6])  # source-detector seperations
    wls = np.array([680, 730, 810, 850, 910])  # LED wavelengths in nm
    abs_offsets = np.array([0.62517, 0.81141, 0.82149, 0.81788, 0.67964])  # calibration coefficients
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
    """
    Get PD reads from Neurokey dataframe returned by NKY-utils.
    """
    d = np.empty((len(df), 12))
    d[:, :5] = df.iloc[:, 1:6].to_numpy()
    d[:, 5] = df.iloc[:, 11].to_numpy()
    d[:, 6:11] = df.iloc[:, 6:11].to_numpy()
    d[:, 11] = df.iloc[:, 12].to_numpy()
    return d

