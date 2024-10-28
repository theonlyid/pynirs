"""
Module housing methods for calculating hb-concentration changes and toi from raw NIRS data.


Author: Ali Zaidi

Date: 05-OCT-2024
"""

import numpy as np
import pandas as pd

class HbConv:
    """
    Class that handles conversion of raw PD reads to converted (and cleaned) Hb-concentration changes and TOI.

    This module contains functions and wrappers for applying various transformations for data decomposition and cleaning.
    The HbConv object takes in a data array with raw absorbances (shape: 12-channels x n-observations) and perfoms ambeint correction,
    and calculates hb-concentration changes and TOI, along with an SVD based cleaning of the data.
    For Hb-concentration changes the SVD is applied to the near and far channel data.
    For TOI, the SVD is applied to both the ambient corrected absorbance data, or the optical density gradients. 


    Attributes
    ----------

        observed: dict
            A dictionary of data with the raw and converted data as was recorded. 
            
            raw_abs: np.ndarray (shape=(12, nobs))
                The raw absorbance data

            hbc_near: np.ndarray (shape=(2, nobs)
                The converted hb-concentration changes for the near channel.

            hbc_far: np.ndarray (shape=(2, nobs)
                The converted hb-concentration changes for the far channel.
            
            toi: np.ndarray
                The TOI calculated from the raw absorbance data (after ambient subtraction).

        cleaned: dict
            A dictionary of data with the raw and converted data after SVD cleaning.
            
            raw_abs: np.ndarray (shape=(10, nobs))
                The cleaned absorbance data (without the ambinet channels)

            hbc_near: np.ndarray (shape=(2, nobs)
                The converted hb-concentration changes for the near channel.

            hbc_far: np.ndarray (shape=(2, nobs)
                The converted hb-concentration changes for the far channel.
            
            toi_abs: np.ndarray (shape=(nobs,))
                The TOI calculated from the svd-cleaned near and far absorbance data.

            toi_ods: np.ndarray (shape=(nobs,))
                The TOI calculated from the svd-cleand OD-grads.
    """
    
    def __init__(self, data: np.ndarray = None, svd_clean: bool = True):
        """
        Instantiate Object with absorbance data and obtain hb-concentration and toi values for both observed and clean data.

        Arguments
        ---------
            data: np.ndarray (12 x nobs)
                The raw data as a 12-channel matrix with each channel as a row vector. 
            svd_clean: bool (default=True)
                Whether or not to apply the SVD cleaning on the data.
        
        Returns
        -------
            Hbconv: Object
                An instance of the class object.
        
        """
       
        #The coefficients for conversion of optical density to Hb-concentrations.
        self.__params = self.__gen_params()
        
        # If data was passed during instantiation
        if data is not None:
            # Convert to hb-concs and toi
            abs_data = data
            hbc_near, hbc_far, toi = self.hbc_from_abs(abs_data)
            obs = dict({"raw_abs":abs_data, "hbc_near": hbc_near, "hbc_far": hbc_far, "toi": toi})
            self.observed = obs

        if svd_clean:
            # Clean the data
            near, far = self.subtract_ambient(abs_data)
            abs_pre_clean = np.vstack((near, far))
            abs_post_clean, noise = self.svd_clean(abs_pre_clean)
            near_clean, far_clean = abs_post_clean[:5, :], abs_post_clean[5:, :]
            hbc_near_clean, hbc_far_clean = self.calc_hb_concs(near_clean, far_clean)
            toi_abs = self.calc_toi(near_clean, far_clean)
            toi_ods = self.calc_toi(near, far, svd_clean=True)
            clean = {"raw_abs": np.vstack((near_clean, far_clean)),
                     "hbc_near": hbc_near_clean,
                     "hbc_far": hbc_far_clean,
                     "toi_abs": toi_abs,
                     "toi_ods": toi_ods}
            self.cleaned = clean


    def __gen_params(self) -> dict:
        "Generate a dictionary of params used in calculating Hb-concs and TOI."
        e_coef = np.array([[0.3194, 2.5713],
                            [0.4383, 1.3029],
                            [0.9291, 0.7987],
                            [1.1596, 0.7861],
                            [1.3514, 0.8968]])
        
        d = np.array([1, 1.6], dtype=np.float16) # Source detector distance per channel
        wavelenths = np.array([680, 730, 810, 850, 910], dtype=np.float16) # Wavelengths
        dpf = 6 # Differential path length factor
    
        # Generate matrices for conversion to Hb-concentration changes
        A_near = e_coef * d[0] * dpf
        A_near_inv  = np.linalg.pinv(A_near)

        A_far = e_coef * d[1] * dpf
        A_far_inv  = np.linalg.pinv(A_far)

        params = {"e_coef": e_coef,
                  "wavelengths": wavelenths,
                  "dpf": dpf,
                  "sd_distance": d,
                  "A_near_inv": A_near_inv,
                  "A_far_inv": A_far_inv,
                  "abs_offsets": np.array([0.62517, 0.81141, 0.82149, 0.81788, 0.67964])}  # calibration coefficients
        
        return params
    
    
    def get_params(self) -> dict:
        "Returns the dictionary of parameters used for Hb-conc and TOI calculation"
        return self.__params

    
    def hbc_from_abs(self, abs_data: np.ndarray) -> list[np.ndarray, np.ndarray]:
        """
        Perform Singluar Value Decomposition on NIRS data

        Arguments
        ---------
            abs_data: np.ndarray
                The raw absorbance data (before ambient subtraction). Shape is nchan x nobs

        Returns
        -------
            near_hbc: np.ndarray (shape=(4, nobs))
                near hb-concentration changes

            far_hbc: np.ndarray (shape=(4, nobs))
                far hb-concentration changes

            toi: np.ndarray
                The calculated toi values
        """
        near, far = self.subtract_ambient(abs_data)
        near_hbc, far_hbc = self.calc_hb_concs(near, far)
        toi = self.calc_toi(near, far)
        return near_hbc, far_hbc, toi

    
    def subtract_ambient(self, pd_reads: np.ndarray) -> np.ndarray:
        """
        Subtracts ambient data from the raw absorbance data.

        Agruments
        ---------
            data: np.ndarray
                The raw data as a 12-channel matrix with each channel as a column vector
            svd_clean: bool
                Whether or not to apply the SVD cleaning on the data

        Returns
        -------
            observed: dict
                A dictionary of data with the raw and converted data as was recorded. 
                
                raw_data: np.ndarray (shape=(12,nobs))
                    The raw absorbance data

                hb_concs: np.ndarray
                    The predicted values based on the x-data, slope and intercept
                
                toi: np.ndarray
                    The TOI calculated from the raw absorbance data (after ambient subtraction)
        """
        near_vals = pd_reads[:5, :] - pd_reads[5, :]
        far_vals = pd_reads[6:11, :] - pd_reads[11, :]
        return near_vals, far_vals


    def calc_hb_concs(self, near_pd_reads: np.ndarray, far_pd_reads:np.ndarray) -> list[np.ndarray, np.ndarray]:
        """
        Convert raw PD reads to near and far Hb-concentration changes.

        Arguments
        ---------
            near_pd_reads: np.ndarray (5 x nobs)
                The raw data as a 12-channel matrix with each channel as a column vector
            far_pd_reads: np.ndarray (5 x nobs)
                Whether or not to apply the SVD cleaning on the data

        Returns
        -------
            conc_near: np.ndarray (2 x nobs)
                Array of near Hb-concentration changes with rows being O2Hb and HHb respectively. 
                
            conc_far: np.ndarray (2 x nobs)
                Array of far Hb-concentration changes with rows being O2Hb and HHb respectively. 
        """

        A_near_inv = self.__params["A_near_inv"]
        A_far_inv = self.__params["A_far_inv"]

        # calculate the relative changes in OD
        near_delta_ODs = -np.log10(near_pd_reads.T / near_pd_reads[:, 0])  # normalize PD reads to first sample
        far_delta_ODs = -np.log10(far_pd_reads.T / far_pd_reads[:, 0])

        # calculate hb concentrations.
        concs_near = 1000 * (A_near_inv) @ near_delta_ODs.T
        concs_far = 1000 * (A_far_inv) @ far_delta_ODs.T

        return concs_near, concs_far
    
    def calc_gradients(self, near_pd_reads: np.ndarray, far_pd_reads: np.ndarray) -> np.ndarray:
        """
        Calculate OD-gradients from ambient corrected near and far absorbance data

        Arguments
        ---------
            near_pd_reads: np.ndarray (5 x nobs)
                The raw data as a 12-channel matrix with each channel as a column vector
            far_pd_reads: np.ndarray (5 x nobs)
                Whether or not to apply the SVD cleaning on the data

        Returns
        -------
            od_grads: np.ndarray (5 x nobs)
                Array of log transformed, calibration corrected OD-gradients. 
                
        """

        d = self.__params["sd_distance"]
        abs_off = self.__params["abs_offsets"]
        OD_grads = np.log10(near_pd_reads / far_pd_reads) / (d[1] - d[0])
        OD_grads = OD_grads.T + abs_off / (d[1] - d[0])
        return OD_grads


    def calc_toi(self, near_pd_reads: np.ndarray, far_pd_reads: np.ndarray, svd_clean=False) -> np.ndarray:
        """
        Calculate TOI fron near and far absorbance data

        Arguments
        ---------
            near_pd_reads: np.ndarray (shape=(5, nobs))
                The near channel absorbance data
            far_pd_reads: np.ndarray (shape=(5, nobs))
                The far channel absorbance data
            svd_clean: bool (default=True)
                Whether to apply SVD cleaning to the absorbance data. If True, the SVD is applied simultaneously
                on the near and far data by stacking them vertically.

        Returns
        -------
            toi: np.ndarray (shape 5 x nobs)
                Array of log transformed, calibration corrected OD-gradients. 
             
        """
        OD_grads = self.calc_gradients(near_pd_reads, far_pd_reads)
        
        if svd_clean:
            OD_grads, _ = self.svd_clean(OD_grads)
            OD_grads = OD_grads.T

        h = 4.6E-4  # Wavelength dependence of scattering (1/nm);
        usp = 1 - h * self.__params["wavelengths"]

        # step 3: Estimate absorbance coefficient
        ua = 1 / (3 * usp) * (np.log(10) * OD_grads - 2 / np.mean(self.__params["sd_distance"])) ** 2

        # step 4: estimate hb concentrations
        hb_concs = np.linalg.pinv(self.__params["e_coef"]) @ ua.T
        hb_concs = np.transpose(hb_concs)

        # step 5: calculate TOI:
        toi = hb_concs[:, 0] / np.sum(hb_concs, 1) * 100

        return toi


    def get_pd_reads(df) -> np.ndarray:
        """
        Get PD reads from Neurokey dataframe returned by NKY-utils (for online use).
        
        Arguments
        ---------
            df: pd.DataFrame
                A dataframe with the NKY-derived absorbance values
        
        Returns
        -------
            data: np.ndarray (shape=(12, nobs))
                A numpy array with the absorbance data extraced from the dataframe
        """
        d = np.empty((len(df), 12))
        d[:, :5] = df.iloc[:, 1:6].to_numpy()
        d[:, 5] = df.iloc[:, 11].to_numpy()
        d[:, 6:11] = df.iloc[:, 6:11].to_numpy()
        d[:, 11] = df.iloc[:, 12].to_numpy()
        return d


    def svd(self, data: np.ndarray, full_matrices: bool=False) -> list[np.ndarray]:
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
    

    def svd_clean(self, data, print_uvar=True, return_noise=True) -> np.ndarray:
        """
        Clean absorbance data with SVD and return cleaned data (and noise, if return_noise is True).

        Arguments
        ---------
            data: np.ndarray (shape=(nchannels, nobs))
                The data to be decomposed with the SVD.
            
            print_uvar: bool (default=True)
                If True (default), will print the stdev of each column of U enabling quick assessment of cleaning. 
                If the first value is less than 0.1, cleaning is good.
            
            return_noise: bool (default=True)
                Whether or not to return the extracted component along with the cleaned data.
        
        Returns
        -------
            clean_data: np.ndarray (shape=(nchannels, nobs))
                The cleaned data with the dominant component removed.

            noise: np.ndarray (shape=(nobs,))
                If return_noise is True, will also return the component removed during cleaning
        
        """

        if data.shape[0] < data.shape[1]:
            data = data.T

        raw_data = data
        raw_data_means = np.mean(raw_data, axis=0)
        raw_data_mc = raw_data - raw_data_means

        u,s,v = np.linalg.svd(raw_data_mc.T, full_matrices=False)
        if print_uvar: print(np.std(u, axis=0))
        
        noise = np.mean(u[:,0])*v[0,:]
        rec = u[:,1:-1] @ np.diag(s[1:-1]) @ v[1:-1,:]
        rec = rec.T
        clean_data = (rec - rec[0,:] + raw_data[0,:]).T

        if return_noise:
            return clean_data, noise
        else:
            return clean_data
        

    def clean_hbc_toi(self, data):
        """
        Ambient subtract, clean and convert raw absorbance data.

        Arguments
        ---------
            data: np.ndarray (shape=(12, nobs))
                The raw data as a 12-channel matrix with each channel as a row vector. 
        
        Returns
        -------
            near_hb_clean: np.ndarray (shape=(2, nobs))
                The SVD-cleaned near hb-concentration changes.
            
            far_hb_clean: np.ndarray (shape=(2, nobs))
                The SVD-cleaned far hb-concentration changes.
            
            toi_clean: np.ndarray (shape=(nobs,))
                The TOI calculated from the SVD-cleaned near and far absorbances.
        
        """

        near, far = self.subtract_ambient(data)
        near_clean, _ = self.svd_clean(near)
        far_clean, _ = self.svd_clean(far)

        near_hb_clean, far_hb_clean = self.calc_hb_concs(near_clean, far_clean)
        toi_clean = self.calc_toi(near_clean, far_clean, svd_clean=False)

        return near_hb_clean, far_hb_clean, toi_clean



if __name__ == "__main__":
    import pickle
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    with open('src/pynirs/data/combined_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    cols = [f"c NIRS 1.channel[{i}].1" for i in range(12)]
    pd_reads = data[cols].to_numpy().T

    hb = HbConv(pd_reads)
    plt.subplots(3,1, sharex=True, sharey=True)
    plt.subplot(311)
    plt.plot(hb.observed["toi"])
    plt.title("TOI Observed")
    plt.subplot(312)
    plt.plot(hb.cleaned["toi_ods"])
    plt.title("TOI (OD-grad cleaned)")
    plt.subplot(313)
    plt.plot(hb.cleaned["toi_abs"])
    plt.title("TOI (Absorbance cleaned)")
    plt.show()
    print("done")

