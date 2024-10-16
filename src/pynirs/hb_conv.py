"""
Python functions for the conversion of raw NIRS data to Hb-concs and TOI
"""

import numpy as np
from decompositions import svd_clean

class HbConv():
    def __init__(self, data: np.ndarray = None, svd_clean: bool = True):
        """
        Class that handles conversion of raw PD reads to converted (and cleaned) Hb-concentration changes and TOI.

        Parameters
        ----------
            data: np.ndarray
                The raw data as a 12-channel matrix with each channel as a column vector
            svd_clean: bool
                Whether or not to apply the SVD cleaning on the data

        Attributes
        ----------
            observed: dict
                A dictionary of data with the raw and converted data as was recorded. 
                
                raw_data: np.ndarray (shape=(12,nobs))
                    The raw absorbance data

                hb_concs: np.ndarray
                The predicted values based on the x-data, slope and intercept
        
        """
       
        #The coefficients for conversion of optical density to Hb-concentrations.
        self.__params = self.__gen_params()
        
        # If data was passed during instantiation
        if data is not None:
            # Convert to hb-concs and toi
            abs_data = data
            hbc_near, hbc_far, toi = self.hbc_from_abs()
            obs = dict({"raw_abs":abs_data, "hbc_near": hbc_near, "hbc_far": hbc_far, "toi": toi})
            self.observed = obs

        if svd_clean:
            # Clean the data
            pass


    def __gen_params(self) -> dict:
        "Generate a dictionary of params used in calculating Hb-concs and TOI."
        e_coef = np.array([[0.3194, 2.5713],
                            [0.4383, 1.3029],
                            [0.9291, 0.7987],
                            [1.1596, 0.7861],
                            [1.3514, 0.8968]])
        
        d = np.array([1, 1.6], dtype=np.float16) # Source detector distance per channel
        wavelenths = (680, 730, 810, 850, 910) # Wavelengths
        dpf = 6 # Differential path length factor
    
        # Generate matrices for conversion to Hb-concentration changes
        A_near = e_coef * d[0] * dpf
        A_near_inv  = np.linalg.pinv(A_near)

        A_far = e_coef * d[1] * dpf
        A_far_inv  = np.linalg.pinv(A_far)

        params = {"e_coef": e_coef,
                  "wl": wavelenths,
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
        near_vals = pd_reads[:5, :] - pd_reads[5, :]
        far_vals = pd_reads[6:11, :] - pd_reads[11, :]
        return near_vals, far_vals


    def calc_hb_concs(self, near_pd_reads: np.ndarray, far_pd_reads:np.ndarray) -> list[np.ndarray, np.ndarray]:

        A_near_inv = self.__params["A_near_inv"]
        A_far_inv = self.__params["A_far_inv"]

        # calculate the relative changes in OD
        near_delta_ODs = -np.log10(near_pd_reads.T / near_pd_reads[:, 0])  # normalize PD reads to first sample
        far_delta_ODs = -np.log10(far_pd_reads.T / far_pd_reads[:, 0])

        # calculate hb concentrations.
        concs_near = (A_near_inv) @ near_delta_ODs.T
        concs_far = (A_far_inv) @ far_delta_ODs.T

        return concs_near, concs_far
    
    def calc_gradients(self, near_pd_reads: np.ndarray, far_pd_reads: np.ndarray) -> np.ndarray:
        d = self.__params["sd_distance"]
        abs_off = self.__params["abs_offsets"]
        OD_grads = np.log10(near_pd_reads / far_pd_reads) / (d[1] - d[0])
        OD_grads = OD_grads.T + abs_off / (d[1] - d[0])
        return OD_grads


    def calc_toi(self, near_pd_reads: np.ndarray, far_pd_reads: np.ndarray) -> np.ndarray:
        """
        Calculates TOI using coefficients similar to the PLM.

        Function inputs:
        1. Ambient subtracted near PD reads
        2. Ambient subtracted far PD reads

        Function outputs:
            toi
        """

        OD_grads = self.calc_gradients(near_pd_reads, far_pd_reads)

        h =4.6E-4  # Wavelength dependence of scattering (1/nm);
        usp = 1 - h * self.__params["wavelengths"]

        # step 3: Estimate absorbance coefficient
        ua = 1 / (3 * usp) * (np.log(10) * OD_grads - 2 / np.mean(self.__params["d"])) ** 2

        # step 4: estimate hb concentrations
        hb_concs = np.matmul(self.__params["e_coef"], np.transpose(ua))
        hb_concs = np.transpose(hb_concs)

        # step 5: calculate TOI:
        toi = hb_concs[:, 0] / np.sum(hb_concs, 1) * 100

        return toi


    def get_pd_reads(df) -> np.ndarray:
        """
        Get PD reads from Neurokey dataframe returned by NKY-utils.
        """
        d = np.empty((len(df), 12))
        d[:, :5] = df.iloc[:, 1:6].to_numpy()
        d[:, 5] = df.iloc[:, 11].to_numpy()
        d[:, 6:11] = df.iloc[:, 6:11].to_numpy()
        d[:, 11] = df.iloc[:, 12].to_numpy()
        return d

if __name__ == "__main__":
    hbc = HbConv()
    # print(hbc.get_params())
    hbc.test_params()

