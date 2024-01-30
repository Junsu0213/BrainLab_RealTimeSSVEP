# -*- coding: utf-8 -*-
"""
Created on Thu Jan. 2 15:13:11 2024
@author: PJS

** Parameters **

"""
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from scipy.signal import butter, lfilter
from Config.data_config import SSVEPDataConfig
from Preprocessing.model_evaluation import ModelEvaluation
import numpy as np


class FBCCA(object):
    # Filter Bank Canonical Correlation Analysis
    def __init__(self, config: SSVEPDataConfig):
        self.config = config
        self.stim_freq_list = list(config.stim_freq_map.values())
        self.sfreq = config.sfreq
        self.low_cut = [3, 14, 19, 23, 28, 33]
        self.high_cut = [14, 18, 23, 27, 32, 35]
        self.bands = len(self.high_cut)

    def fbcca(self, X, y, idx_fb=None, num_harms=3):
        if idx_fb is None:
            idx_fb = self.bands
        elif (idx_fb < 0 or idx_fb < self.bands):
            raise ValueError(rf'Stats: filter bank: Invalid Input (The number of sub-band must be 0 <= idx_fb <= {self.bands}.)')

        # fb_coefs = np.power(np.arange(1, idx_fb + 1), (-1.25)) + 0.25
        # test
        fb_coefs = [1.4, 0.6, 0.2, 0.2, 0.2, 0.2]
        # print(fb_coefs)

        num_targs, _, num_smpls = X.shape  # 6 target (means 40 fre-phase combination that we want to predict)
        y_ref = self.cca_reference(self.stim_freq_list, self.sfreq, num_smpls, num_harms)
        # print(y_ref.shape)

        num_stim_freq = len(self.stim_freq_list)

        cca = CCA(n_components=1)  # initialize CCA

        # result matrix
        r = np.zeros((idx_fb, num_stim_freq))
        # print(r.shape)

        results = np.zeros(num_targs)

        for targ_i in range(num_targs):
            test_tmp = np.squeeze(X[targ_i, :, :])  # deal with one target a time
            fb_i = 0
            for band in range(idx_fb):
                # print(band)
                start_band, end_band = self.low_cut[band], self.high_cut[band]
                # print(start_band, end_band)

                x_filter = self.butter_bandpass_filter(
                    signal=test_tmp, low_cut=start_band, high_cut=end_band, fs=self.sfreq
                )
                for class_i in range(y_ref.shape[0]):
                    # print(class_i)
                    refdata = np.squeeze(y_ref[class_i, :, :])  # pick corresponding freq target reference signal
                    # print(x_filter.shape, refdata.shape)

                    test_C, ref_C = cca.fit_transform(x_filter.T, refdata.T)

                    # len(row) = len(observation), len(column) = variables of each observation
                    # number of rows should be the same, so need transpose here
                    # output is the highest correlation linear combination of two sets
                    r_tmp, _ = pearsonr(np.squeeze(test_C),
                                        np.squeeze(ref_C))  # return r and p_value, use np.squeeze to adapt the API
                    # print(r_tmp)
                    # print(class_i, fb_i)
                    r[fb_i, class_i] = r_tmp
                    # print(r.shape)
                    # exit()
                # exit()
                fb_i += 1
            # print(r.shape)
            rho = np.dot(fb_coefs, r)  # weighted sum of r from all different filter banks' result
            tau = np.argmax(rho)  # get maximum from the target as the final predict (get the index)
            # tau = np.argmax(r)
            # print(rho.shape)
            print(tau)
            results[targ_i] = tau  # index indicate the maximum(most possible) target
        print(results)
        return results

    @staticmethod
    def cca_reference(list_freqs, fs, num_smpls, num_harms=3):
        num_freqs = len(list_freqs)
        tidx = np.arange(1, num_smpls + 1) / fs  # time index

        y_ref = np.zeros((num_freqs, 2 * num_harms, num_smpls))
        for freq_i in range(num_freqs):
            tmp = []
            for harm_i in range(1, num_harms + 1):
                stim_freq = list_freqs[freq_i]  # in HZ

                # Sin and Cos
                tmp.extend([np.sin(2 * np.pi * tidx * harm_i * stim_freq),
                            np.cos(2 * np.pi * tidx * harm_i * stim_freq)])
            y_ref[freq_i] = tmp  # 2*num_harms because include both sin and cos

        return y_ref

    @staticmethod
    def butter_bandpass_filter(signal, low_cut, high_cut, fs, order=5):
        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = butter(order, [low, high], btype="band")
        y = lfilter(b, a, signal, axis=-1)
        return y
