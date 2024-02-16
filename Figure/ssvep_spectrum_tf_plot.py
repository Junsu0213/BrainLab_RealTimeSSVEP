# -*- coding:utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mne
from scipy.signal import spectrogram
from mne.channels import make_standard_montage
from Config.data_config import BrainLabSSVEPDataConfig


class FigurePlot(object):
    def __init__(self, config: SSVEPDataConfig):
        self.config = config
        self.sub_num = config.sub_num
        self.path = config.path
        self.epoch_len = float(config.epoch_len)

        """
         Created on Sat. Dec. 23 18:16:07 2023
         @author: Jun-su Park

         ** Epoch annotations **

         print(epoch_data.event_id)
         ssvep event id               : ['4.62 Hz', '5.45 Hz', '6.67 Hz', '8.57 Hz', '12 Hz', '20 Hz']

         print(epoch_data.ch_names)
         channel list                 : ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'P7', 'PO9', 'O9', 'Iz', 'O10', 'PO10', 'P8']

         print(int(epoch_data.info['sfreq']))
         sampling frequency integer   : 125 Hz

         Input:
           data                       : epoch eeg data
                                        (# of trials, # of channels, Data length [sample])

         Output:
           figure save                : SSVEP figure plot (PSD, SNR spectral)
         """

    def ssvep_psd_snr_spectrum(self):
        # make directory
        try:
            if not os.path.exists(rf'{self.path}\Figure\PSD_SNR\S{self.sub_num}'):
                os.makedirs(rf'{self.path}\Figure\PSD_SNR\S{self.sub_num}')
        except OSError:
            print('Error: Creating directory. ')

        fmax = 54

        # ssvep epoch data load
        epoch_data = joblib.load(fr"{self.path}\Database\Epoch_data\S{self.sub_num}_epoch_data.pkl")

        # montage (standard 1020 system)
        montage = make_standard_montage('standard_1020')
        epoch_data.set_montage(montage)

        event_id = epoch_data.event_id
        for event in event_id:

            hz_name = str(event).split('H')[0]
            # print(epoch_data[event].get_data().shape, epoch_data[event].info)
            method_kw_args = {
                'n_fft': 125*5
            }
            spectrum = epoch_data[event].compute_psd(method="welch", picks=['PO9', 'PO10'], **method_kw_args)
            psds, freqs = spectrum.get_data(return_freqs=True)

            snrs = self.snr_spectrum(psd=psds, noise_n_neighbor_freqs=5, noise_skip_neighbor_freqs=1)
            fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(7.5, 6))
            freq_range = range(
                np.where(np.floor(freqs) == 1.0)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0]
            )

            psds_plot = 10 * np.log10(psds)
            psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
            psds_std = psds_plot.std(axis=(0, 1))[freq_range]
            axes[0].plot(freqs[freq_range], psds_mean, color="b")
            axes[0].fill_between(
                freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
            )

            axes[0].set(
                title=rf"PSD and SNR spectrum [{hz_name} Hz]",
                ylim=[-141, -109],
            )
            axes[0].set_ylabel("PSD [dB]", fontsize=13)
            axes[0].title.set_fontsize(17)

            # SNR spectrum
            snr_mean = snrs.mean(axis=(0, 1))[freq_range]
            snr_std = snrs.std(axis=(0, 1))[freq_range]

            axes[1].plot(freqs[freq_range], snr_mean, color="r")
            axes[1].fill_between(
                freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
            )
            axes[1].set(
                ylim=[-.9, 3.5],
                xlim=[4, 48]
            )
            axes[1].set_ylabel("SNR", fontsize=13)
            axes[1].yaxis.set_label_coords(-0.085, 0.5)
            axes[1].set_xlabel("Frequency [Hz]", fontsize=13)

            # padding (remove white space)
            plt.tight_layout()

            # save png file
            plt.savefig(rf'{self.path}\Figure\PSD_SNR\S{self.sub_num}\{event}.png', dpi=500)

            # save eps file
            plt.savefig(rf'{self.path}\Figure\PSD_SNR\S{self.sub_num}\{event}.eps', dpi=500, format='eps')

    def snr_spectrum(self, psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
        """
        Reference: MNE-python (Frequency-tagging: Basic analysis of an SSVEP/vSSR dataset)
        [https://mne.tools/dev/auto_tutorials/time-freq/50_ssvep.html]

        Compute SNR spectrum from PSD spectrum using convolution.

        Parameters
        ----------
        psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
            Data object containing PSD values. Works with arrays as produced by
            MNE's PSD functions or channel/trial subsets.
        noise_n_neighbor_freqs : int
            Number of neighboring frequencies used to compute noise level.
            increment by one to add one frequency bin ON BOTH SIDES
        noise_skip_neighbor_freqs : int
            set this >=1 if you want to exclude the immediately neighboring
            frequency bins in noise level calculation

        Returns
        -------
        snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
            Array containing SNR for all epochs, channels, frequency bins.
            NaN for frequencies on the edges, that do not have enough neighbors on
            one side to calculate SNR.
        """
        # Construct a kernel that calculates the mean of the neighboring
        # frequencies
        averaging_kernel = np.concatenate(
            (
                np.ones(noise_n_neighbor_freqs),
                np.zeros(2 * noise_skip_neighbor_freqs + 1),
                np.ones(noise_n_neighbor_freqs),
            )
        )
        averaging_kernel /= averaging_kernel.sum()

        # Calculate the mean of the neighboring frequencies by convolving with the
        # averaging kernel.
        mean_noise = np.apply_along_axis(
            lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
        )

        # The mean is not defined on the edges so we will pad it with nas. The
        # padding needs to be done for the last dimension only so we set it to
        # (0, 0) for the other ones.
        edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
        pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
        mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

        return psd / mean_noise

    def ssvep_psd_tf_plot(self):
        # make directory
        try:
            if not os.path.exists(rf'{self.path}\Figure\PSD_SNR\S{self.sub_num}'):
                os.makedirs(rf'{self.path}\Figure\PSD_SNR\S{self.sub_num}')
        except OSError:
            print('Error: Creating directory. ')

        # ssvep epoch data load
        epoch_data = joblib.load(fr"{self.path}\Database\Epoch_data\S{self.sub_num}_epoch_data.pkl")

        # montage (standard 1020 system)
        montage = make_standard_montage('standard_1020')
        epoch_data.set_montage(montage)

        # time frequency analysis (parameter)
        frequencies = np.arange(4, 25, 0.5)
        n_cycles = frequencies / frequencies[0]

        event_id = epoch_data.event_id
        for event in event_id:

            # subplot
            fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))

            # Specifies the location of the subplot
            gs = axes[0, 0].get_gridspec()

            # remove first row
            axes[0, 0].remove()
            axes[0, 1].remove()
            axes[0, 2].remove()
            axes[0, 0] = fig.add_subplot(gs[0, :])

            # plot psd
            epoch_data[event].plot_psd(method='welch', show=False, fmin=4., fmax=48., dB=True,
                                       picks=['Iz', 'O9', 'O10'], ax=axes[0, 0])

            # mne
            # time frequency analysis
            power, itc = mne.time_frequency.tfr_morlet(epoch_data[event], freqs=frequencies, n_cycles=n_cycles,
                                                       return_itc=True)

            # plot time-frequency
            k = 0
            for i, ch in enumerate(epoch_data[event].ch_names):
                if ch == 'Iz' or ch == 'O9' or ch == 'O10':
                    # TF plotting
                    power.plot([i], baseline=(-0.2, 0), mode='logratio', vmin=-1, vmax=1, axes=axes[1, k],
                               title='Power spectral density {} {}'.format(event, ch), show=False)
                    # set subplot title
                    axes[1, k].set_title('{}'.format(ch))
                    k += 1

            # title overlap
            fig.subplots_adjust(top=0.9, hspace=0.8)

            # title
            plt.suptitle('{} SSVEP'.format(event), fontsize=20)  # y=0.88, x=0.5,

            # padding (remove white space)
            plt.tight_layout()

            # save png file
            plt.savefig(rf'{self.path}\Figure\PSD_SNR\S{self.sub_num}\TF_{event}.png', dpi=500)

            # save eps file
            plt.savefig(rf'{self.path}\Figure\PSD_SNR\S{self.sub_num}\TF_{event}.eps', dpi=500, format='eps')


if __name__ == '__main__':
    for i in range(2, 11):
        if i == 2:
            sub_num = '00'  # all subjects
        else:
            sub_num = rf'{i:02}'
        config = BrainLabSSVEPDataConfig(sub_num=sub_num)
        fig = FigurePlot(config=config)
        fig.ssvep_psd_snr_spectrum(), # fig.ssvep_psd_tf_plot()

    # config = SSVEPDataConfig(sub_num='99')
    # fig = FigurePlot(config=config)
    # fig.ssvep_psd_snr_spectrum()
