# -*- coding:utf-8 -*-
"""
 Created on Fri. Dec. 22 19:57:10 2023
 @author: Jun-su Park

 ** Epoch annotations **

 print(self.event_dict)
 ssvep event dict             : {13: "5.45Hz", 23: "6.67Hz", 33: "7.5Hz", 43: "8.57Hz", 53: "12Hz"}

 print(self.ch_list)
 channel list                 : ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4','P3', 'Pz', 'P4', 'P7', 'PO9', 'O9', 'Iz', 'O10', 'PO10', 'P8']

 Input:
   data                       : raw eeg data (# of channels, Data length [sample])
   sampling frequency         : 125 Hz
   channel list               : ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4','P3', 'Pz', 'P4', 'P7', 'PO9', 'O9', 'Iz', 'O10', 'PO10', 'P8']
   event_id (ssvep)           : event id dictionary
                                {'4.62 Hz': 13, '5.45 Hz': 23, '6.67 Hz': 33, '8.57 Hz': 43, '12 Hz': 53, '20 Hz': 63}

 Output:
   epoch_data                 : epoch eeg data
                                (# of trials, # of channels, Data length [sample])
 """

import glob
import os
import numpy as np
from brainflow.data_filter import DataFilter
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import joblib
from Config.data_config import SSVEPDataConfig
# import matplotlib
# matplotlib.use('Qt5Agg')


class DataEpoching(object):
    def __init__(self, config: SSVEPDataConfig):
        self.config = config
        self.sub_num = config.sub_num
        self.path = config.path
        self.epoch_len = config.epoch_len
        self.sfreq = config.sfreq
        self.ch_list = config.ch_list
        self.tri_mapping = config.tri_mapping
        self.event_dict = config.event_dict

    def data_epoch_parser(self, raw_plot=False):
        # make directory
        try:
            if not os.path.exists(rf'{self.path}\DataBase\Epoch_data'):
                os.mkdir(rf'{self.path}\DataBase\Epoch_data')
        except OSError:
            print('Error: Creating directory.')

        # epoch data load
        flist = glob.glob(rf'{self.path}\DataBase\Raw\S{self.sub_num}\*.csv')

        epoch_list = []
        for path in flist:

            data = DataFilter.read_file(path)

            tri_list = self.trigger_list(raw=data)

            epochs = self.epoch(raw=data, epoch_len=self.epoch_len, ch_list=self.ch_list, sfreq=self.sfreq, tri_list=tri_list,
                                tri_mapping=self.tri_mapping, event_dict=self.event_dict, raw_plot=raw_plot)
            epoch_list.append(epochs)

        # concatenate epoch data
        epochs_standard = mne.concatenate_epochs(epoch_list)

        # drop baseline
        epoch_data = epochs_standard.crop(tmin=0., tmax=5.)

        # save epoch data
        joblib.dump(epoch_data, rf'{self.path}\DataBase\Epoch_data\S{self.sub_num}_epoch_data.pkl')

    @staticmethod
    def epoch(raw, epoch_len, ch_list, sfreq, tri_list, tri_mapping, event_dict, raw_plot=False):
        # channel length
        ch_len = len(ch_list)

        # column, index remove
        raw = raw[1:ch_len+1, 1:]/1e6

        # filtering (1~50Hz BPF)
        eeg_info = mne.create_info(ch_names=ch_list, ch_types='eeg', sfreq=sfreq)
        data = mne.io.RawArray(raw, eeg_info)
        data.filter(l_freq=5, h_freq=40)
        # data.notch_filter(60)

        # # independent component analysis (artifact remove)
        # ica = ICA(n_components=16, max_iter='auto', random_state=7)
        # ica.fit(data)
        # ica.plot_sources(data)
        # plt.show()
        # ica.apply(data)

        # trigger
        annot_from_events = mne.annotations_from_events(events=tri_list, event_desc=tri_mapping, sfreq=sfreq)
        annot_from_events.duration = annot_from_events.duration + epoch_len + 0.5
        data.set_annotations(annot_from_events)

        # figure plot
        if raw_plot:
            data.plot(duration=10.7, start=10., highpass=3., lowpass=30.)
            plt.show()

        # epoch
        epoch = mne.Epochs(raw=data, events=tri_list, event_id=event_dict, tmin=-0.5, tmax=5.,
                           baseline=(-0.5, 0), preload=True)
        return epoch

    @staticmethod
    # epoch trigger list
    def trigger_list(raw):
        trigger = raw[-1, 1:]
        num = 0
        tri_list = []
        for i in trigger:
            i = int(i)
            if i == 0:
                pass
            else:
                tri = [num-63, 0, i]
                tri_list.append(tri)
            num += 1
        tri_list = np.array(tri_list)
        return tri_list


if __name__ == "__main__":

    # for i in range(3, 11):
    #     sub_num = rf'{i:02}'
    #     config = SSVEPDataConfig(sub_num=sub_num)
    #     DataEpoching(config=config).data_epoch_parser(raw_plot=False)

    config = SSVEPDataConfig(sub_num='100')
    DataEpoching(config=config).data_epoch_parser()
