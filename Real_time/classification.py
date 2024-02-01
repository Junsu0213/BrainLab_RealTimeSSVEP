# -*- coding:utf-8 -*-
import sys
sys.path.append('C:\\Users\\Brainlab\\Desktop\\real_time')
import numpy as np
import mne
import time
from sklearn.preprocessing import StandardScaler
from config.data_config import EpochConfig
import joblib
import matplotlib.pyplot as plt


class real_time_classification(object):
    def __init__(self, config: EpochConfig):
        self.config = config
        self.path = config.path
        self.sub_name = config.sub_name
        self.ch_list = config.ch_list
        self.sfreq = config.sfreq

    def ssvep_model(self, data, class_num):
        # ssvep model load
        model = joblib.load(rf'{self.path}\model_save\fbcsp_ssvep_{str(class_num)}_class.pkl')

        # preprocessing: data shape (9, 626)
        data = self.preprocessing(raw=data, ch_list=self.ch_list, sfreq=self.sfreq)

        # sliding window data augmentation
        x = self.sliding_window_augmentation(data=data)

        # pre-trained model
        _, prob = model.predicate(x)

        # Soft Voting
        out = self.soft_voting(prob)
        return out

    def mi_model(self, data, class_num):
        # ssvep model load
        model = joblib.load(rf'{self.path}\model_\model_save\fbcsp_mi_{str(class_num)}_class.pkl')

        # preprocessing: data shape (9, 501)
        data = self.preprocessing(raw=data, ch_list=self.ch_list, sfreq=self.sfreq)

        # sliding window data augmentation
        x = self.sliding_window_augmentation(data=data)

        # pre-trained model
        _, prob = model.predicate(x) # sliding_window == False --> data

        # Soft Voting
        out = self.soft_voting(prob)
        return out

    @staticmethod
    def preprocessing(raw, ch_list, sfreq):
        # filtering (3~30Hz BPF)
        eeg_info = mne.create_info(ch_names=ch_list, ch_types='eeg', sfreq=sfreq)
        data = mne.io.RawArray(raw, eeg_info)
        data.filter(3., 30.)
        data = data.get_data()
        return data

    @staticmethod
    def sliding_window_augmentation(data, split_num=10):
        epoch_len = int(data.shape[1]/125) - 1
        for i in range(split_num):
            split_data = data[:, int((i*0.1)*125): int((i*0.1 + epoch_len)*125)+1]
            split_data = split_data[np.newaxis, :, :]
            if i == 0:
                concat_data = split_data
            else:
                concat_data = np.concatenate((concat_data, split_data), axis=0)
        return concat_data

    @staticmethod
    def soft_voting(prob_data):
        mean_prob = np.mean(prob_data, axis=0)
        pred = np.argmax(mean_prob)
        return pred


if __name__ == '__main__':
    config = EpochConfig(sub_name='jung_woo')
    model = real_time_classification(config=config)

    # SSVEP
    dummy_ssvep_data = np.random.sample((9, 626))
    start = time.time()
    out = model.ssvep_model(data=dummy_ssvep_data, class_num=4)
    print('SSVEP result:', out)
    end = time.time()
    print('SSVEP:', round(end-start, 3), 'sec')

    # MI
    dummy_mi_data = np.random.sample((9, 501))
    start = time.time()
    out = model.mi_model(data=dummy_mi_data, class_num=4)
    print('MI result:', out)
    end = time.time()
    print('MI:', round(end - start, 3), 'sec')