# -*- coding:utf-8 -*-
"""
Created on Fri. Dec. 22 19:57:10 2023
@author: Jun-su Park

** BrainLab SSVEP dataset **
S03-S10 (5 males, 3 females)

channels: [POz, PO3, PO4, PO5, PO6, Oz, O1, O2]
data length: 5 sec
[Baseline: 1.5 sec, Stim: 5 sec, After stim: 4 sec]
sfreq: 125 Hz
target: 6
block: 16 (1 block: 30 trials)
session: 4 (1 session: 4 block)
total trials: 480
"""


class SSVEPDataConfig:
    def __init__(
            self,
            sub_num=None,
            path=r'A:/BrainLab_RealTimeSSVEP',
            epoch_len=5,
            data_len=5,
            sfreq=125,
            random_seed=777,
            tri_mapping=None,
            event_dict=None,
            select_label=None,
            ch_select=False,
            norm=True,
    ):
        if ch_select is True:
            ch_list = ['P3', 'Pz', 'P4',
                       'P7', 'PO9', 'O9',
                       'Iz', 'O10', 'PO10', 'P8']
        else:
            ch_list = ['F3', 'Fz', 'F4',
                       'C3', 'Cz', 'C4',
                       'P3', 'Pz', 'P4',
                       'P7', 'PO9', 'O9',
                       'Iz', 'O10', 'PO10', 'P8']

        if tri_mapping is None:
            tri_mapping = {13: '4.62 Hz', 23: '5.45 Hz', 33: '6.67 Hz',
                           43: '8.57 Hz', 53: '12 Hz', 63: '20 Hz'}

        if event_dict is None:
            event_dict = {'4.62 Hz': 13, '5.45 Hz': 23, '6.67 Hz': 33,
                          '8.57 Hz': 43, '12 Hz': 53, '20 Hz': 63}

        if select_label is 2:
            event_id = ['4.62 Hz', '5.45 Hz']
            stim_freq_map = {0: 4.62, 1: 5.45}
        elif select_label is 3:
            event_id = ['4.62 Hz', '5.45 Hz', '6.67 Hz']
            stim_freq_map = {0: 4.62, 1: 5.45, 2: 6.67}
        elif select_label is 4:
            event_id = ['5.45 Hz', '6.67 Hz', '8.57 Hz', '12 Hz']
            stim_freq_map = {0: 5.45, 1: 6.67, 2: 8.57, 3: 12}
        elif select_label is 5:
            event_id = ['4.62 Hz', '5.45 Hz', '6.67 Hz', '8.57 Hz', '12 Hz']
            stim_freq_map = {0: 4.62, 1: 5.45, 2: 6.67, 3: 8.57, 4: 12}
        else:
            event_id = ['4.62 Hz', '5.45 Hz', '6.67 Hz', '8.57 Hz', '12 Hz', '20 Hz']
            stim_freq_map = {0: 4.62, 1: 5.45, 2: 6.67, 3: 8.57, 4: 12, 5: 20}

        self.sub_num = sub_num
        self.path = path
        self.epoch_len = epoch_len
        self.sfreq = sfreq
        self.random_seed = random_seed
        self.ch_list = ch_list
        self.tri_mapping = tri_mapping
        self.event_dict = event_dict
        self.select_label = select_label
        self.event_id = event_id
        self.stim_freq_map = stim_freq_map
        self.data_len = data_len
        self.ch_select = ch_select
        self.norm = norm


if __name__ == '__main__':
    config = SSVEPDataConfig()
    print(config.event_id)
