# -*- coding:utf-8 -*-
"""
Created on Fri. Dec. 22 19:57:10 2023
@author: Jun-su Park
"""


class BrainLabSSVEPDataConfig:
    """
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


class OpenBMISSVEPDataConfig:
    """
    ** OpenBMI SSVEP dataset **
    S1-S54 (29 males, 25 females)

    channels (10 electrodes): ['P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'PO10', 'O1', 'Oz', 'O2']
    data length: 4 sec
    [Baseline: 0.0 sec, Stim: 4 sec, After stim: 0 sec]
    sfreq: 250 Hz
    target: 4
    session: 2
    trials: 25
    total trials: 200 (25 * 4 * 2)
    """
    def __init__(
            self,
            sub_num=None,
            path=r'D:/DataBase/SSVEP/OpenBMI',
            epoch_len=4,
            data_len=4,
            random_seed=777,
            tri_mapping=None,
            sfreq=None,
            event_dict=None,
            event_id=None,
            stim_freq_map=None,
            select_label=None,
            ch_select=False,
            norm=True,
    ):
        if sfreq is None:
            sfreq = 1000
        else:
            sfreq = sfreq
        if ch_select:
            ch_list = ['PO9', 'O1', 'Oz', 'O2', 'PO10']
        else:
            ch_list = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
                       'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
                       'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
                       'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1',
                       'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h',
                       'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']
        if event_dict is None:
            event_dict = {'5.45 Hz': 3, '6.67 Hz': 2,
                          '8.57 Hz': 1, '12 Hz': 0}
        if event_id is None:
            event_id = ['5.45 Hz', '6.67 Hz', '8.57 Hz', '12 Hz']

        if stim_freq_map is None:
            stim_freq_map = {0: 5.45, 1: 6.67, 2: 8.57, 3: 12}
        self.sub_num = sub_num
        self.path = path
        self.epoch_len = epoch_len
        self.data_len = data_len
        self.random_seed = random_seed
        self.tri_mapping = tri_mapping
        self.select_label = select_label
        self.sfreq = sfreq
        self.norm = norm
        self.ch_list = ch_list
        self.event_dict = event_dict
        self.stim_freq_map = stim_freq_map
        self.event_id = event_id
        self.ch_select = ch_select


class BetaSSVEPDataConfig:
    """"
      BETA dataset
      S1-S70 (42 males, 28 females)
      channels: ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
                 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
                 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
                 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ',
                 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
                 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
                 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
      data length: 3, 4 sec
      [Before stim onset: 0.5 sec, Stim: 2 sec (S1-S15) and 3 sec (S16-S70), After stim: 0.5 sec]
      sfreq: 250 Hz
      target:40
      block:4
      total trials: 160
      """
    def __init__(
            self,
            sub_num=None,
            epoch_len=3,
            data_len=3,
            random_seed=777,
            sfreq=None,
            event_dict=None,
            resample=False,
            ch_select=False,
            norm=True,
    ):
        if sfreq is None:
            sfreq = 250
        else:
            resample = True

        if ch_select:
            ch_list = ['O1', 'OZ', 'O2']
        else:
            ch_list = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
                       'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
                       'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
                       'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ',
                       'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
                       'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
                       'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
        if event_dict is None:
            event_dict = {"8.6": 0, "8.8": 1, "9.0": 2, "9.2": 3, "9.4": 4, "9.6": 5, "9.8": 6, "10.0": 7,
                          "10.2": 8, "10.4": 9, "10.6": 10, "10.8": 11, "11.0": 12, "11.2": 13, "11.4": 14,
                          "11.6": 15, "11.8": 16, "12.0": 17, "12.2": 18, "12.4": 19, "12.6": 20, "12.8": 21,
                          "13.0": 22, "13.2": 23, "13.4": 24, "13.6": 25, "13.8": 26, "14.0": 27, "14.2": 28,
                          "14.4": 29, "14.6": 30, "14.8": 31, "15.0": 32, "15.2": 33, "15.4": 34, "15.6": 35,
                          "15.8": 36, "8.0": 37, "8.2": 38, "8.4": 39}
        self.sub_num = sub_num
        self.epoch_len = epoch_len
        self.data_len = data_len
        self.sfreq = sfreq
        self.random_seed = random_seed
        self.ch_list = ch_list
        self.event_dict = event_dict
        self.norm = norm
        self.resample = resample


if __name__ == '__main__':
    config = BrainLabSSVEPDataConfig()
    print(config.event_id)
