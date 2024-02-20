# -*- coding:utf-8 -*-
"""
 Created on Sat. Dec. 23 23:50:42 2023
 @author: Jun-su Park

 ** Epoch annotations **

 print(list(event_id.keys()))
 ssvep event keys()           : ['4.62 Hz', '5.45 Hz', '6.67 Hz', '8.57 Hz', '12 Hz', '20 Hz']

 print(epoch_data.event_id)
 ssvep event dict             : {'4.62 Hz': 13, '5.45 Hz': 23, '6.67 Hz': 33, '8.57 Hz': 43, '12 Hz': 53, '20 Hz': 63}

 print(epoch_data.ch_names)
 channel list                 : ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'P7', 'PO9', 'O9', 'Iz', 'O10', 'PO10', 'P8']


 print(int(epoch_data.info['sfreq']))
 sampling frequency integer   : 125 Hz

 Input:
   data                       : epoch eeg data
   event_id (ssvep)           : event id list
                                ['4.62 Hz', '5.45 Hz', '6.67 Hz', '8.57 Hz', '12 Hz', '20 Hz']

 Output:
   X                          : epoch eeg data concatenate
                                (# of trials, # of channels, Data length [sample])
   y                          : epoch data label
                                (# of trials)
 """
import joblib
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from mne.decoding import Scaler

from Model.Trainer.model_trainer import ModelTrainer


class ModelEvaluation(object):
    def __init__(self, data_config):
        self.data_config = data_config
        self.sub_num = data_config.sub_num
        self.path = data_config.path
        self.random_seed = data_config.random_seed
        self.event_id = data_config.event_id
        self.epoch_len = float(data_config.epoch_len)
        self.data_len = data_config.data_len
        self.sfreq = data_config.sfreq
        self.ch_select = data_config.ch_select
        self.norm = data_config.norm

    def train_test_split(self, test_size=0.2):
        # Load epoch data
        epoch_data = joblib.load(fr"{self.path}\Database\Epoch_data\S{self.sub_num}_epoch_data.pkl")

        # Make dataset
        X, y = self.make_dataset(event_id=self.event_id, epoch_data=epoch_data, data_len=self.data_len, sfreq=self.sfreq, norm=self.norm)

        # Channel selection (occipital lobe)
        if self.ch_select is True and self.data_config.__class__.__name__ == 'BrainLabSSVEPDataConfig':
            X = X[:, 10:-1, :]
        if self.ch_select is True and self.data_config.__class__.__name__ == 'OpenBMISSVEPDataConfig':
            X = X[:, 27:32, :]

        # Data split (stratify)
        train_data, test_data, train_label, test_label = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, shuffle=True, stratify=y
        )

        return train_data, test_data, train_label, test_label, self.event_id, X, y

    # K-folds cross validation
    def ml_cross_validation(self, load_model, n_splits=5, dl=False, train_config=None, model_name=None):
        if dl:
            if train_config is None or model_name is None:
                raise ValueError('dl=True requires both "train_config" and "model_name" to be specified')

        # Open pickle file
        epoch_data = joblib.load(fr"{self.path}\Database\Epoch_data\S{self.sub_num}_epoch_data.pkl")

        # Number of classes
        num_classes = len(self.event_id)

        # Make dataset
        X, y = self.make_dataset(event_id=self.event_id, epoch_data=epoch_data, data_len=self.data_len, sfreq=self.sfreq, norm=self.norm)

        # Channel selection (occipital lobe)
        if self.ch_select is True and self.data_config.__class__.__name__ == 'BrainLabSSVEPDataConfig':
            X = X[:, 10:-1, :]
        if self.ch_select is True and self.data_config.__class__.__name__ == 'OpenBMISSVEPDataConfig':
            X = X[:, 27:32, :]

        # Stratified K fold cross validation
        skf = StratifiedKFold(n_splits=n_splits, random_state=self.random_seed, shuffle=True)
        skf.get_n_splits(X, y)

        results = []
        columns = ['Accuracy', 'ITR']
        index = [rf'Fold {i+1}' for i in range(n_splits)]

        # Initialize the confusion matrix
        total_confusion_matrix = np.zeros((num_classes, num_classes))

        # Cross validation
        for train_index, test_index in skf.split(X, y):
            train_data, test_data = X[train_index, :, :], X[test_index, :, :]
            train_label, test_label = y[train_index], y[test_index]

            if dl:  # Deep learning
                # Reset model weight
                model_instance = load_model.apply(self.reset_weights)
                model = ModelTrainer(
                    data_config=self.data_config, train_config=train_config,
                    model=model_instance, model_name=model_name
                )
                model.train(X=train_data, y=train_label)
                pred, prob = model.predicate(X=test_data)
            else:  # Machine learning
                model = load_model
                model.train(x=train_data, y=train_label)
                pred, prob = model.predicate(x=test_data)

            # accuracy score
            acc = accuracy_score(y_true=test_label, y_pred=pred)

            # information transfer rate
            itr = self.calculate_itr(n=num_classes, p=acc, t=self.epoch_len)

            # result = [accuracy, itr]
            result = [acc, itr]

            # append K-fold result
            results.append(result)

            # confusion matrix update
            total_confusion_matrix += confusion_matrix(test_label, pred)

        #  create dataframe (K-fold result)
        df = pd.DataFrame(data=results, index=index, columns=columns)

        # Calculate the mean and standard deviation
        mean_values = df.mean()
        std_values = df.std()

        # Add mean and standard deviation as 'Mean' and 'Std' rows
        df.loc['Mean'] = mean_values
        df.loc['Std'] = std_values

        # Calculate the probability confusion matrix
        prob_confusion_matrix = total_confusion_matrix / total_confusion_matrix.sum(axis=1, keepdims=True)
        return df, prob_confusion_matrix, total_confusion_matrix

    # Leave one subject out (LOSO) cross validation
    def ml_loso(self, load_model, subject_id, dl=False, train_config=None, model_name=None):
        if dl:
            if train_config is None or model_name is None:
                raise ValueError('dl=True requires both "train_config" and "model_name" to be specified')

        data = []
        label = []
        # Load epoch dataset
        for sub in subject_id:
            # Load epoch data
            epoch_data = joblib.load(fr"{self.path}\Database\Epoch_data\S{sub}_epoch_data.pkl")

            # Make dataset
            X, y = self.make_dataset(event_id=self.event_id, epoch_data=epoch_data, data_len=self.data_len,
                                     sfreq=self.sfreq, norm=self.norm)

            # Channel selection (occipital lobe)
            if self.ch_select is True and self.data_config.__class__.__name__ == 'BrainLabSSVEPDataConfig':
                X = X[:, 10:-1, :]
            if self.ch_select is True and self.data_config.__class__.__name__ == 'OpenBMISSVEPDataConfig':
                X = X[:, 27:32, :]

            data.append(X)
            label.append(y)

        data = np.array(data)
        label = np.array(label)

        # Number of subjects
        sub_len = len(subject_id)
        results = []
        columns = ['Accuracy', 'ITR']
        index = [f'S{i+2:02}' for i in range(sub_len)]
        # Number of classes
        num_classes = len(self.event_id)
        # Initialize the confusion matrix
        total_confusion_matrix = np.zeros((num_classes, num_classes))

        for i in range(sub_len):
            test_data = data[i, :, :, :]
            test_label = label[i, :]

            # Test data remove
            train_data = np.delete(data, i, axis=0)
            # Train data concatenate
            train_data = train_data.reshape(
                (
                    (sub_len-1) * test_label.shape[0],
                    len(self.data_config.ch_list),
                    int(self.data_len * self.sfreq)
                )
            )
            # Test label remove
            train_label = np.delete(label, i, axis=0)
            # Train label concatenate
            train_label = train_label.reshape((-1))

            if dl:  # Deep learning
                # Reset model weight
                model_instance = load_model.apply(self.reset_weights)
                model = ModelTrainer(
                    data_config=self.data_config, train_config=train_config,
                    model=model_instance, model_name=model_name
                )
                model.train(X=train_data, y=train_label)
                pred, prob = model.predicate(X=test_data)
            else:  # Machine learning
                model = load_model
                model.train(x=train_data, y=train_label)
                pred, prob = model.predicate(x=test_data)

            # accuracy score
            acc = accuracy_score(y_true=test_label, y_pred=pred)

            # information transfer rate
            itr = self.calculate_itr(n=num_classes, p=acc, t=self.epoch_len)

            # result = [accuracy, itr]
            result = [acc, itr]

            # append K-fold result
            results.append(result)

            # confusion matrix update
            total_confusion_matrix += confusion_matrix(test_label, pred)

        #  create dataframe (K-fold result)
        df = pd.DataFrame(data=results, index=index, columns=columns)

        # Calculate the mean and standard deviation
        mean_values = df.mean()
        std_values = df.std()

        # Add mean and standard deviation as 'Mean' and 'Std' rows
        df.loc['Mean'] = mean_values
        df.loc['Std'] = std_values

        # Calculate the probability confusion matrix
        prob_confusion_matrix = total_confusion_matrix / total_confusion_matrix.sum(axis=1, keepdims=True)
        return df, prob_confusion_matrix, total_confusion_matrix

    @ staticmethod
    def make_dataset(event_id, epoch_data, data_len, sfreq, norm=False):
        # Make label
        label_num = 0
        y = []
        for event in event_id:
            epoch_data = epoch_data.resample(sfreq=sfreq)
            data = epoch_data[event].get_data()
            if norm is True:  # RobustScaler: 'median', StandardScaler: 'mean', 1e6: None
                data = Scaler(info=epoch_data.info, scalings='median').fit_transform(data)
            label = np.zeros(data.shape[0])
            label.fill(float(label_num))
            label_num += 1
            if event == event_id[0]:
                X = data
            else:
                X = np.concatenate((X, data), axis=0)
            y.extend(label)
        y = np.array(y)
        X = X[:, :, :data_len * sfreq]
        return X, y

    @staticmethod
    def calculate_itr(n, p, t):
        """
        Calculate the Information Transfer Rate (ITR) for a BCI system.

        Parameters:
        n (int): The number of distinct commands that can be chosen.
        p (float): The probability of the system correctly identifying a command.
        t (float): The average time in seconds for the system to recognize a command.

        Returns:
        float: The ITR measured in bits per second (bps).
        """
        if n == 1:  # If there is only one command, the ITR is zero.
            return 0

        # Make sure the probability is between 0 and 1
        p = max(0, min(p, 1))

        if p == 1:
            p = 0.99999999

        # Calculate the ITR using the provided formula
        itr = 60 * ((math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) / t)

        # Handle the case when p is 1 or 0 which would result in math domain error due to log2(0)
        if p == 0:
            itr = 0

        return itr

    @staticmethod
    def reset_weights(m):
        """
          Try resetting model weights to avoid
          weight leakage.
        """
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()
