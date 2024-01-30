# -*- coding:utf-8 -*-
"""
Created on Fri Jan. 19 13:21:09 2024
@author: PJS

** Parameters **
data_config: dataset configuration
train_config:
model:
model_name (str):
criterion:
optimizer:
scheduler:
"""
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(
            self,
            data_config,
            train_config,
            model,
            model_name: str,
            criterion=None,
            optimizer=None,
            scheduler=None,
    ):
        if criterion is None:
            criterion = nn.CrossEntropyLoss().to(train_config.device)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, betas=(0.9, 0.99))
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        self.data_config = data_config
        self.train_config = train_config
        self.model = model
        self.model_name = model_name
        self.device = train_config.device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, X, y, validation=False, validation_data=None, validation_label=None):
        if validation:
            if validation_data is None or validation_label is None:
                raise ValueError('dl=True requires both "validation_data" and "validation_label" to be specified')

        if validation:
            train_data, validation_data, train_label, validation_label = X, validation_data, y, validation_label
        else:
            train_data, validation_data, train_label, validation_label = train_test_split(
                X, y, test_size=0.2, random_state=777, shuffle=True, stratify=y
            )

        train_data = torch.Tensor(train_data)
        validation_data = torch.Tensor(validation_data)
        train_label = torch.LongTensor(train_label)
        validation_label = torch.LongTensor(validation_label)

        train = TensorDataset(train_data, train_label)
        loader_train = DataLoader(train, batch_size=self.train_config.train_batch_size, shuffle=True)

        validation = TensorDataset(validation_data, validation_label)
        loader_validation = DataLoader(validation, batch_size=self.train_config.test_batch_size, shuffle=True)

        metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.data_config.select_label).to(self.device)

        loss_list = []
        early_stop_counter = 0

        for epoch in range(self.train_config.epochs):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for x, y in loader_train:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                pred1 = pred.cpu()
                y1 = y.cpu()
                acc = metric(pred1, y1)
                train_loss += loss.item()
                train_acc += acc

            # self.scheduler.step()
            if epoch % 10 == 0:
                print(
                    f'Epoch: {epoch}, Train Accuracy: {train_acc / len(loader_train): .4f}, Train Loss: {train_loss / len(loader_train): .4f}'
                )

            self.model.eval()
            with torch.no_grad():
                test_acc = 0
                test_loss = 0
                for x, y in loader_validation:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    pred1 = pred.cpu()
                    y1 = y.cpu()
                    acc_ = metric(pred1, y1)
                    test_loss += loss.item()
                    test_acc += acc_
            t_loss = test_loss/len(loader_validation)
            loss_list.append(t_loss)
            best_loss = min(loss_list)
            if epoch % 10 == 0:
                print(
                    f'Epoch: {epoch}, Validation Accuracy: {test_acc / len(loader_validation): .4f}, Validation Loss: {t_loss: .4f}'
                )
                print('________________________________________________________________')

            if t_loss > best_loss:
                early_stop_counter += 1

            else:
                early_stop_counter = 0
                torch.save(self.model, f'{self.data_config.path}/Model/{self.model_name}/{self.data_config.sub_num}_best_model.pt')

            if early_stop_counter == self.train_config.early_stop:
                break

    def predicate(self, X):
        model = torch.load(f'{self.data_config.path}/Model/{self.model_name}/{self.data_config.sub_num}_best_model.pt', map_location=self.device)
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            model.eval()
            out = model(X)
            out = out.cpu()
            prob = F.softmax(out, dim=1)
            pred = torch.argmax(out, dim=1)
        return pred, prob
