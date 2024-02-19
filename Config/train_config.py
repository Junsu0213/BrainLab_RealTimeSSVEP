# -*- coding:utf-8 -*-


class ModelTrainerConfig:
    def __init__(
            self,
            lr: float = 0.001,
            train_batch_size: int = 128,
            test_batch_size: int = 64,
            epochs: int = 50,
            early_stop: int = 40,
            device: str = 'cuda'
    ):
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
