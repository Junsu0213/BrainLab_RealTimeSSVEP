# -*- coding:utf-8 -*-


class DeepConvNetConfig:
    def __init__(
            self,
            data_config,
            model_name: str = 'DeepConvNet',
            dropout_rate: int = 0.5,
            pool_size: int = 3,
            out_channel: int = 25,
            kernel_length: int = None,
            block_out_channels=None,

    ):
        if kernel_length is None:
            kernel_length = data_config.epoch_len * 2
        if block_out_channels is None:
            block_out_channels = [out_channel, out_channel * 2, out_channel * 4, out_channel * 8]

        self.data_config = data_config
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.out_channel = out_channel
        self.block_out_channels = block_out_channels
        self.kernel_length = kernel_length
        self.num_electrodes = len(data_config.ch_list)
        self.input_time_length = data_config.epoch_len
        self.sampling_rate = data_config.sfreq
        self.num_classes = data_config.select_label


if __name__ == '__main__':
    from Config.data_config import SSVEPDataConfig
    data_config = SSVEPDataConfig()
    model_config = DeepConvNetConfig(data_config=data_config)
    print(model_config)
