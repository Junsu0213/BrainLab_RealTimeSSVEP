# -*- coding:utf-8 -*-
from Config.data_config import BrainLabSSVEPDataConfig
from Config.train_config import ModelTrainerConfig
from Config.model_config import DeepConvNetConfig
from Model.DeepConvNet.DeepConvNet_model import DeepConvNet
from Evaluation.model_evaluation import ModelEvaluation
from Figure.confusion_matrix import con_mat_plot

sub_list = ['03', '04', '05', '06',
            '07', '08', '09', '10']
for i in sub_list:
    sub_num = i

    # Subject parameter
    data_length = 5
    select_label = 4
    ch_select = True

    # K-folds cross validation
    n_split = 10

    # Dataset Config
    data_config = BrainLabSSVEPDataConfig(
        sub_num=sub_num,
        data_len=data_length,
        select_label=select_label,
        ch_select=ch_select
    )

    # Model training parameter
    train_config = ModelTrainerConfig()

    # Model parameter
    model_config = DeepConvNetConfig(data_config=data_config)

    # Load model
    model = DeepConvNet(model_config=model_config).to(train_config.device)
    eval = ModelEvaluation(data_config=data_config)

    # K-fold cross validation
    df, con_mat, total_trials_mat = eval.ml_cross_validation(
        load_model=model,
        dl=True,
        train_config=train_config,
        model_name=model_config.model_name,
        n_splits=n_split
    )

    # Results save
    df.to_csv(
        f'{data_config.path}/Results/{model_config.model_name}/S{data_config.sub_num}_{str(data_config.select_label)}class_{str(data_length)}sec.csv'
    )


    # Confusion matrix save
    con_mat_plot(con_mat=con_mat, total_trials_mat=total_trials_mat, model_name=model_config.model_name, data_config=data_config)
