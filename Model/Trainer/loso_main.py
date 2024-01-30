# -*- coding:utf-8 -*-
from Config.data_config import SSVEPDataConfig
from Config.train_config import ModelTrainerConfig
from Config.model_config import DeepConvNetConfig
from Model.DeepConvNet.DeepConvNet_model import DeepConvNet
from Preprocessing.model_evaluation import ModelEvaluation
from Figure.confusion_matrix import con_mat_plot

# Subject parameter
subject_id = ['03', '04', '05', '06',
              '07', '08', '09', '10']
sub_num = 'LOSO'
data_length = 5
select_label = 4
ch_select = True

# Dataset Config
data_config = SSVEPDataConfig(
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

# Leave one subject out cross validation
df, con_mat, total_trials_mat = eval.ml_loso(
    load_model=model,
    subject_id=subject_id,
    dl=True,
    train_config=train_config,
    model_name=model_config.model_name
)

# Results save
df.to_csv(
    f'{data_config.path}/Results/{model_config.model_name}/S{data_config.sub_num}_{str(data_config.select_label)} class.csv')

# Confusion matrix save
con_mat_plot(con_mat=con_mat, total_trials_mat=total_trials_mat, model_name=model_config.model_name,
             data_config=data_config)
