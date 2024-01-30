# -*- coding:utf-8 -*-
from Config.data_config import SSVEPDataConfig
from Preprocessing.data_epoching import DataEpoching
from Model.Riemann.riemann_model import XdawnRG
from Preprocessing.model_evaluation import ModelEvaluation
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

sub_num = '03'

# Epoch length
data_len = 5

# K-fold cross validation
n_splits = 5

# select class
class_list = [3, 4, 5, 6]

# Grid search
path = r"A:\BrainLab_RealTimeSSVEP"

try:
    if not os.path.exists(rf'{path}\Model\Riemann\results\S{sub_num}'):
        os.makedirs(rf'{path}\Model\Riemann\results\S{sub_num}')
except OSError:
    print('Error: Creating directory.')

n_filters = np.arange(8, 17)

grid_dict = {}
for select_class in class_list:
    config = SSVEPDataConfig(sub_num=sub_num, data_len=data_len, select_label=select_class)

    components_list = []
    for nfilter in n_filters:

        model = XdawnRG(nfilter=nfilter)
        eval = ModelEvaluation(config=config)
        df, con_mat = eval.ml_cross_validation(load_model=model, n_splits=n_splits)
        acc = df['Accuracy'][-2]
        components_list.append(acc)
    grid_dict[select_class] = components_list

joblib.dump(grid_dict, rf'{path}\Model\Riemann\results\S{sub_num}\S{sub_num}_{str(data_len)}_fbcsp_grid_search_acc.pkl')
data = joblib.load(rf'{path}\Model\Riemann\results\S{sub_num}\S{sub_num}_{str(data_len)}_fbcsp_grid_search_acc.pkl')

for i in class_list:
    acc = np.array(data[i])*100
    filter_label = [rf'filter {i+3}' for i in range(acc.shape[0])]

    heat_map = sns.heatmap(
        data=acc,
        xticklabels=filter_label,
        cmap=plt.cm.Blues,
        annot=True,
        fmt=".0f"
    )
    plt.title(rf"Heatmap of accuracy ({i} class, {str(data_len)} sec)", fontsize=15)
    plt.xlabel("n_filters")
    plt.savefig(rf'{path}\Model\FBCSP\results\S{sub_num}\Grid Search ({i} class, {str(data_len)} sec).png', dpi=500)
    # plt.show()
