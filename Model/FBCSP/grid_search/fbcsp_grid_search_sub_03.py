# -*- coding:utf-8 -*-
from Config.data_config import SSVEPDataConfig
from Preprocessing.data_epoching import DataEpoching
from Model.FBCSP.FBCSP_model import FBCSP
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

# try:
#     if not os.path.exists(rf'{path}\Model\FBCSP\results\S{sub_num}'):
#         os.makedirs(rf'{path}\Model\FBCSP\results\S{sub_num}')
# except OSError:
#     print('Error: Creating directory.')
#
# n_component = np.arange(8, 17)
# n_select = np.arange(5, 22)
#
# grid_dict = {}
# for select_class in class_list:
#     config = SSVEPDataConfig(sub_num=sub_num, data_len=data_len, select_label=select_class)
#
#     components_list = []
#     for component in n_component:
#         select_list = []
#         for select in n_select:
#             model = FBCSP(sampling_rate=125, n_components=int(component), n_select=int(select))
#             eval = ModelEvaluation(config=config)
#             df, con_mat = eval.ml_cross_validation(load_model=model, n_splits=n_splits)
#             acc = df['Accuracy'][-2]
#             select_list.append(acc)
#         components_list.append(select_list)
#     grid_dict[select_class] = components_list
#
# joblib.dump(grid_dict, rf'{path}\Model\FBCSP\results\S{sub_num}\S{sub_num}_{str(data_len)}_fbcsp_grid_search_acc.pkl')
data = joblib.load(rf'{path}\Model\FBCSP\results\S{sub_num}\S{sub_num}_{str(data_len)}_fbcsp_grid_search_acc.pkl')

for i in class_list:
    acc = np.array(data[i])*100
    component_label = [rf'com{i+8}' for i in range(acc.shape[0])]
    select_label = [rf'sel{i+5}' for i in range(acc.shape[1])]

    heat_map = sns.heatmap(
        data=acc,
        xticklabels=select_label,
        yticklabels=component_label,
        cmap=plt.cm.Blues,
        annot=True,
        fmt=".0f",
        cbar=False
    )
    plt.title(rf"Heatmap of accuracy ({i} class, {str(data_len)} sec)", fontsize=15)
    plt.xlabel("n_select")
    plt.ylabel("n_components")
    plt.savefig(rf'{path}\Model\FBCSP\results\S{sub_num}\Grid Search ({i} class, {str(data_len)} sec).png', dpi=500)
    plt.show()
