# -*- coding:utf-8 -*-
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')


def con_mat_plot(con_mat, total_trials_mat, model_name, data_config):
    labels = data_config.event_id
    sub_num = data_config.sub_num
    select_class = data_config.select_label
    path = data_config.path
    data_len = data_config.data_len

    disp = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, colorbar=False)

    # Iterate over each cell and update the text to include probability and number of correct trials
    for i in range(con_mat.shape[0]):
        for j in range(con_mat.shape[1]):
            total_trials = int(total_trials_mat[i, j])
            text = f"{con_mat[i, j]:.2f}\n({total_trials})"
            disp.text_[i, j].set_text(text)
            disp.text_[i, j].set_fontsize(14)  # Adjust font size as needed

    plt.xlabel('True label', fontsize=15)
    plt.ylabel('Predicted label', fontsize=15)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)
    # plt.grid(b=None)
    plt.savefig(rf'{path}/Results/{model_name}/S{sub_num} Confusion matrix ({select_class} class, {data_len} sec).png', dpi=500)
    plt.savefig(rf'{path}/Results/{model_name}/S{sub_num} Confusion matrix ({select_class} class, {data_len} sec).eps', dpi=500, format='eps')
    plt.show()
