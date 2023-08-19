# -*- coding: utf-8 -*- 
# @Time : 2022/7/22 15:39 
# @Author : xlf
# @File : analyse_dataset.py

import pandas as pd
import numpy as np
from sklearn import preprocessing
import glob
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_2d(df_dup):
    data_frame = pd.DataFrame()
    for i in range(0, df_dup.shape[0] - 59):
        is_anomaly = False
        mylist = []
        for j in range(i, i + 60):
            mylist.append(df_dup['value'].iat[j])
            if df_dup['is_anomaly'].iat[j] == 1:
                is_anomaly = True
        if is_anomaly:
            mylist.append(1)
        else:
            mylist.append(0)
        np_Array = np.array(mylist)
        mylist = np_Array.T
        data_frame = data_frame.append(pd.Series(mylist), ignore_index=True)
    return data_frame

def get_data():
    val_list = []
    path = r'../Dataset'
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        # 将数据中value为0的替换成NaN
        df = df.replace(0, np.nan)
        # 处理value那层数据，将0去除掉
        df = df.dropna(axis=0, how='any', subset=['value'])
        df.value = preprocessing.normalize([df.value]).T
        val=list(df['value'])
        val_list.extend(val)
    return val_list

dataset_conc=get_data()
lens=len(dataset_conc)
index_list=list(range(lens))

plt.plot(index_list,dataset_conc)
plt.show()













