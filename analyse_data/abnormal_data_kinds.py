# -*- coding: utf-8 -*-
# @Time : 2022/9/29 16:15
# @Author : xlf
# @File : abnormal_data_kinds.py
import pandas as pd
import numpy as np
from sklearn import preprocessing
import glob
import torch
from torch import nn
import torch.nn.functional as F
import  torch.optim as optim
from    matplotlib import pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3) # two axes on figure

def get_data(filename):
    data_list=[]
    flag_data=[]
    #path
    # filename = r'../Dataset/real_2.csv'
    # all_files = glob.glob(path + "/*.csv")
    # for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    # 将数据中value为0的替换成NaN
    df = df.replace(0, np.nan)
    # 处理value那层数据，将0去除掉
    df = df.dropna(axis=0, how='any', subset=['value'])
    value=df['value'].to_list()
    flag=df['is_anomaly'].to_list()
    data_list=data_list+value
    flag_data=flag_data+flag

    return data_list,flag_data

def ax1_plt():
    filename=r'../Dataset/real_2.csv'
    data_list,flag_data=get_data(filename)
    d_len=len(data_list)
    num_list=list(range(0,d_len))

    # mark=[]
    # for i in range(d_len):
    #     flag=flag_data[i]
    #     if flag==1:
    #         mark.append(i)
    # print(mark)
    mark=[1362,1433]
    ax1.set_title("(a)",y=-0.2)
    ax1.set_xlabel("timestamp")
    ax1.set_ylabel("normalized value")
    ax1.plot(num_list,data_list,linewidth = '1',color='blue',label='Accuracy')
    ax1.plot(num_list,data_list,markevery=mark, ls="", marker="o",color='red', label="points")


def ax2_plt():
    filename=r'../Dataset/real_55.csv'
    data_list,flag_data=get_data(filename)
    d_len=len(data_list)
    num_list=list(range(0,d_len))

    # mark=[]
    # for i in range(d_len):
    #     flag=flag_data[i]
    #     if flag==1:
    #         mark.append(i)
    # print(mark)
    mark=[1018, 1205]
    ax2.set_title("(b)",y=-0.2)
    ax2.set_xlabel("timestamp")
    ax2.set_ylabel("normalized value")
    ax2.plot(num_list,data_list,linewidth = '1',color='blue',label='Accuracy')
    ax2.plot(num_list,data_list,markevery=mark, ls="", marker="o",color='red', label="points")


def ax3_plt():
    filename=r'../Dataset/real_20.csv'
    data_list,flag_data=get_data(filename)
    d_len=len(data_list)
    num_list=list(range(0,d_len))

    # mark=[]
    # for i in range(d_len):
    #     flag=flag_data[i]
    #     if flag==1:
    #         mark.append(i)
    # print(mark)
    ax3.set_title("(c)",y=-0.2)
    mark=[541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564]
    ax3.set_xlabel("timestamp")
    ax3.set_ylabel("normalized value")
    ax3.plot(num_list,data_list,linewidth = '1',color='blue',label='Accuracy')
    ax3.plot(num_list,data_list,markevery=mark, ls="", marker="o",color='red', label="points")

ax1_plt()
ax2_plt()
ax3_plt()
plt.show()



























