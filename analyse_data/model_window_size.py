# -*- coding: utf-8 -*-
# @Time : 2022/9/7 20:45
# @Author : xlf
# @File : model_window_size.py

import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font',family='Times New Roman', size=12)

data=pd.read_excel('windows_size.xlsx')
num_list=data.number.tolist()
accuracy_list=data.accuracy.tolist()
recall_list=data.recall.tolist()
f1score_list=data.F1score.tolist()

plt.xlabel("Window Size")
plt.ylabel("CTGA")

plt.plot(num_list,accuracy_list,marker='o',linewidth = '1',color='red',label='Accuracy')
plt.plot(num_list,recall_list,marker='*',linewidth = '1',color='green',label='Recall')
plt.plot(num_list,f1score_list,marker='^',linewidth = '1',color='blue',label='F1')

plt.rcParams.update({'font.size':12})
plt.legend()
plt.show()
























