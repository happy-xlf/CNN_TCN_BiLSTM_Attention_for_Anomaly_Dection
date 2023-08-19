# -*- coding: utf-8 -*- 
# @Time : 2022/9/12 15:33 
# @Author : xlf
# @File : model_bigru_size.py
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Times New Roman', size=16)

data=pd.read_excel('bigru_size.xlsx')
num_list=data.bigrusize.tolist()
accuracy_list=data.accuracy.tolist()
recall_list=data.recall.tolist()
f1score_list=data.F1score.tolist()

plt.xlabel("BiGRU Hidden Size")
plt.ylabel("CT-BiGRU-Attention")

plt.plot(num_list,accuracy_list,linewidth = '1',color='red',label='Accuracy')
plt.plot(num_list,recall_list,linewidth = '1',color='green',label='Recall')
plt.plot(num_list,f1score_list,linewidth = '1',color='blue',label='F1-score')

plt.rcParams.update({'font.size':10})
plt.legend()
plt.show()





















