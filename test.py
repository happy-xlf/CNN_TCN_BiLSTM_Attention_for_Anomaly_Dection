# -*- coding: utf-8 -*- 
# @Time : 2022/6/29 15:24 
# @Author : xlf
# @File : test.py
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
plt.rc('font',family='Times New Roman', size=16)

cnn_lstm=pd.read_excel('csv/cnn_lstm.xlsx')
cnn_gru=pd.read_excel('csv/cnn_gru.xlsx')
cnn_bilstm=pd.read_excel('csv/cnn_bilstm.xlsx')
cnn_bigru=pd.read_excel('csv/cnn_bigru.xlsx')
#
cnn_lstm_accuracy=cnn_lstm.cnn_lstm_accuracy.tolist()
cnn_lstm_loss=cnn_lstm.cnn_lstm_loss.tolist()

cnn_gru_accuracy=cnn_gru.cnn_gru_accuracy.tolist()
cnn_gru_loss=cnn_gru.cnn_gru_loss.tolist()

cnn_bilstm_accuracy=cnn_bilstm.cnn_bilstm_accuracy.tolist()
cnn_bilstm_loss=cnn_bilstm.cnn_bilstm_loss.tolist()

cnn_bigru_accuracy=cnn_bigru.cnn_bigru_accuracy.tolist()
cnn_bigru_loss=cnn_bigru.cnn_bigru_loss.tolist()



plt.xlabel("Epoch")
plt.ylabel("Accuracy")
epoch_list=list(range(501))
# plt.plot(epoch_list,cnn_tcn_accuracy, linewidth = '1',color='red',label='CNN_TCN')
plt.plot(epoch_list,cnn_lstm_accuracy,linewidth = '1',color='green',label='CNN_LSTM')
plt.plot(epoch_list,cnn_gru_accuracy,linewidth = '1',color='blue',label='CNN_GRU')
plt.plot(epoch_list,cnn_bilstm_accuracy,linewidth = '1',color='orange',label='CNN_BiLSTM')
plt.plot(epoch_list,cnn_bigru_accuracy,linewidth = '1',color='red',label='CNN_BiGRU')
plt.rcParams.update({'font.size':10})
plt.legend()
plt.show()

import torch.nn as nn

# nn.ReLU
# nn.Sigmoid
# nn.Tanh
# nn.LeakyReLU
# nn.Softplus






