# -*- coding: utf-8 -*-
# @Time : 2022/8/26 18:47
# @Author : xlf
# @File : model_accuracy_loss.py

import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rc('font',family='Times New Roman', size=12)

lstm=pd.read_excel('csv/lstm.xls')
gru=pd.read_excel('csv/gru.xls')
tcn=pd.read_excel('csv/tcn.xls')
cnn=pd.read_excel('csv/cnn.xls')

cnn_lstm=pd.read_excel('csv/cnn_lstm.xlsx')
cnn_gru=pd.read_excel('csv/cnn_gru.xlsx')
cnn_bilstm=pd.read_excel('csv/cnn_bilstm.xlsx')
cnn_bigru=pd.read_excel('csv/cnn_bigru.xlsx')

# cnn_tcn=pd.read_excel('csv/cnn_tcn.xlsx')
cnn_tcn_lstm=pd.read_excel('csv/cnn_tcn_lstm.xlsx')
cnn_tcn_gru=pd.read_excel('csv/cnn_tcn_gru.xlsx')
cnn_tcn_bilstm=pd.read_excel('csv/cnn_tcn_bilstm.xlsx')
cnn_tcn_bigru=pd.read_excel('csv/cnn_tcn_bigru.xlsx')
cnn_tcn_bilstm_attention=pd.read_excel('csv/cnn_tcn_bilstm_attention.xlsx')
cnn_tcn_bigru_attention=pd.read_excel('csv/cnn_tcn_bigru_attention.xlsx')

# cnn_tcn_accuracy=cnn_tcn.cnn_tcn_accuracy.tolist()
# cnn_tcn_loss=cnn_tcn.cnn_tcn_loss.tolist()
lstm_accuracy=lstm.lstm_accuracy.tolist()
gru_accuracy=gru.gru_accuracy.tolist()
tcn_accuracy=tcn.tcn_accuracy.tolist()
cnn_accuracy=cnn.cnn_accuracy.tolist()

cnn_lstm_accuracy=cnn_lstm.cnn_lstm_accuracy.tolist()
cnn_gru_accuracy=cnn_gru.cnn_gru_accuracy.tolist()
cnn_bilstm_accuracy=cnn_bilstm.cnn_bilstm_accuracy.tolist()
cnn_bigru_accuracy=cnn_bigru.cnn_bigru_accuracy.tolist()


cnn_tcn_lstm_accuracy=cnn_tcn_lstm.cnn_tcn_lstm_accuracy.tolist()
cnn_tcn_lstm_loss=cnn_tcn_lstm.cnn_tcn_lstm_loss.tolist()

cnn_tcn_gru_accuracy=cnn_tcn_gru.cnn_tcn_gru_accuracy.tolist()
cnn_tcn_gru_loss=cnn_tcn_gru.cnn_tcn_gru_loss.tolist()

cnn_tcn_bilstm_accuracy=cnn_tcn_bilstm.cnn_tcn_bilstm_accuracy.tolist()
cnn_tcn_bilstm_loss=cnn_tcn_bilstm.cnn_tcn_bilstm_loss.tolist()

cnn_tcn_bigru_accuracy=cnn_tcn_bigru.cnn_tcn_bigru_accuracy.tolist()
cnn_tcn_bigru_loss=cnn_tcn_bigru.cnn_tcn_bigru_loss.tolist()

cnn_tcn_bilstm_attention_accuracy=cnn_tcn_bilstm_attention.cnn_tcn_bilstm_attention_accuracy.tolist()
cnn_tcn_bilstm_attention_loss=cnn_tcn_bilstm_attention.cnn_tcn_bilstm_attention_loss.tolist()

cnn_tcn_bigru_attention_accuracy=cnn_tcn_bigru_attention.cnn_tcn_bigru_attention_accuracy.tolist()
cnn_tcn_bigru_attention_loss=cnn_tcn_bigru_attention.cnn_tcn_bigru_attention_loss.tolist()

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
epoch_list=list(range(501))
plt.plot(epoch_list,lstm_accuracy,linewidth = '1',color='aqua',label='LSTM')
plt.plot(epoch_list,gru_accuracy,linewidth = '1',color='#CC0066',label='GRU')
plt.plot(epoch_list,cnn_accuracy,linewidth = '1',color='green',label='CNN')
plt.plot(epoch_list,tcn_accuracy,linewidth = '1',color='blue',label='TCN')

plt.plot(epoch_list,cnn_bilstm_accuracy,linewidth = '1',color='maroon',label='CNN-BiLSTM')
plt.plot(epoch_list,cnn_bigru_accuracy,linewidth = '1',color='gold',label='CNN-BiGRU')

plt.plot(epoch_list,cnn_tcn_bilstm_accuracy,linewidth = '1',color='orange',label='CNN-TCN-BiLSTM')
plt.plot(epoch_list,cnn_tcn_bigru_accuracy,linewidth = '1',color='purple',label='CNN-TCN-BiGRU')
plt.plot(epoch_list,cnn_tcn_bilstm_attention_accuracy,linewidth = '1',color='brown',label='CT-BiLSTM-Attention')
plt.plot(epoch_list,cnn_tcn_bigru_attention_accuracy,linewidth = '1',color='red',label='CTGA')
plt.rcParams.update({'font.size':12})
plt.legend()
plt.show()
