# -*- coding: utf-8 -*-
# @Time : 2022/9/4 17:31
# @Author : xlf
# @File : model_optimizer.py
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font',family='Times New Roman', size=12)

model_adam=pd.read_excel('model_optimizer/Adam_Loss.xlsx')
model_adadelta=pd.read_excel('model_optimizer/Adadelta_Loss.xlsx')
model_adagrad=pd.read_excel('model_optimizer/Adagrad_Loss.xlsx')
model_sgd=pd.read_excel('model_optimizer/SGD_Loss.xlsx')

#cnn_tcn_bigru_attention_loss
model_adam_loss=model_adam.cnn_tcn_bigru_attention_loss.tolist()
model_adadelta_loss=model_adadelta.cnn_tcn_bigru_attention_loss.tolist()
model_adagrad_loss=model_adagrad.cnn_tcn_bigru_attention_loss.tolist()
model_sgd_loss=model_sgd.cnn_tcn_bigru_attention_loss.tolist()

plt.xlabel("Epoch")
plt.ylabel("Loss")
epoch_list=list(range(501))
plt.plot(epoch_list,model_adam_loss,linewidth = '1',color='red',label='Adam')
plt.plot(epoch_list,model_adadelta_loss,linewidth = '1',color='green',label='Adadelta')
plt.plot(epoch_list,model_adagrad_loss,linewidth = '1',color='blue',label='Adagrad')
plt.plot(epoch_list,model_sgd_loss,linewidth = '1',color='orange',label='SGD')
plt.rcParams.update({'font.size':12})
plt.legend()
plt.show()
















