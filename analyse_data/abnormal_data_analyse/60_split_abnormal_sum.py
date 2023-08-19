# -*- coding: utf-8 -*- 
# @Time : 2022/9/15 21:13 
# @Author : xlf
# @File : 60_split_abnormal_sum.py
from matplotlib import pyplot as plt
import matplotlib
plt.rc('font',family='Times New Roman', size=12)

#柱状图绘制
x_label=['1-10','11-20','21-30','31-40','41-50','51-60']
x=[1,2,3,4,5,6]
#y_value=[62,64,68,73,48,55]
y_value=[90,72,44,102,104,102]
#'steelblue','cornflowerblue'
color=['steelblue']
plt.xlabel("Window Range")
plt.ylabel("Anomaly Count")
plt.xticks(x, x_label)
plt.bar(x, y_value,color=color,alpha=0.8)
# 显示数字
for a, b in zip(x, y_value):
    plt.text(a, b+1, str(b),ha='center', va='bottom', fontsize=12,rotation=0)

#折线图绘制
plt.plot(x, y_value, "r", marker='.',linestyle='--', c='black', ms=5, linewidth='1')

plt.show()














