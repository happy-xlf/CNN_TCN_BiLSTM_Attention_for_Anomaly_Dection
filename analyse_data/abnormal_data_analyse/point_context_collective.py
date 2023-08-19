# -*- coding: utf-8 -*- 
# @Time : 2022/9/16 11:32 
# @Author : xlf
# @File : point_context_collective.py
from matplotlib import pyplot as plt
import matplotlib
plt.rc('font',family='Times New Roman', size=12)

#柱状图绘制
x_label=['Point','Contextual','Collective']
x=[1,2,3]
#y_value=[62,64,68,73,48,55]
y_value=[7,49,458]
#'steelblue','cornflowerblue'
color=['steelblue']
plt.xlabel("Abnormal Type")
plt.ylabel("Misclassification Count")
plt.xticks(x, x_label)
plt.bar(x, y_value,color=color,alpha=0.8)
# 显示数字
for a, b in zip(x, y_value):
    plt.text(a, b+1, str(b),ha='center', va='bottom', fontsize=12,rotation=0)

plt.show()











