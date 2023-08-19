# -*- coding: utf-8 -*- 
# @Time : 2022/9/29 16:50 
# @Author : xlf
# @File : xtest.py

import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3) # two axes on figure
max_1_season=[1,2,3]
base_1_season=[4,5,6]
ax1.plot(max_1_season, base_1_season,'r')
ax2.plot(max_1_season, base_1_season, 'b')
ax3.plot(max_1_season, base_1_season, 'g')

plt.savefig('plot.png', dpi=600)
plt.show()