import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from scipy import signal #滤波等

x = np.arange(0, 240)

y =[160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 161, 160, 158, 158, 158, 158, 158, 158, 160, 160, 160, 159, 159, 160, 160, 158, 158, 158, 158, 158, 159, 158, 155, 156, 155, 153, 151, 151, 151, 149, 147, 148, 141, 141, 139, 135, 133, 131, 126, 125, 124, 120, 118, 115, 109, 106, 101, 96, 93, 91, 86, 85, 84, 79, 79, 78, 73, 72, 71, 69, 68, 68, 67, 66, 65, 65, 65, 63, 63, 63, 62, 63, 63, 64, 64, 65, 65, 67, 65, 67, 69, 69, 70, 70, 71, 73, 75, 75, 76, 81, 82, 82, 85, 88, 91, 93, 96, 98, 99, 100, 104, 110, 108, 114, 116, 116, 121, 122, 125, 125, 131, 133, 137, 141, 142, 142, 143, 143, 145, 149, 150, 150, 152, 154, 154, 155, 156, 156, 155, 153, 150, 149, 147, 145, 139, 137, 134, 127, 126, 122, 117, 113, 110, 100, 98, 94, 90, 85, 85, 82, 79, 77, 74, 74, 71, 68, 67, 68, 66, 66, 66, 66, 64, 63, 60, 60, 61, 60, 60, 60, 62, 64, 64, 64, 64, 64, 66, 66, 68, 72, 72, 73, 78, 79, 80, 84, 86, 87, 92, 93, 96, 104, 107, 109, 114, 113, 118, 121, 123, 126, 130, 130, 134, 140, 141, 142, 144, 145, 147, 150, 150, 149, 149, 151, 153, 154, 157, 155]

z1 = np.polyfit(x, y, 100) # 用7次多项式拟合

p1 = np.poly1d(z1) #多项式系数

yvals=p1(x)


max=yvals[signal.argrelextrema(yvals, np.greater)]
max=[x for x in max if x>=140]
print("max: ",max)


min=(yvals[signal.argrelextrema(yvals, np.less)])
min=[x for x in min if x<=80]
print("min: ",min)
