import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,8))
x=[]
y=[]
with open(r'D:\python\torch_study\datasets\liner.csv','r') as f:
    lines = f.read().split('\n')

    x = [float(i.split(',')[0]) for i in lines]
    y = [float(i.split(',')[1]) for i in lines]

plt.scatter(x[0:50],y[0:50]) # 画出数据点

weight = 1.9968
bia = 0.0279
x = np.arange(0,5,0.1)
y = weight*x + bia

plt.plot(x,y,'r')

plt.show()

