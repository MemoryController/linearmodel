import random

path = r'D:\python\torch_study\datasets\liner.csv'

with open(path,'w') as f:
    for i in range(0,1000,1):
        x = i*0.1
        y = (random.random()-0.5)*0.1 + 2*x
        f.write('%f,%.4f\n' % (x,y))
