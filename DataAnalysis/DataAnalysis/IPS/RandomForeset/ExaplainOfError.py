import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn.svm import SVR

xs1 = np.arange(290 , 300 , 1)
xs2 = np.arange(310 , 330 , 1)
xs3 = np.arange(301,309,1)

t1 = [ 10*sin(x/20*2*pi)+x for x in xs3]
t2 = [ x + log10(10*x) for x in xs3]
t3 = [301 for x in xs3]


xs = np.concatenate((xs1,xs2)).reshape(-1,1)
ys = np.concatenate((xs1,xs2)).reshape(-1,1)
yss = np.concatenate((xs1,xs2)).reshape(-1,1)

plt.plot(yss,yss,'b')
plt.scatter(xs.flatten(),ys.flatten(),s = 10 , c = 'r' )



plt.scatter(xs3,t1,s = 10 , c = 'green' )
plt.scatter(xs3,t2,s = 10 , c = 'orange' )
plt.scatter(xs3,t3,s = 10 , c = 'black' )



plt.xlabel = "Predict"
plt.ylabel = "Target"
plt.show()

