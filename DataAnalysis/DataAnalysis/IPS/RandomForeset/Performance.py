import numpy as np
import os
from IPSData import CollectorIPSData
import csv 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import time


start = time.time() 
forest   = RandomForestRegressor(n_estimators = 20 , max_features = 'sqrt' , criterion = 'mse' , max_depth = 12 , n_jobs = 2 )

basepath = r"F:\IPSData2"
thckpath = r"F:\KLAData\ThicknessData"
ipsdata = CollectorIPSData(basepath , thckpath)

Samples = [1,2,3,4]
SamplesIdx = [0,1,2,3]

xs , ys , ps  = ipsdata.ReadMulti_RflcThck(Samples)


#Shuffle

# Datas
XShuf,yShuf ,pShuf       = {},{},{}
XTrain , yTrain , pTrain = {},{},{}
XTest  , yTest  , pTest  = {},{},{}
XTrainALl , yTrainAll = {},{} 
XTestALl , yTestAll   = {},{}
XALL , yALL = {},{}
NumOfTest = 125
PredicTrainPer = {}
PredicTestPer = {}

#Datas for Csv
Xcsv = {}
ycsv = {}
pcsv = {}

for i in SamplesIdx:
    xs[i] = xs[i][:,450:850]
    XShuf[i] ,yShuf[i] ,pShuf[i] = shuffle(xs[i] , ys[i] , ps[i])

val =  tuple(XShuf.values())
XALL = np.concatenate( tuple(XShuf.values()) , axis = 0)
yALL = np.concatenate( tuple(yShuf.values())).flatten()
yALL = yALL.reshape( yALL.shape[0], 1).flatten()


#XTrainAll = np.concatenate( (XTrain[0],XTrain[3],XTrain[2]) , axis = 0)
#yTrainAll  = np.concatenate((yTrain[0],yTrain[3],yTrain[2])).flatten()
#yTrainAll  = yTrainAll.reshape( yTrainAll.shape[0], 1)


########## Fitting ###########
end = time.time() 
print(end - start) 
start = time.time() 
forest.fit(XALL,yALL) # Use Train Data for Train
end = time.time() 
print(end - start) 

start = time.time() 
predict = forest.predict(XALL) # Prediction of All Datas. use for calc error and correlation
end = time.time()
print(end - start) 
print()
