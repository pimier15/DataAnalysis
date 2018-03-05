import numpy as np
import os
from IPSData import CollectorIPSData
import csv 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle




forest   = RandomForestRegressor(n_estimators = 10 , max_features = 'sqrt' , criterion = 'mse' , max_depth = 5 , n_jobs = 2 )

basepath = r"F:\IPSData2"
thckpath = r"F:\KLAData\ratioData"
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
NumOfTest = 10
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

for i in SamplesIdx:
    XTrain[i] = XShuf[i][NumOfTest+1 ::, :]
    XTest[i]  = XShuf[i][:NumOfTest, :]
    yTrain[i] = yShuf[i][NumOfTest+1::]
    yTest[i]  = yShuf[i][:NumOfTest]
    pTest[i]  = pShuf[i][:NumOfTest]

for i in SamplesIdx:
    yTrain[i] = yTrain[i].reshape( yTrain[i].shape[0], 1)
    yTest[i]  = yTest[i] .reshape(  yTest[i].shape[0], 1)

XTrainAll = np.concatenate( tuple(XTrain.values()) , axis = 0)
yTrainAll  = np.concatenate( tuple(yTrain.values())).flatten()
yTrainAll  = yTrainAll.reshape( yTrainAll.shape[0], 1)

#XTrainAll = np.concatenate( (XTrain[0],XTrain[3],XTrain[2]) , axis = 0)
#yTrainAll  = np.concatenate((yTrain[0],yTrain[3],yTrain[2])).flatten()
#yTrainAll  = yTrainAll.reshape( yTrainAll.shape[0], 1)


########## Fitting ###########
forest.fit(XTrainAll,yTrainAll) # Use Train Data for Train

predict = forest.predict(XALL) # Prediction of All Datas. use for calc error and correlation
errors = mean_squared_error(yALL , predict)
core = np.corrcoef( predict , yALL )[0,1]
fit = np.polyfit(predict, yALL, deg=1)




########### Save Result in Csv #############
predictForCsv = {}
for i in SamplesIdx:
    predictForCsv[i] = forest.predict(xs[i])

stream = open( r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\result\RandomForest\Result.csv" , 'w')
wt = csv.writer(stream )

for i in SamplesIdx:
    wt.writerow(["Sample"+str(i)])
    wt.writerow(["Pos","Thickness","Target Thickness"])
    for j in range(len(predictForCsv[i])):
        wt.writerow([ps[i][j] , predictForCsv[i][j] , ys[i][j]])
    
stream.close()



########### predict for display ################

for i in SamplesIdx:
    PredicTrainPer[i] = forest.predict(XTrain[i])
    PredicTestPer[i] =  forest.predict(XTest[i])

################ Display ###################
print("Error : ",errors)
print("Correlation : " , core)

plt.title( "Correlation = {0:.4f} , Error = {1:.4f} ".format(core , errors) )
plt.xlabel( "Predict" )
plt.ylabel( "Target" )

colorTrain = ["b","r","purple","g"]
colorTest = ["deepskyblue","salmon","orchid","mediumseagreen"]

for i in SamplesIdx:
    plt.scatter(PredicTrainPer[i] , yTrain[i] , c = colorTrain[i] , alpha = 0.5 , label = "#{0}Train".format(str(i+1)))
    plt.scatter(PredicTestPer[i] , yTest[i]   , c = colorTest[i]  , alpha = 0.9 , label = "#{0}Test".format(str(i+1)))

plt.scatter( [11]*len(yALL) , yALL   , c = 'brown' , alpha = 0.9 , label = "Target")
plt.plot( predict , fit[0]*predict + fit[1] , color = 'Turquoise')
plt.legend()
plt.xlim(10,25)
plt.ylim(10,25)
plt.legend()
#plt.plot(LossTest , 'b' )

#for label,x,y,i,pos in zip(Position,predict,y,range(0,len(y)),Position ):
#    
#    posstr = pos.split(" ")
#    posx = float(posstr[0])
#    posy = float(posstr[1])
#    rho  = ( posx**2+posy**2 )**0.5
#
#    if i%31 == 0:
#        rand = np.random.uniform(1.0 , 2.0)
#        plt.annotate(
#            label,
#            xy = (x,y),
#            xytext=(10*rand,-10*rand),
#            textcoords='offset points',
#            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
#            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
#
#    #if i%71 == 0:
#    #    plt.annotate(
#    #        label,
#    #        xy = (x,y),
#    #        xytext=(30*(i/60),30*(i/60)),
#    #        textcoords='offset points',
#    #        bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.2),
#    #        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()

