import numpy as np
import os
from IPSData import CollectorIPSData
import csv 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error




forest   = RandomForestRegressor(n_estimators = 10 , max_features = 'sqrt' , criterion = 'mse' , max_depth = 5 , n_jobs = 2 )

basepath = r"F:\IPSData2"
thckpath = r"F:\KLAData\ThicknessData"
ipsdata = CollectorIPSData(basepath , thckpath)
xs , ys , ps  = ipsdata.ReadMulti_RflcThck([2,4])

X = np.concatenate( (xs[0] , xs[1]) , axis = 0)
y = np.concatenate( (ys[0] , ys[1]) ).flatten()

X2Train = xs[0][: , 450:850]
y2Train = ys[0]
X3Train = xs[1][: , 450:850]
y3Train = ys[1]

X = np.concatenate((X2Train , X3Train ))
y = np.concatenate((y2Train , y3Train ))
y = y.reshape( y.shape[0], 1)


X2Test = X2Train[:10,:]
X2Train = X2Train[11:: , : ]
y2Test = y2Train[:10]
y2Train = y2Train[11:]

X3Test = X3Train[:10,:]
X3Train = X3Train[11:: , : ]
y3Test = y3Train[:10]
y3Train = y3Train[11:]

y2Test = y2Test.reshape( y2Test.shape[0] , 1)
y3Test = y3Test.reshape( y3Test.shape[0] , 1)

XTrain = np.concatenate((X2Train , X3Train ))
yTrain = np.concatenate((y2Train , y3Train ))
yTrain = yTrain.reshape( yTrain.shape[0], 1)

p2 = ps[0]
p3 = ps[1]
Position = np.concatenate((p2,p3))

########## Fitting ###########

forest.fit(XTrain,yTrain) # Use Train Data for Train
predict = forest.predict(X) # Prediction of All Datas. use for calc error and correlation
y = y.flatten()
errors = mean_squared_error(y , predict)
core = np.corrcoef( predict , y )[0,1]
fit = np.polyfit(predict, y, deg=1)


########### predict for display ################
pre2Train = forest.predict(X2Train)
pre2Test = forest.predict(X2Test)
pre3Train = forest.predict(X3Train)
pre3Test = forest.predict(X3Test)




################ Display ###################
print("Error : ",errors)
print("Correlation : " , core)

plt.title( "Correlation = {0} , Error = {1},  (Red : #2 , Blue : #4) ".format(core , errors) )
plt.xlabel( "Predict" )
plt.ylabel( "Target" )
s1 = plt.scatter(pre2Train , y2Train , c = 'r' , alpha = 0.5 , label = "#2Train")
s2 = plt.scatter(pre3Train , y3Train , c = 'b' , alpha = 0.5 , label = "#3Train")
s3 = plt.scatter(pre2Test , y2Test   , c = 'g' , alpha = 0.9 , label = "#2Test")
s4 = plt.scatter(pre3Test , y3Test   , c = 'y' , alpha = 0.9 , label = "#3Test")
s5 = plt.scatter( [285]*len(yTrain) , yTrain   , c = 'brown' , alpha = 0.9 , label = "Target")
plt.plot( predict , fit[0]*predict + fit[1] , color = 'Turquoise')
plt.legend()
plt.xlim(280,340)
plt.ylim(280,340)
plt.legend()
#plt.plot(LossTest , 'b' )

for label,x,y,i,pos in zip(Position,predict,y,range(0,len(y)),Position ):
    
    posstr = pos.split(" ")
    posx = float(posstr[0])
    posy = float(posstr[1])
    rho  = ( posx**2+posy**2 )**0.5

    if i%31 == 0:
        rand = np.random.uniform(1.0 , 2.0)
        plt.annotate(
            label,
            xy = (x,y),
            xytext=(10*rand,-10*rand),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    #if i%71 == 0:
    #    plt.annotate(
    #        label,
    #        xy = (x,y),
    #        xytext=(30*(i/60),30*(i/60)),
    #        textcoords='offset points',
    #        bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.2),
    #        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()

