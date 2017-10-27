import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv    
from IPSData import CollectorIPSData
from sklearn.preprocessing import normalize
from ReadIpsData import *
tf.set_random_seed(777)  # reproducibility
np.random.seed(7)

basepath = r"F:\IPSData"
thckpath = r"F:\KLAData\ThicknessData"
ipsdata = CollectorIPSData(basepath , thckpath)
xs , ys , ps , ns = ipsdata.ReadMulti_RflcThck_Norm([2,3])

savepath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\save" 

x2Train = xs[0][: , 350:850]
y2Train = ys[0]
x3Train = xs[1][: , 350:850]
y3Train = ys[1]



x2Test = x2Train[:10,:]
x2Train = x2Train[11:: , : ]
y2Test = y2Train[:10]
y2Train = y2Train[11:]
x3Test = x3Train[:10,:]
x3Train = x3Train[11:: , : ]
y3Test = y3Train[:10]
y3Train = y3Train[11:]

y2Test = y2Test.reshape( y2Test.shape[0] , 1)
y3Test = y3Test.reshape( y3Test.shape[0] , 1)


xTrain = np.concatenate((x2Train , x3Train ))
yTrain = np.concatenate((y2Train , y3Train ))
yTrain = yTrain.reshape( yTrain.shape[0], 1)

xTest = np.concatenate((x2Test , x3Test ))
yTest = np.concatenate((y2Test , y3Test ))
yTest = yTest.reshape( yTest.shape[0], 1)

def SaveWBN(session):
    w , b = session.run([W1,b1])
    print(type(w))
    print(type(b))
    w = w.flatten()
    b = b.flatten()
    
    basewbpath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\save\wb"
    np.savetxt(os.path.join(basewbpath , "W.csv") , w)
    np.savetxt(os.path.join(basewbpath , "b.csv") , b)
    
    f2 = open(os.path.join(basewbpath , "N2.csv"), 'w', encoding='utf-8', newline='')
    f3 = open(os.path.join(basewbpath , "N3.csv"), 'w', encoding='utf-8', newline='')
    wr2 = csv.writer(f2)
    wr3 = csv.writer(f3)
    
    wr2.writerow([ns[0]['Y'].meanList ,ns[0]['Y'].stdList ])
    wr3.writerow([ns[1]['Y'].meanList ,ns[1]['Y'].stdList])
    
    f2.close()
    f3.close()
    




iSize = x2Train.shape[1]

lr = 0.01
h1Size = 200
h2Size = 30
h3Size = 10
oSize = 1
batchSize = 5
trainEpoch = 100000



X = tf.placeholder( tf.float32 , [None,iSize])
Y = tf.placeholder( tf.float32  , [None,oSize])

#Layer
W1 = tf.Variable( tf.random_normal( [iSize , oSize] ) , name="W1" )
b1 = tf.Variable( tf.zeros( [oSize] ))

#W2 = tf.Variable( tf.random_normal( [h1Size , h2Size] ) , name="W2" )
#b2 = tf.Variable( tf.zeros( [h2Size] ))
#
#W3 = tf.Variable( tf.random_normal( [h2Size , oSize] ) , name="W3" )
#b3 = tf.Variable( tf.zeros( [oSize] ))

#W4 = tf.Variable( tf.random_normal( [h3Size , oSize] ) , name="W4" )
#b4 = tf.Variable( tf.zeros( [oSize] ))
paramList = [W1,b1]
saver = tf.train.Saver(paramList)

#Output
 
#L1 = tf.nn.relu( tf.matmul( X , W1) + b1 )
#L2 = tf.nn.relu( tf.matmul( L1 , W2 ) + b2)
#L3 = tf.nn.relu( tf.matmul( L2 , W3 ) + b3)
Output =  tf.matmul( X , W1) + b1                                                                       
#Output =  tf.matmul( L2 , W3) + b3                                                                      
#Output = tf.nn.softmax( tf.matmul( L3 , W4 ) + b4)

#Loss = tf.reduce_mean( - tf.reduce_sum( Y * tf.log( Output ) , reduction_indices = [1] ) ) 
Loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, Output))))

optimizer = tf.train.AdamOptimizer( lr ).minimize( Loss )


#init

sess =tf.Session()
sess.run( tf.global_variables_initializer() )
saver.restore(sess , os.path.join(savepath , 'Best_model.ckpt')) 


LossList = []
#LossTest = []

CorrBest = 0
LossBest = 99999999

#for i in range(0,trainEpoch):
count = 1
while True:
    batchMask2 = np.random.choice( x2Train.shape[0] , batchSize)
    batchMask3 = np.random.choice( x3Train.shape[0] , batchSize)
    x2Batch = x2Train[batchMask2]
    y2Batch = y3Train[batchMask2]
    x3Batch = x2Train[batchMask3]
    y3Batch = y3Train[batchMask3]
    
    xBatch = np.concatenate( (x2Batch,x3Batch) )
    yBatch = np.concatenate( (y2Batch,y3Batch) )
    yBatch = yBatch.reshape( yBatch.shape[0] , 1)

    sess.run([optimizer] , feed_dict = { X  :xBatch , Y : yBatch })
    
    #lossTest = sess.run([Loss] , feed_dict = { X  :x_test , Y : y_test })
    CurrentLoss = None
    PredictLabel = []
    if count % 500 == 0 :
        lossTrain = sess.run([Loss] , feed_dict = { X  :xTrain , Y : yTrain })
        lossTest = sess.run([Loss] , feed_dict = { X  :xTest , Y : yTest })
        CurrentLoss = lossTrain[0]
        print("Iter : {0} , Trrain Loss : {1}  ,  Test Loss : {2} ".format(count , lossTrain[0] , lossTest[0]))

        Predict2 , Label2 = sess.run([Output , Y] , feed_dict = { X : x2Test , Y : y2Test })
        Predict3 , Label3 = sess.run([Output , Y] , feed_dict = { X : x3Test , Y : y3Test })

        Predict2 = ns[0]['Y'].DeNormalization(Predict2)    
        Label2 = ns[0]['Y'].DeNormalization(Label2)    
        Predict3 = ns[1]['Y'].DeNormalization(Predict3)    
        Label3 = ns[1]['Y'].DeNormalization(Label3)

        Predict = np.concatenate((Predict2,Predict3))
        Label = np.concatenate((Label2,Label3))

        pre = np.ndarray.flatten( Predict)
        lb = np.ndarray.flatten( Label)

        core = np.corrcoef( pre , lb )[0,1]
        PredictLabel.append([Predict,Label])

        criticByLoss = True
        if criticByLoss :
             if LossBest > lossTrain[0] + lossTest[0]*0.3:
               print()
               print("Loss Best : {0}".format(lossTrain[0]))
               LossBest = lossTrain[0]
               saver.save(sess , os.path.join(savepath , 'Best_model.ckpt') )
               for i in range(20):
                   print("Predict  : {0}  Target L {1}".format(  Predict[i], Label[i]))
               print()
               SaveWBN(sess)
              
        else : 
           if CorrBest < core :
               print()
               print("Current Best Correlation : {0}".format(core))
               CorrBest = core
               saver.save(sess , os.path.join(savepath , 'Best_model.ckpt') )
               for i in range(20):
                   print("Predict  : {0}  Target L {1}".format(  Predict[i], Label[i]))
               print() 
               SaveWBN(sess)

               

        #plt.scatter(Predict , Label )
        #plt.xlabel("Predict")
        #plt.ylabel("Target")
        #plt.show()
        LossList.append(CurrentLoss)
        if count%10000 == 0 :
            saver.save(sess , os.path.join(savepath , str(i)+'__model.ckpt') )
        
    count += 1
    #LossTest.append(lossTest)


#plot								

plt.title( "Loss Graph" )
plt.xlabel( "Epoch" )
plt.ylabel( "Cross Entropy Loss" )
plt.plot(LossList , 'r' )
#plt.plot(LossTest , 'b' )
plt.show()
 









