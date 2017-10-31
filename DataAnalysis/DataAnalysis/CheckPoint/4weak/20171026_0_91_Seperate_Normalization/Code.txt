import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DataTranform import Normalizer
from sklearn.preprocessing import normalize
from ReadIpsData import *
tf.set_random_seed(777)  # reproducibility
np.random.seed(7)

NormX2 = Normalizer()
NormY2 = Normalizer()
NormX3 = Normalizer()
NormY3 = Normalizer()

ips = ReadIPS(r"F:\IPSData")
thicknessPath = r"F:\KLAData\ThicknessData"
savepath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\save" 
x2Temp , y2Temp , p2Temp = ips.GetAllReflcThick(2 , thicknessPath)
x2Temp = x2Temp[: , 350:850]

x2Train = NormX2.Normalization( x2Temp)
y2Train = NormY2.Normalization( y2Temp)


x3Temp , y3Temp , p3Temp = ips.GetAllReflcThick(3 , thicknessPath)
x3Temp = x3Temp[: , 350:850]

x3Train = NormX3.Normalization( x3Temp)
y3Train = NormY3.Normalization( y3Temp)


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
#saver.restore(sess , os.path.join(savepath , 'Best_model.ckpt')) 


LossList = []
#LossTest = []

CorrBest = 0

for i in range(0,trainEpoch):
    batchMask2 = np.random.choice( x2Train.shape[0] , batchSize)
    batchMask3 = np.random.choice( x3Train.shape[0] , batchSize)
    x2Batch = x2Train[batchMask2]
    y2Batch = y3Train[batchMask2]
    x3Batch = x2Train[batchMask3]
    y3Batch = y3Train[batchMask3]
    
    xBatch = np.concatenate( (x2Batch,x3Batch) )
    yBatch = np.concatenate( (y2Batch,y3Batch) )
    yBatch = yBatch.reshape( yBatch.shape[0] , 1)


    
    lossTrain , _ = sess.run([Loss , optimizer] , feed_dict = { X  :xBatch , Y : yBatch })
    #lossTest = sess.run([Loss] , feed_dict = { X  :x_test , Y : y_test })
    PredictLabel = []
    if i % 500 == 0 :
        print("Iter : {0} , Loss : {1}".format(i , lossTrain))


    if i%500 == 0:
        
        Predict2 , Label2 = sess.run([Output , Y] , feed_dict = { X : x2Test , Y : y2Test })
        Predict3 , Label3 = sess.run([Output , Y] , feed_dict = { X : x3Test , Y : y3Test })

        Predict2 = NormY2.DeNormalization(Predict2)    
        Label2 = NormY2.DeNormalization(Label2)    
        Predict3 = NormY3.DeNormalization(Predict3)    
        Label3 = NormY3.DeNormalization(Label3)

        Predict = np.concatenate((Predict2,Predict3))
        Label = np.concatenate((Label2,Label3))

        pre = np.ndarray.flatten( Predict)
        lb = np.ndarray.flatten( Label)

        core = np.corrcoef( pre , lb )[0,1]
        PredictLabel.append([Predict,Label])

        if CorrBest < core :
            print()
            print("Current Best Correlation : {0}".format(core))
            CorrBest = core
            saver.save(sess , os.path.join(savepath , 'Best_model.ckpt') )
            for i in range(20):
                print("Predict  : {0}  Target L {1}".format(  Predict[i], Label[i]))

            print()

        #plt.scatter(Predict , Label )
        #plt.xlabel("Predict")
        #plt.ylabel("Target")
        #plt.show()

    LossList.append(lossTrain)
    #LossTest.append(lossTest)


#plot								

plt.title( "Loss Graph" )
plt.xlabel( "Epoch" )
plt.ylabel( "Cross Entropy Loss" )
plt.plot(LossList , 'r' )
#plt.plot(LossTest , 'b' )
plt.show()
 









