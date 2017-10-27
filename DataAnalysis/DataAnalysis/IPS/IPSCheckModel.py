import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DataTransform import Normalizer
from sklearn.preprocessing import normalize
from IPSData import *
tf.set_random_seed(777)  # reproducibility
np.random.seed(7)

NormX2 = Normalizer()
NormY2 = Normalizer()
ips = CollectorIPSData(r"F:\IPSData")
thicknessPath = r"F:\KLAData\ThicknessData"
savepath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\save" 
x2Temp , y2Temp , p2temp = ips.Read_RflcThck(2 , thicknessPath)
x2Temp = x2Temp[: , 350:850]

x2Train = NormX2.Normalization( x2Temp)
y2Train = NormY2.Normalization( y2Temp)


x3Temp , y3Temp , p3temp = ips.Read_RflcThck(3 , thicknessPath)
x3Temp = x3Temp[: , 350:850]

x3Train = NormX2.Normalization( x3Temp)
y3Train = NormY2.Normalization( y3Temp)


#x2Test = x2Train[:10,:]
#x2Train = x2Train[11:: , : ]
#y2Test = y2Train[:10]
#y2Train = y2Train[11:]
#
#x3Test = x3Train[:10,:]
#x3Train = x3Train[11:: , : ]
#y3Test = y3Train[:10]
#y3Train = y3Train[11:]





xTrain = np.concatenate( (x2Train , x3Train) , axis = 0)
yTrain = np.concatenate( (y2Train , y3Train) )
pTrain = np.concatenate( ( p2temp,p3temp) )
yTrain = yTrain.reshape( yTrain.shape[0] , 1)

#xTest = np.concatenate( (x2Test , x3Test) , axis = 0)
#yTest = np.concatenate( (y2Test , y3Test) )
#yTest = yTest.reshape( yTest.shape[0] , 1)




sampleSize , iSize = xTrain.shape

lr = 0.01
h1Size = 200
h2Size = 30
h3Size = 10
oSize = 1
batchSize = 10
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

lossTrain , _ = sess.run([Loss , optimizer] , feed_dict = { X  :xTrain , Y : yTrain })
Predict , Label = sess.run([Output , Y] , feed_dict = { X : xTrain , Y : yTrain })
Predict = NormY2.DeNormalization(Predict)    
Label = NormY2.DeNormalization(Label)    

pre = np.ndarray.flatten( Predict)
lb = np.ndarray.flatten( Label )
core = np.corrcoef( pre , lb )[0,1]
print("Current Best Correlation : {0}".format(core))


savepath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\result"

pre.tofile( os.path.join(savepath, "Predict.csv") ,sep=',')
lb.tofile( os.path.join(savepath, "Label.csv"), sep=',')
pTrain.tofile( os.path.join(savepath, "Point.csv"), sep=',')

print(pre.shape)
print(lb.shape)

#for i in range(0,xTrain.shape[0]):
#    print("Predict  : {0}  Target L {1}".format(  Predict[i], Label[i]))

plt.title( "Corr" )
plt.xlabel( "Predict" )
plt.ylabel( "Target" )
plt.scatter(pre , lb  )
#plt.plot(LossTest , 'b' )
plt.show()
 









