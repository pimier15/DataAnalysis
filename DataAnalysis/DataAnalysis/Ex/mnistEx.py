import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
sys.path.append( os.pardir)
sys.path.append( os.pardir + "dataset")


from mnist import load_mnist

(x_train,y_train) , (x_test,y_test) = load_mnist(True,True,True)

tf.set_random_seed(777)  # reproducibility
np.random.seed(7)

batchSize = 100
trainEpoch = 1000

lr = 0.5
iSize = 784
h1Size = 50
h2Size = 50
oSize = 10

X = tf.placeholder( tf.float32 , [None,iSize])
Y = tf.placeholder( tf.float32  , [None , oSize])

#Layer
W1 = tf.Variable( tf.random_normal( [iSize , h1Size] ) , name="W1" )
b1 = tf.Variable( tf.zeros( [h1Size] ))

W2 = tf.Variable( tf.random_normal( [h1Size , h2Size] ) , name="W2" )
b2 = tf.Variable( tf.zeros( [h2Size] ))

W3 = tf.Variable( tf.random_normal( [h2Size , oSize] ) , name="W3" )
b3 = tf.Variable( tf.zeros( [oSize] ))


#Output
 
L1 = tf.nn.sigmoid( tf.matmul( X , W1) + b1 )
L2 = tf.nn.sigmoid( tf.matmul( L1 , W2 ) + b2)
																		   
Output = tf.nn.softmax( tf.matmul( L2 , W3 ) + b3)

#Loss = tf.reduce_mean( - tf.reduce_sum( Y * tf.log( Output ) , reduction_indices = [1] ) ) 
Loss = tf.reduce_mean( - tf.reduce_sum( Y * tf.log( Output ) , reduction_indices = [1] ) ) 

optimizer = tf.train.AdagradOptimizer( lr ).minimize( Loss )


#init

sess =tf.Session()
sess.run( tf.global_variables_initializer() )


LossList = []
LossTest = []

for i in range(0,trainEpoch):
	batchMask = np.random.choice( x_train.shape[0] , batchSize)
	xBatch = x_train[batchMask]
	yBatch = y_train[batchMask]

	lossTrain = sess.run([Loss , optimizer] , feed_dict = { X  :xBatch , Y : yBatch })
	lossTest = sess.run([Loss] , feed_dict = { X  :x_test , Y : y_test })
	
	LossList.append(lossTrain)
	LossTest.append(lossTest)


#plot								

plt.title( "Loss Graph" )
plt.xlabel( "Epoch" )
plt.ylabel( "Cross Entropy Loss" )
plt.plot(LossList , 'r--' )
plt.plot(LossTest , 'b' )
plt.show()
 









