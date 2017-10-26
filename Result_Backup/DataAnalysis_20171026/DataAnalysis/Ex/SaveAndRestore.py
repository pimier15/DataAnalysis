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


savepath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\save"
batchSize = 100
trainEpoch = 80000
printstep = 50

lr = 0.01
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
param_list = [W1,b1,W2,b2,W3,b3]
saver = tf.train.Saver(param_list)


#Output
 
L1 = tf.nn.sigmoid( tf.matmul( X , W1) + b1 )
L2 = tf.nn.sigmoid( tf.matmul( L1 , W2 ) + b2)
                                                                           
Output = tf.nn.softmax( tf.matmul( L2 , W3 ) + b3)

#Loss = tf.reduce_mean( - tf.reduce_sum( Y * tf.log( Output ) , reduction_indices = [1] ) ) 
Loss = tf.reduce_mean( - tf.reduce_sum( Y * tf.log( Output ) , reduction_indices = [1] ) ) 

optimizer = tf.train.AdagradOptimizer( lr ).minimize( Loss )


#init

sess =tf.Session()
sess.run( tf.global_variables_initializer() ) # restore 하기 전에 한번 런 시킨다. 
saver.restore(sess , os.path.join(savepath , str(700) + '_model.ckpt')) 




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
    if i % 100 == 0 :
        #saver.save(sess , os.path.join(savepath , str(i) + '_model.ckpt') )
        print("Train : {0}  Tset : {1}".format(lossTrain, lossTest))


#plot								

plt.title( "Loss Graph" )
plt.xlabel( "Epoch" )
plt.ylabel( "Cross Entropy Loss" )
plt.plot(LossList , 'r' )
plt.plot(LossTest , 'b' )
plt.show()
 









