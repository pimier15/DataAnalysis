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

x2Temp , y2Temp , p2temp = ips.GetAllReflcThick(2 , thicknessPath)
x2Temp = x2Temp[: , 350:850]

x2Test = NormX2.Normalization( x2Temp)
y2Test = NormY2.Normalization( y2Temp)
y2Test = y2Test.reshape( y2Test.shape[0] , 1)

x3Temp , y3Temp , p3temp = ips.GetAllReflcThick(3 , thicknessPath)
x3Temp = x3Temp[: , 350:850]

x3Test = NormX3.Normalization( x3Temp)
y3Test = NormY3.Normalization( y3Temp)
y3Test = y3Test.reshape( y3Test.shape[0] , 1)


iSize = x2Test.shape[1]

lr = 0.01
h1Size = 200
h2Size = 30
h3Size = 10
oSize = 1

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


#Start Check

Predict2 , Label2 = sess.run([Output , Y] , feed_dict = { X : x2Test , Y : y2Test })
Predict3 , Label3 = sess.run([Output , Y] , feed_dict = { X : x3Test , Y : y3Test })
 
Predict2 = NormY2.DeNormalization(Predict2)    
Label2 = NormY2.DeNormalization(Label2)    
Predict3 = NormY3.DeNormalization(Predict3)    
Label3 = NormY3.DeNormalization(Label3)

Predict = np.concatenate((Predict2,Predict3))
Label = np.concatenate((Label2,Label3))
pTest = np.concatenate((p2temp,p3temp))


pre2 = np.ndarray.flatten( Predict2)
lb2 = np.ndarray.flatten( Label2 )
pre3 = np.ndarray.flatten( Predict3)
lb3 = np.ndarray.flatten( Label3 )

pre = np.ndarray.flatten( Predict)
lb = np.ndarray.flatten( Label )
core = np.corrcoef( pre , lb )[0,1]
print("Current Best Correlation : {0}".format(core))


savepath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\result"

pre.tofile( os.path.join(savepath, "Predict.csv") ,sep=',')
lb.tofile( os.path.join(savepath, "Label.csv"), sep=',')
pTest.tofile( os.path.join(savepath, "Point.csv"), sep=',')

fit = np.polyfit(pre, lb, deg=1)


print(pre.shape)
print(lb.shape)
posdigit = 12
digit = 18
print("#2 Sample")
for i in range(0,pre2.shape[0]):
    print("Pos : {0:<{n1}}  | Predict  : {1:<{n}}  |  Target {2:<{n}}".format(  p2temp[i],pre2[i], lb2[i] , n1 = digit,  n  = digit ))

print()
print("#3 Sample")
for i in range(0,pre3.shape[0]):
    print("Pos : {0:<{n1}}  | Predict  : {1:<{n}}  |  Target {2:<{n}}".format(  p3temp[i],pre3[i], lb3[i] , n1 = digit,  n  = digit ))


plt.title( "Correlation = {0} ,  (Red : #2 , Blue : #3) ".format(core) )
plt.xlabel( "Predict" )
plt.ylabel( "Target" )
plt.scatter(pre2 , lb2 , c = 'r' , alpha = 0.6 )
plt.scatter(pre3 , lb3  , alpha = 0.6)
plt.plot( pre , fit[0]*pre + fit[1] , color = 'green')
plt.xlim(280,340)
plt.ylim(280,340)
#plt.plot(LossTest , 'b' )

for label,x,y,i in zip(pTest,pre,lb,range(0,len(lb))):
    if i %20 == 0:
        plt.annotate(
            label,
            xy = (x,y),
            xytext=(30*(i/60),-30*(i/60)),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()
 









