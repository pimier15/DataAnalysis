import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from DataTransform import Normalizer
from os import path
from IPSData import CollectorIPSData
from sklearn.preprocessing import normalize
from ReadIpsData import *
tf.set_random_seed(777)  # reproducibility
np.random.seed(7)



savepath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\save" 
#basepath = r"F:\IPSData"
basepath = r"F:\IPSData2"
thckpath = r"F:\KLAData\ThicknessData"
ipsdata = CollectorIPSData(basepath , thckpath)
xs , ys , ps , ns = ipsdata.ReadMulti_RflcThck_Norm([2,4])

savepath = r"F:\Program\DataAnalysis\DataAnalysis\DataAnalysis\save" 

X2= xs[0][: , 450:850]
y2= ys[0]
y2 = y2.reshape( y2.shape[0], 1)
X3= xs[1][: , 450:850]
y3= ys[1]
y3 = y3.reshape( y3.shape[0], 1)

p2 = ps[0]
p3 = ps[1]

iSize = X2.shape[1]

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
saver.restore(sess , os.path.join(savepath , 'Best_model_LR.ckpt')) 


#Start Check

Predict2 , Label2 = sess.run([Output , Y] , feed_dict = { X : X2 , Y : y2 })
Predict3 , Label3 = sess.run([Output , Y] , feed_dict = { X : X3 , Y : y3 })
 
Predict2 = ns[0]['Y'].DeNormalization(Predict2)    
Label2 = ns[0]['Y'].DeNormalization(Label2)    
Predict3 = ns[1]['Y'].DeNormalization(Predict3)    
Label3 = ns[1]['Y'].DeNormalization(Label3)

Predict = np.concatenate((Predict2,Predict3))
Label = np.concatenate((Label2,Label3))
Position = np.concatenate((p2,p3))

pre = np.ndarray.flatten( Predict)
lb = np.ndarray.flatten( Label )

core = np.corrcoef( pre , lb )[0,1]

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
Position.tofile( os.path.join(savepath, "Point.csv"), sep=',')

fit = np.polyfit(pre, lb, deg=1)

posdigit = 12
digit = 18

stream = open( os.path.join(savepath , "Resulet.csv") ,'w')
wr = csv.writer(stream , quoting=csv.QUOTE_ALL)

resList = []
print("#2 Sample")
for i in range(0,pre2.shape[0]):
    res = "Pos : {0:<{n1}}  | Predict  : {1:<{n}}  |  Target {2:<{n}}".format(  p2[i],pre2[i], lb2[i] , n1 = digit,  n  = digit )
    resList.append(res)
    print(res)
    wr.writerow([res])
  
print()
print("#3 Sample")
for i in range(0,pre3.shape[0]):
    res  = "Pos : {0:<{n1}}  | Predict  : {1:<{n}}  |  Target {2:<{n}}".format(  p3[i],pre3[i], lb3[i] , n1 = digit,  n  = digit )
    resList.append(res)
    print(res)
    wr.writerow([res])

plt.title( "Correlation = {0} ,  (Red : #2 , Blue : #3) ".format(core) )
plt.xlabel( "Predict" )
plt.ylabel( "Target" )
plt.scatter(pre2 , lb2 , c = 'r' , alpha = 0.6 )
plt.scatter(pre3 , lb3  , alpha = 0.6)
plt.plot( pre , fit[0]*pre + fit[1] , color = 'green')
plt.xlim(280,340)
plt.ylim(280,340)
#plt.plot(LossTest , 'b' )

for label,x,y,i,pos in zip(Position,pre,lb,range(0,len(lb)),Position ):
    
    posstr = pos.split(" ")
    posx = float(posstr[0])
    posy = float(posstr[1])
    rho  = ( posx**2+posy**2 )**0.5

    if rho < 9 and posx > 0:
        plt.annotate(
            label,
            xy = (x,y),
            xytext=(30*(i/60),-30*(i/60)),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    if rho > 9 and posx > 0 and posy > 0:
        plt.annotate(
            label,
            xy = (x,y),
            xytext=(-30*(i/60),30*(i/60)),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.2),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()
print()
 









