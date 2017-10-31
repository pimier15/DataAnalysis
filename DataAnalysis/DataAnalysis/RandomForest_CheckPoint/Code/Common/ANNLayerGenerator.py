import tensorflow as tf 
from collections import OrderedDict
from enum import Enum

class LossType(Enum):
    MSE = 1
    CrossEntropy = 2
        

class ANN:
    def __init__(self, inputSize, hiddenSizeList , outputSize):
        self.inputSize = inputSize
        self.hSizeList = hiddenSizeList
        self.outputSize = outputSize
        self.layerkeys = []
        self.outputkeys = []
        self.X = tf.placeholder( tf.float32 , [None,inputSize])
        self.Y = tf.placeholder( tf.float32  , [None , outputSize])

    def CreatLayer(self):
        res = len(self.hSizeList)+1
        for i in range(0,len(self.hSizeList)+1):
            self.layerkeys.append("W" + str(i))
            self.layerkeys.append("b" + str(i))
            self.outputkeys.append("L"+ str(i))

        sizeList = [self.inputSize] + self.hSizeList + [self.outputSize]

        sizeDict = OrderedDict()
        for i in range(0,len(sizeList)-1):
            sizeDict[self.layerkeys[2*i]] = [ sizeList[i],sizeList[i+1] ]
            sizeDict[self.layerkeys[2*i+1]] = [ sizeList[i+1] ]

        layers = OrderedDict()
        for key in self.layerkeys:
            initdata = tf.random_normal( sizeDict[key] ) 
            layers[key] = tf.Variable(initdata , name = key )
        return layers


    def CombineLayer(self,layer,activations = None):
        if activations == None:
            return self.CombineLayerNoActivation(layer)
        else:
            return self.CombineLayerActivation(layer,activations)

    def CombineLayerNoActivation(self,layer):
        if len(self.outputkeys) < 2:
            print(" Create Layer First Please")
            return None

        outputList = OrderedDict()
        for i in range(0,len(self.layerkeys)):
            if i == 0:
                outputList[self.layerkeys[i]] =  tf.matmul(self.X , layers[2*i]) + layers[2*i+1] 
            outputList[self.layerkeys[i]] = tf.matmul(outputList[self.layerkeys[i-1]] , layers[2*i]) + layers[2*i+1] 
            
        return outputList
        pass

    def CombineLayerActivation(self,layers,activations):
        if len(self.outputkeys) < 2:
            print(" Create Layer First Please")
            return None

        outputList = OrderedDict()
        for i in range(0,len(self.layerkeys)):
            if i == 0:
                outputList[self.layerkeys[i]] = activations[i]( tf.matmul(self.X , layers[2*i]) + layers[2*i+1] )
            outputList[self.layerkeys[i]] = activations[i]( tf.matmul(outputList[self.layerkeys[i-1]] , layers[2*i]) + layers[2*i+1] )
            
        return outputList

    def LossLayer(self,losstype,output , target):
        if losstype == LossType().MSE:
            return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, output))))
        elif losstype == LossType().CrossEntropy:
            return tf.reduce_mean( - tf.reduce_sum( target * tf.log( output ) , reduction_indices = [1] ) ) 
        else :
            print("This LossType is not supported")
            return None

if __name__ == "__main__":
    
    
    ann = ANN(30,[50,60,40],10)

    ann.CreatLayer()



