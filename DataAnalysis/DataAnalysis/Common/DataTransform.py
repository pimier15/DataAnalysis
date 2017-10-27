import numpy as np

class Normalizer:

    def __init__(self):
        self.meanList = None
        self.stdList = None

    def Normalization(self , datas):
        if datas.ndim == 1 :
            self.meanList = np.mean( datas )
            self.stdList = np.std( datas )
        else :
            self.meanList = np.mean( datas , axis = 1)
            self.stdList = np.std( datas , axis = 1 )
        return (( datas.T - self.meanList ) / self.stdList).T

    def DeNormalization(self, datas):
        return (( datas.T * self.stdList + self.meanList ) ).T
