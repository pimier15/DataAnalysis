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
            self.meanList = np.mean( datas , axis = 0)
            self.stdList = np.std( datas , axis = 0)
        return (( datas - self.meanList ) / self.stdList)

    def DeNormalization(self, datas):
        return (( datas.T * self.stdList + self.meanList ) ).T


if __name__ == "__main__":
    a = [[1,2],[3,4],[1,7] ]
    b = np.array(a)
    
    norm = Normalizer()
    res = norm.Normalization(b)
    print(res)