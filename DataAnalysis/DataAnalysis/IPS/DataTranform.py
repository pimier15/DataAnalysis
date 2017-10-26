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


if __name__ == "__main__":

    data = np.array([[1,2,3],[2,4,5]])
    mean = np.mean(data ,axis = 1)
    std = np.std(data ,axis = 1)
    print()


    k = np.array([10,10,10])
    k2 = np.array([100,100])

    res1 = data + k
    res2 = data.T + k2
    res3 = res2.T
    

    nm = Normalizer()
    norm = nm.Normalization(data)
    denorm = nm.DeNormalization(norm)

    print()
        