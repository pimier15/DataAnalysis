import numpy as np 
import os
from functools import reduce
from os.path import join
from DataTransform import Normalizer
from collections import OrderedDict

#Complete 2017 10 27
class CollectorIPSData:
    def __init__(self, baseDir , thcknessPath , startdata = 1 , enddata = 26):
        self.BaseDirPath = baseDir
        self.ThckDirPath = thcknessPath
        self.startData = startdata
        self.endData = enddata 

    ##########################################
    ## Read Multi Data into list

    def ReadMulti_RflcThck(self, trgNumList):
        xlist = []
        ylist = []
        plist = []

        for i in trgNumList:
            xi ,yi , pi = self.Read_RflcThck(i)
            xlist.append(xi)
            ylist.append(yi)
            plist.append(pi)
        return xlist , ylist , plist

    def ReadMulti_RflcThck_Norm(self, trgNumList):
        xlist = []
        ylist = []
        plist = []
        nlist = []

        for i in trgNumList:
            xi ,yi , pi , normi = self.Read_RflcThck_Norm(i)
            xlist.append(xi)
            ylist.append(yi)
            plist.append(pi)
            nlist.append(normi)
        return xlist , ylist , plist , nlist

    ###########################################

      #Use this method for Main. Get all Datas include subdirectory
    def Read_RflcThck(self,trgNum ):
        pathlist = self.GetAllRflctPath(trgNum,self.BaseDirPath)
        tcknessList = self.GetTicknessPath(trgNum,self.ThckDirPath)
        Xtemp = []
        Ytemp = []
        Pointtemp = []

        for path in pathlist:
            pos , datas = self.ReadReflectivity(path)
            tckness = self.ReadThickness(tcknessList)
            Xtemp.append( datas )
            Ytemp.append( tckness)
            Pointtemp.append( pos )

        x = reduce( lambda f,s : np.concatenate((f,s) , axis = 1) , Xtemp)
        y = reduce( lambda f,s : np.concatenate((f,s) , axis = 0) , Ytemp)
        pos = reduce( lambda f,s : np.concatenate((f,s) , axis = 0) , Pointtemp)
         
        return x.T , y , pos

    def Read_RflcThck_Norm(self, trgNum ):
        x , y, pos = self.Read_RflcThck(trgNum)
        normDict ={}
        normDict['X'] = Normalizer()
        normDict['Y'] = Normalizer()

        xnorm = normDict['X'].Normalization(x)
        ynorm = normDict['Y'].Normalization(y)
        
        return xnorm , ynorm , pos , normDict


    #single Refelectivity file Reader
    # need full path on name or subdir path from base path need to be include
    def ReadReflectivity(self, name ,  path = None ,isFullPath = False):
        path = path if path is not None else self.BaseDirPath
        if not isFullPath:
            name = join(path , name)

        pos = np.loadtxt(name , dtype = np.str , delimiter = ',' , usecols = range(self.startData,self.endData) )[0]
        datas = np.loadtxt(name ,  dtype = np.float32 , delimiter = ',' , skiprows = 1 , usecols = range(self.startData,self.endData)  )
        return pos , datas

    #single Thickness file Reader
    def ReadThickness(self,name , path = None , isFullPath = False):
        path = path if path is not None else self.ThckDirPath
        if not isFullPath:
            name = join(path , name)
        thickness = np.loadtxt(name ,  dtype = np.float32 , delimiter = ',' , skiprows = 1 , usecols = 2  )[:25]
        return thickness


    

    #### Helper Function ####

    def GetSubDirFiles(self , nameFilter):
        sublist = os.listdir(self.BaseDirPath)
        for filename in sublist:
            fullname = join(self.BaseDirPath , filename)
            print(fullname)

    def GetAllRflctPath(self,trgNum,basePath = None ):
        basePath = basePath if basePath is not None else self.BaseDirPath
        fileNames = []
        trgN = str(trgNum)
        for (path , dir , filse ) in os.walk(basePath):
            for filename in  filse:
                key = filename.split('_')
                num = key[0].split('-')[0]
                type = key[1]
                
                if ( num == trgN and type == "Refelctivity.csv" ):
                    fileNames.append( os.path.join(path , filename) )
        return fileNames

    def GetTicknessPath(self,trgNum,basePath = None):
        basePath = basePath if basePath is not None else self.ThckDirPath
        fileNames = []
        trgN = str(trgNum)
        for (path , dir , filse ) in os.walk(basePath):
            for filename in  filse:
                num , type= filename.split('-')
                
                if ( num == trgN and type == "KLAResult.csv" ):
                    return os.path.join(path , filename) 
        return None

if __name__ == "__main__":
    basepath = r"F:\IPSData"
    thckpath = r"F:\KLAData\ThicknessData"

    ips = CollectorIPSData(basepath , thckpath)

    #get single data
    nameRefl = r"2-1\2-1_Refelctivity.csv"
    nameThck = r"3-KLAResult.csv"

    pos , data = ips.ReadReflectivity(nameRefl)
    res2 = ips.ReadThickness(nameThck)


    x1,y1,pos1 = ips.Read_RflcThck(2)
    x11,y11,pos11,normlist = ips.Read_RflcThck_Norm(2)

    xs , ys , ps = ips.ReadMulti_RflcThck([2,3])
    xs , ys , ps , ns = ips.ReadMulti_RflcThck_Norm([2,3])
    print()


