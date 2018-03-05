import numpy as np
import csv
from os.path import join
import os
from functools import *

class ReadIPS:
    def __init__(self, baseDir , onlyroot = False , startdata = 1 , enddata = 26):
        self.BasePath = baseDir
        self.startData = startdata
        self.endData = enddata = enddata

    def ReadReflectivity(self,name , isFullPath = True):
        if not isFullPath:
            name = join(self.BasePath , name)

        pos = np.loadtxt(name , dtype = np.str , delimiter = ',' , usecols = range(self.startData,self.endData) )[0]
        datas = np.loadtxt(name ,  dtype = np.float32 , delimiter = ',' , skiprows = 1 , usecols = range(self.startData,self.endData)  )
        return pos , datas

    def ReadReflectivity(self,name):
        name = join(self.BasePath , name)
        pos = np.loadtxt(name , dtype = np.str , delimiter = ',' , usecols = range(self.startData,self.endData) )[0]
        datas = np.loadtxt(name ,  dtype = np.float32 , delimiter = ',' , skiprows = 1 , usecols = range(self.startData,self.endData)  )
        return pos , datas

    def ReadThickness(self,name , isFullPath = True):
        if not isFullPath:
            name = join(self.BasePath , name)
        thickness = np.loadtxt(name ,  dtype = np.float32 , delimiter = ',' , skiprows = 1 , usecols = 2  )[:25]
        return thickness

    def GetSubDirFiles(self , nameFilter):
        sublist = os.listdir(self.BasePath)
        for filename in sublist:
            fullname = join(self.BasePath , filename)
            print(fullname)

    def GetAllRflctPath(self,trgNum):
        fileNames = []
        trgN = str(trgNum)
        for (path , dir , filse ) in os.walk(self.BasePath):
            for filename in  filse:
                key = filename.split('_')
                num = key[0].split('-')[0]
                type = key[1]
                
                if ( num == trgN and type == "Refelctivity.csv" ):
                    fileNames.append( os.path.join(path , filename) )
        return fileNames

    def GetTicknessPath(self,basePath,trgNum):
        fileNames = []
        trgN = str(trgNum)
        for (path , dir , filse ) in os.walk(basePath):
            for filename in  filse:
                num , type= filename.split('-')
                
                if ( num == trgN and type == "KLAResult.csv" ):
                    return os.path.join(path , filename) 
        return None

    def GetAllReflcThick(self,trgNum, thickPath):
        pathlist = self.GetAllRflctPath(trgNum)
        tcknessList = self.GetTicknessPath(thickPath,trgNum)
        Xtemp = []
        Ytemp = []
        Pointtemp = []

        for path in pathlist:
            pos , datas = self.ReadReflectivity(path)
            tckness = self.ReadThickness(tcknessList)
            Xtemp.append( datas )
            Ytemp.append( tckness)
            Pointtemp.append( pos )

        xdatas = reduce( lambda f,s : np.concatenate((f,s) , axis = 1) , Xtemp)
        ydatas = reduce( lambda f,s : np.concatenate((f,s) , axis = 0) , Ytemp)
        pdatas = reduce( lambda f,s : np.concatenate((f,s) , axis = 0) , Pointtemp)
         
        return xdatas.T , ydatas , pdatas
        


if __name__ == "__main__":
    ips = ReadIPS(r"F:\IPSData")
    thicknessPath = r"F:\KLAData\ThicknessData"

    pathlist = ips.GetAllRflctPath(2)
    tcknessList = ips.GetTicknessPath(thicknessPath,2)
    a = 'F:\\KLAData\\ThicknessData\\2-KLAResult.csv'

    Xtemp = []
    Ytemp = []

    for path in pathlist:
        _ , datas = ips.ReadReflectivity(path)
        tckness = ips.ReadThickness(tcknessList)
        Xtemp.append( datas )
        Ytemp.append( tckness)

    xdatas = reduce( lambda f,s : np.concatenate((f,s) , axis = 1) , Xtemp)
    ydatas = reduce( lambda f,s : np.concatenate((f,s) , axis = 0) , Ytemp)


    print()
        

    x , y =  ips.GetAllReflcThick(2,thicknessPath)

    print()

    

         


    