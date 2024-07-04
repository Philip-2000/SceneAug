from . import OBJES
import numpy as np

def fTheta(theta):
    while theta > np.math.pi:
        theta -= 2*np.math.pi
    while theta < -np.math.pi:
        theta += 2*np.math.pi
    return theta

def matrix(ori):
    return np.array([[np.math.cos(ori),0,np.math.sin(ori)],[0,1,0],[-np.math.sin(ori),0,np.math.cos(ori)]])

#WALLS=[]
class grup():
    def __init__(self, objIdList):
        self.objIdList = objIdList
        cs = np.array([OBJES[i].corners2() for i in objIdList]).reshape((-1,2))
        Xs,Zs = cs[:,0],cs[:,1]
        self.translation = [(np.min(Xs)+np.max(Xs))/2.0,0,(np.min(Zs)+np.max(Zs))/2.0]
        self.orientation = [1,0,0]
        pass

    def bbox2(self):
        cs = np.array([OBJES[i].corners2() for i in self.objIdList]).reshape((-1,2))
        return [np.min(cs,dim=0), np.max(cs,dim=1)]

    def adjust(self, t, o):
        rTrans = {}
        for i in self.objIdList:
            rTrans[i] = [matrix(-self.orientation)@(OBJES[i].translation-self.translation),fTheta(OBJES[i].orientation-self.orientation)]
        self.translation, self.orientation=t,o
        for i in self.objIdList:
            OBJES[i].setTransformation(matrix(o)@rTrans[i][0]+t,fTheta(rTrans[i][1]+o))

    def recommendedWalls(self):
        #we are going 
        pass