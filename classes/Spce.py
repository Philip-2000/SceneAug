from . import SPCES
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

#WALLS=[]
class spce():
    def __init__(self,corner0,corner1,walls=None):
        self.corner0 = corner0
        self.corner1 = corner1
        self.c = (corner0+corner1)/2.0
        self.a = self.corner0 - self.c
        pass

    def maxZ(self):
        return max(self.corner0[2],self.corner1[2])

    def minZ(self):
        return min(self.corner0[0],self.corner1[0])
    
    def maxX(self):
        return max(self.corner0[2],self.corner1[2])
    
    def minX(self):
        return min(self.corner0[0],self.corner1[0])

    def absoluteBbox(self):
        pass

    def recommendedWalls(self):
        #we are going 
        pass

    def draw(self):
        scl = [1.0,0.8,0.6,0.4]
        c,a = self.c, self.a
        for s in scl:
            corners = np.array([[c[0]+s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]+s*a[2]]])
            plt.plot( corners[:,0], -corners[:,1], marker="x", color="pink")


def adjacent(s,Qbox):
    #还有一些问题就是如果t的展开是可量化的。那能不能直接以s为限制去给出t的量化指标。
    #比如说，t的一个端点是固定的，另一个端点还未限定，那么由s给出t的另一个端点的范围指标。
    #
    Actions = {}
    if (Qbox["signX"] > 0 and s.minX > Qbox["fixedX"] and Qbox["maxX"] > s.minX):
        Actions["maxX"] = s.minX

    if (Qbox["signX"] < 0 and s.maxX < Qbox["fixedX"] and Qbox["minX"] < s.maxX):
        Actions["minX"] = s.maxX

    if (Qbox["signZ"] > 0 and s.minZ > Qbox["fixedZ"] and Qbox["maxZ"] > s.minZ):
        Actions["maxZ"] = s.minZ

    if (Qbox["signZ"] < 0 and s.maxZ < Qbox["fixedZ"] and Qbox["minZ"] > s.maxZ):
        Actions["minZ"] = s.maxZ

    if len(Actions.keys())>0:
        K = Actions.keys()[0]
        Square = -1
        for k in Actions.keys():
            Pbox = deepcopy(Qbox)
            Pbox[k] = Actions[k]
            square = (Pbox["maxX"]-Pbox["minX"])*(Pbox["maxX"]-Pbox["minX"])
            if square > Square:
                Square = square
                K = k
        Qbox[K] = Actions[K]

    #{"maxX":???,"minX":???,"maxZ":???, "minZ":???}
    return Qbox