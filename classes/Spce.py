from . import SPCES,WALLS
from .Wall import wall
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def stickWall(w,x):
    wp,wq,xp,xq = np.cross(np.abs(w.n),w.p)[1],np.cross(np.abs(w.n),w.q)[1],np.cross(np.abs(x.n),x.p)[1],np.cross(np.abs(x.n),x.q)[1]
    return w.v and abs(w.n@x.n)>0.9 and abs(np.abs(w.n)@w.p-np.abs(x.n)@x.p)<0.01 and min(wp,wq)<max(xp,xq) and min(xp,xq)<max(wp,wq)

class bond():
    def __init__(self,p,q,idx,spceIdx):
        self.p=p
        self.q=q
        self.n = np.cross(np.array([0,1,0]),p-q)/np.linalg.norm(np.cross(np.array([0,1,0]),p-q))
        #print(self.n) #assert (p[0]-q[0])*self.n[2]<=(p[2]-q[2])*self.n[0]
        self.length=(((self.p-self.q)**2).sum())**0.5
        self.idx=idx
        self.spceIdx=spceIdx
        self.relatedWalls = []
        self.full = False
        self.checkWalls()

    def addWall(self,w):
        #(self.p-w.p)@(w.q-w.p)/w.length  
        # print("addWall "+str(self.idx)+" "+str(w.idx))
        # print(self.p)
        # print(self.q)
        # print(self.n)
        # print(w.p)
        # print(w.q)
        # print(w.n)
        # print("low")
        # print((self.p-w.p)@(w.q-w.p)/(w.length)**2)

                

        low = max((self.p-w.p)@(w.q-w.p)/(w.length)**2,0.01)
        # print(low)
        # print("upp")
        # print((self.q-w.p)@(w.q-w.p)/(w.length)**2)
        upp = min((self.q-w.p)@(w.q-w.p)/(w.length)**2,0.99)
        # print(upp)
        # print("lower")
        # print((w.p-self.p)@(self.q-self.p)/(self.length)**2)
        lower = max((w.p-self.p)@(self.q-self.p)/(self.length)**2,0.01)
        # print(lower)
        # print("upper")
        # print((w.q-self.p)@(self.q-self.p)/(self.length)**2)
        upper = min((w.q-self.p)@(self.q-self.p)/(self.length)**2,0.99)
        # print(upper)
        self.relatedWalls.append({"idx":w.idx,"low":low,"upp":upp,"lower":lower,"upper":upper})
        if low==0.01 and upp==0.99 and lower==0.01 and upper==0.99 :
            self.full=True

    def checkWalls(self):
        for w in WALLS:
            if stickWall(w,self):
                self.addWall(w)
        pass

    def mWall(self, L, moveFull=True):
        oldp = np.copy(self.p)
        self.p += self.n*L
        oldq = np.copy(self.q)
        self.q += self.n*L
        self.length=(((self.p-self.q)**2).sum())**0.5
        SPCES[self.spceIdx].bounds[(self.idx-1)%4].adjustWall(oldp,self.p,self.idx)
        SPCES[self.spceIdx].bounds[(self.idx+1)%4].adjustWall(oldq,self.q,self.idx)
        if moveFull and self.full:
            WALLS[self.relatedWalls[0]["idx"]].mWall(L)

    def adjustWall(self,oldp,p,hint):
        if hint == (self.idx-1)%4:
            self.p=p
        elif hint == (self.idx+1)%4:
            self.q=p
        else:
            print("false hint")
        if (self.p-self.q)[0]*self.n[2] > (self.p-self.q)[2]*self.n[0]:
            self.n = -self.n
        self.length=(((self.p-self.q)**2).sum())**0.5

#WALLS=[]
class spce():
    def __init__(self,c0,c1,idx=-1):
        self.c0 = np.min(np.array([c0,c1]),axis=0)
        self.c1 = np.max(np.array([c0,c1]),axis=0)
        self.c = (c0+c1)/2.0
        self.a = self.c0 - self.c
        
        self.bounds = []
        self.bounds.append(bond(self.c0,np.array([self.c1[0],self.c1[1],self.c0[2]]),0,idx))
        self.bounds.append(bond(np.array([self.c1[0],self.c1[1],self.c0[2]]),self.c1,1,idx))
        self.bounds.append(bond(self.c1,np.array([self.c0[0],self.c0[1],self.c1[2]]),2,idx))
        self.bounds.append(bond(np.array([self.c0[0],self.c0[1],self.c1[2]]),self.c0,3,idx))
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

    def adjustSingle(self,id):
        if id > 3 or (not self.bounds[id].full):
            print("error:noFullBound?")
            return 
        L = 0
        if self.bounds[0].length/self.bounds[1].length < 0.6:
            if id%2 == 0:
                L = self.bounds[1].length - self.bounds[0].length/0.6 #positive
                #reduce myself to reduce self.bounds[1].length
                pass
            else:
                L = self.bounds[0].length - self.bounds[1].length*0.6 #negative
                #add myself to add self.bounds[0].length
                pass
        else:#self.bounds[0].length/self.bounds[1].length > 1.6
            if id%2 == 0:
                L = self.bounds[1].length - self.bounds[0].length/1.6 #negative
                #add myself to add self.bounds[1].length
                pass
            else:
                L = self.bounds[0].length - self.bounds[1].length*1.6 #positive
                #reduce myself to reduce self.bounds[0].length
                pass
            pass 
        self.bounds[id].mWall(L)

    def adjustSize(self):
        if abs(self.bounds[0].length/self.bounds[1].length-1.0)<0.4:
            return
        if (self.bounds[0].full or self.bounds[2].full) and (self.bounds[1].full or self.bounds[3].full):
            if(self.bounds[0].length>self.bounds[1].length):
                if self.bounds[0].full:
                    self.adjustSingle(0)
                elif self.bounds[2].full:
                    self.adjustSingle(2)
            else:
                if self.bounds[1].full:
                    self.adjustSingle(1)
                elif self.bounds[3].full:
                    self.adjustSingle(3)
        else:
            id=0
            while id < 4 and (not self.bounds[id].full):
                id += 1
            self.adjustSingle(id)

        #check my own size,
        #find an full-covered wall
        #drag that wall to a propriate place
        # self.c0 = self.bounds[0].p
        # self.c1 = self.bounds[1].q
        # self.c = (self.c0+self.c1)/2.0
        # self.a = self.c0 - self.c


    def draw(self):
        scl = [1.0,0.8,0.6,0.4]
        c,a = self.c, self.a
        for s in scl:
            corners = np.array([[c[0]+s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]+s*a[2]],[c[0]-s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]-s*a[2]],[c[0]+s*a[0],c[2]+s*a[2]]])
            plt.plot( corners[:,0], -corners[:,1], marker="x", color="pink")


# def adjacent(s,Qbox):
#     #还有一些问题就是如果t的展开是可量化的。那能不能直接以s为限制去给出t的量化指标。
#     #比如说，t的一个端点是固定的，另一个端点还未限定，那么由s给出t的另一个端点的范围指标。
#     #
#     Actions = {}
#     if (Qbox["signX"] > 0 and s.minX() > Qbox["fixedX"] and Qbox["maxX"] > s.minX()):
#         Actions["maxX"] = s.minX()

#     if (Qbox["signX"] < 0 and s.maxX() < Qbox["fixedX"] and Qbox["minX"] < s.maxX()):
#         Actions["minX"] = s.maxX()

#     if (Qbox["signZ"] > 0 and s.minZ() > Qbox["fixedZ"] and Qbox["maxZ"] > s.minZ()):
#         Actions["maxZ"] = s.minZ()

#     if (Qbox["signZ"] < 0 and s.maxZ() < Qbox["fixedZ"] and Qbox["minZ"] > s.maxZ()):
#         Actions["minZ"] = s.maxZ()

#     if len(Actions.keys())>0:
#         K = Actions.keys()[0]
#         Square = -1
#         for k in Actions.keys():
#             Pbox = deepcopy(Qbox)
#             Pbox[k] = Actions[k]
#             square = (Pbox["maxX"]-Pbox["minX"])*(Pbox["maxX"]-Pbox["minX"])
#             if square > Square:
#                 Square = square
#                 K = k
#         Qbox[K] = Actions[K]

#     #{"maxX":???,"minX":???,"maxZ":???, "minZ":???}
#     return Qbox