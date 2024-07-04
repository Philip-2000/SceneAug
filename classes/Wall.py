from . import LINKS
from . import WALLS
import numpy as np

#WALLS=[]
class wall():
    def __init__(self, p, q, n, w1, w2, idx):
        assert abs((p[0]-q[0])*n[0]+(p[2]-q[2])*n[2]) < 0.01
        assert (p[0]-q[0])*n[2]<=(p[2]-q[2])*n[0]
        self.linkIndex=[]
        self.idx=idx
        self.p=np.copy(p)
        self.q=np.copy(q)
        self.n=np.copy(n)
        self.length=(((self.p-self.q)**2).sum())**0.5
        self.w1=w1
        self.w2=w2

    def lengthh(self):
        self.length=(((self.p-self.q)**2).sum())**0.5

    def mWall(self,L): 
        oldp = np.copy(self.p)
        self.p += self.n*L
        WALLS[self.w1].adjustWall(oldp,self.p,self.idx)

        oldq = np.copy(self.q)
        self.q += self.n*L
        WALLS[self.w2].adjustWall(oldq,self.q,self.idx)
        
        for i in self.linkIndex:
            LINKS[i].adjust(self.n*L)
        
    def adjustWall(self,oldp,p,hint=-1):
        oldn = self.n
        if hint<0:
            if ((self.p-oldp)**2).sum()<0.001:
                oldq = self.q
                self.p=p
            elif ((self.q-oldp)**2).sum()<0.001:
                oldq = oldp
                oldp = self.p
                self.q=p
            else:
                print("adjustWall error")
                return
        else:
            if hint == self.w1:
                oldq = self.q
                self.p=p
            elif hint == self.w2:
                oldq = oldp
                oldp = self.p
                self.q=p
            else:
                print("false hint")
                return
        if (self.p-self.q)[0]*self.n[2] > (self.p-self.q)[2]*self.n[0]:
            self.n = -self.n
        self.length=(((self.p-self.q)**2).sum())**0.5

        for i in self.linkIndex:
            LINKS[i].modify(oldp, oldq, oldn)
        pass

def breakWall(id,rate):
    #load all the links of 
    
    delList = []
    for l in WALLS[id].linkIndex:
        r = LINKS[l].rate
        if r < rate:
            LINKS[l].rate = r / rate
        else:
            LINKS[l].src = len(WALLS)+1
            LINKS[l].rate = (r-rate) / (1-rate)
            delList.append(l)
    for l in delList:
        WALLS[id].linkIndex.remove(l)
    #WALLS[id].linkIndex

    cutP = rate*WALLS[id].q + (1-rate)*WALLS[id].p
    A = len(WALLS)
    WALLS.append(
        wall(cutP,cutP,np.cross(WALLS[id].n,np.array([0,1,0])),id,A+1,A)
    )
    WALLS.append(
        wall(cutP,WALLS[id].q,WALLS[id].n,A,WALLS[id].w2,A+1)
    )
    for l in delList:
        WALLS[A+1].linkIndex.append(l)
    WALLS[WALLS[id].w2].w1= A+1
    WALLS[id].q = np.copy(cutP)
    WALLS[id].w2= A
    pass